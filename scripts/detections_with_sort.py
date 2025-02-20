import torch
import tqdm
import wandb
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

from torch_geometric.data import DataLoader
from pprint import pprint

from dagr.utils.logging import set_up_logging_directory, log_hparams
from dagr.utils.args import FLAGS
from dagr.utils.testing import run_test_with_visualization

from dagr.data.augment import Augmentations
from dagr.data.dsec_data import DSEC

from dagr.model.networks.dagr import DAGR
from dagr.model.networks.ema import ModelEMA


def to_npy(detections):
    n_boxes = len(detections['boxes'])
    dtype = np.dtype([('t', '<u8'), ('x', '<f4'), ('y', '<f4'), ('w', '<f4'), ('h', '<f4'), ('class_id', 'u1'), ('class_confidence', '<f4')])
    data = np.zeros(shape=(n_boxes,), dtype=dtype)
    data['t'] = detections['t']
    data['x'] = detections['boxes'][:,0]
    data['y'] = detections['boxes'][:,1]
    data['w'] = detections['boxes'][:,2] - data['x']
    data['h'] = detections['boxes'][:,3] - data['y']
    data['class_id'] = detections['labels']
    data['class_confidence'] = detections['scores']
    return data

def save_detections(directory, detections):
    sequence_detections_map = dict()
    for d in tqdm.tqdm(detections, desc="compiling detections for saving..."):
        s = d['sequence']
        if s not in sequence_detections_map:
            sequence_detections_map[s] = to_npy(d)
        else:
            sequence_detections_map[s] = np.concatenate([sequence_detections_map[s], to_npy(d)])

    for s, detections in sequence_detections_map.items():
        detections = detections[detections['t'].argsort()]
        np.save(directory / f"detections_{s}.npy", detections)

def convert_bbox_to_z(bbox):
    """
    将边界框 (x1, y1, x2, y2) 格式转换为卡尔曼滤波的状态表示形式 (x, y, s, r)，
    其中 x,y 是中心坐标，s 是比例（面积），r 是宽高比
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    将卡尔曼滤波的状态输出转换回边界框 (x1, y1, x2, y2) 格式，
    如果有得分 score 也可以一并返回带有得分的边界框格式
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    x1 = x[0] - w / 2.
    y1 = x[1] - h / 2.
    x2 = x[0] + w / 2.
    y2 = x[1] + h / 2.
    if score is None:
        return np.array([x1, y1, x2, y2]).reshape((1, 4))
    else:
        return np.array([x1, y1, x2, y2, score]).reshape((1, 5))


def convert_detections(detections):
    """
    将给定格式的检测结果（以字典形式表示每个检测）转换为SORT算法所需的格式（边界框坐标(x1, y1, x2, y2) 以及 score）
    :param detections: 原始检测结果，格式为包含字典元素的列表，字典元素包含('t', 'x', 'y', 'w', 'h', 'class_id', 'class_confidence')这些键
    :return: 转换后的检测结果，格式为N x 5的数组，每行表示 (x1, y1, x2, y2, score)
    """
    converted_detections = []
    for det in detections:
        x1 = det['x']
        y1 = det['y']
        x2 = det['x'] + det['w']
        y2 = det['y'] + det['h']
        score = det['class_confidence']
        class_id = det['class_id']
        converted_detections.append([x1, y1, x2, y2, score,class_id])
    return np.array(converted_detections)

from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

class Track:
    def __init__(self, bbox, track_id):
        """
        初始化一个跟踪目标
        :param bbox: 初始边界框 (x1, y1, x2, y2)
        :param track_id: 跟踪目标的唯一ID
        """
        self.track_id = track_id
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0, 0],
                              [0, 1, 0, 0, 0, 1, 0, 0],
                              [0, 0, 1, 0, 0, 0, 1, 0],
                              [0, 0, 0, 1, 0, 0, 0, 1],
                              [0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P *= 10.

        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)

        self.time_since_update = 0
        self.hits = 1
        self.total_visible_count = 1

    def predict(self):
        """
        使用卡尔曼滤波进行状态预测
        """
        self.kf.predict()
        self.time_since_update += 1

    def update(self, bbox):
        """
        根据新的观测值（边界框）更新卡尔曼滤波状态
        """
        self.kf.update(convert_bbox_to_z(bbox))
        self.time_since_update = 0
        self.hits += 1
        self.total_visible_count += 1

def sort_tracker(detections, max_age=1, min_hits=3):
    """
    SORT算法主函数，对输入的检测结果进行目标跟踪，加入类别ID处理
    :param detections: 检测结果，格式为 N x 6 的数组，每行表示 (x1, y1, x2, y2, score, class_id)，这里假设加入了class_id字段
    :param max_age: 目标丢失后最大存活帧数，超过这个帧数则删除该跟踪目标
    :param min_hits: 认定一个跟踪目标有效的最小命中次数
    :return: 跟踪结果，格式为 M x 6 的数组，每行表示 (x1, y1, x2, y2, track_id, class_id)
    """
    trackers = []
    track_id_count = 0
    result = []

    for frame_detections in detections:
        # 预测当前帧中所有跟踪目标的状态
        for tracker in trackers:
            tracker.predict()

        # 将检测结果转换为合适的格式，同时提取类别ID
        detections_ = np.array([det[:4] for det in frame_detections])
        scores = np.array([det[4] for det in frame_detections])
        class_ids = np.array([det[5] for det in frame_detections])

        # 构建代价矩阵，用于匈牙利算法匹配，这里先只考虑位置匹配，后续可加入类别匹配逻辑
        cost_matrix = np.zeros((len(trackers), len(detections_)))
        for i, tracker in enumerate(trackers):
            for j, det in enumerate(detections_):
                z = convert_bbox_to_z(det)
                pred = tracker.kf.x[:4]
                cost_matrix[i, j] = np.linalg.norm(z - pred)

        # 加入类别匹配逻辑，这里简单示例，如果跟踪目标类别和检测目标类别不一致，增加一个较大的代价惩罚
        for i, tracker in enumerate(trackers):
            for j, det in enumerate(detections_):
                if tracker.class_id!= det[5]:
                    cost_matrix[i, j] += 100  # 较大的惩罚值，可根据实际调整

        # 通过匈牙利算法进行匹配
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # 匹配成功的进行更新跟踪目标状态
        matched_trackers = []
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] > 5:
                continue
            trackers[r].update(detections_[c])
            matched_trackers.append(r)
            result.append(np.concatenate((convert_x_to_bbox(trackers[r].kf.x),
                                          np.array([trackers[r].track_id])[np.newaxis].T,
                                          np.array([trackers[r].class_id])[np.newaxis].T), axis=1))

        # 处理未匹配的跟踪目标（可能丢失）
        unmatched_trackers = list(set(range(len(trackers))) - set(matched_trackers))
        for idx in unmatched_trackers:
            if trackers[idx].time_since_update > max_age:
                del trackers[idx]

        # 处理未匹配的检测结果（新出现目标）
        unmatched_detections = list(set(range(len(detections_))) - set(col_ind))
        for idx in unmatched_detections:
            track_id_count += 1
            new_track = Track(detections_[c], track_id_count, class_ids[idx])  # 传入类别ID初始化新跟踪目标
            trackers.append(new_track)

    return np.array(result)


if __name__ == '__main__':
    import torch_geometric
    import random
    import numpy as np

    seed = 42
    torch_geometric.seed.seed_everything(seed)
    torch.random.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    args = FLAGS()

    output_directory = set_up_logging_directory(args.dataset, args.task,  args.output_directory)
    log_hparams(args)

    print("init datasets")
    test_dataset = DSEC(root=args.dataset_directory, split="test", transform=Augmentations.transform_testing,
                        debug=False, min_bbox_diag=15, min_bbox_height=10, only_perfect_tracks=True,
                        no_eval=args.no_eval)
    test_loader = DataLoader(test_dataset, follow_batch=['bbox', "bbox0"], batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)

    print("init net")
    model = DAGR(args, height=test_dataset.height, width=test_dataset.width)
    model = model.cuda()
    ema = ModelEMA(model)

    assert "checkpoint" in args
    checkpoint = torch.load(args.checkpoint)
    ema.ema.load_state_dict(checkpoint['ema'])
    ema.ema.cache_luts(radius=args.radius, height=test_dataset.height, width=test_dataset.width)

    detections = []
    """
    初始化一个空列表 `detections` 用于存储检测结果
    """
    with torch.no_grad():
        """
        在不计算梯度的上下文环境中
        """
        for n_us in np.linspace(0, 50000, args.num_interframe_steps):
            """
            遍历由 `np.linspace` 生成的一系列数值 `n_us`
            """
            test_loader.dataset.set_num_us(int(n_us))
            """
            设置测试数据集中的某个参数
            """
            # metrics, detections_one_offset = run_test_with_visualization(test_loader, ema.ema, dataset=args.dataset, name=wandb.run.name, compile_detections=True,
            metrics, detections_one_offset = run_test_with_visualization(test_loader, ema.ema, dataset=args.dataset,
                                                                         name="test003", compile_detections=True,
                                                                         no_eval=args.no_eval)
            """
            运行带有可视化的测试函数，并获取评估指标 `metrics` 和当前偏移量下的检测结果 `detections_one_offset`
            """
            detections.extend(detections_one_offset)
            """
            将当前偏移量下的检测结果添加到 `detections` 列表中
            """

            if metrics is not None:
                pprint(f"Time Window: {int(n_us)} ms \t mAP: {metrics['mAP']}")
                """
                如果评估指标存在，打印时间窗口和平均精度（mAP）
                """

        # 在此处通过detection进行sort算法
        # detections.dtype.names ('t', 'x', 'y', 'w', 'h', 'class_id', 'class_confidence')
        detections = convert_detections(detections)
        tracking_results = sort_tracker(detections)
        print(tracking_results)

        save_detections(output_directory, detections)
        """
        保存最终的检测结果
        """