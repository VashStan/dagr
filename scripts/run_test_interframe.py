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
                                                                         name="test111", compile_detections=True,
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

        save_detections(output_directory, detections)
        """
        保存最终的检测结果
        """