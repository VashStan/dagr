import torch
from dagr.utils.logging import log_bboxes
from dagr.utils.buffers import DetectionBuffer, format_data
import tqdm

def to_npy(detections):
    """
    用途：将输入的检测结果中的值转换为 NumPy 数组，并移动到 CPU 上

    输入含义：
    - `detections`：一个包含字典的列表，每个字典中的值是张量

    输出含义：
    - 返回一个新的列表，其中每个字典中的值已转换为 NumPy 数组

    例子：
    如果 `detections = [{'key1': torch.Tensor([1, 2, 3]), 'key2': torch.Tensor([4, 5, 6])}]`
    那么 `to_npy(detections)` 可能返回 `[{'key1': np.array([1, 2, 3]), 'key2': np.array([4, 5, 6])}]`
    """
    return [{k: v.cpu().numpy() for k, v in d.items()} for d in detections]

def format_detections(sequences, t, detections):
    """
     用途：对检测结果进行格式化，包括将其值转换为 NumPy 数组，并添加 `sequence` 和 `t` 字段

     输入含义：
     - `sequences`：可能是与检测相关的序列数据
     - `t`：可能是时间相关的数据
     - `detections`：原始的检测结果

     输出含义：
     - 返回格式化后的检测结果列表

     例子：
     假设 `sequences = [10, 20]`，`t = [5, 6]`，`detections = [{'key': torch.Tensor([7, 8, 9])}]`
     那么 `format_detections(sequences, t, detections)` 可能返回 `[{'key': np.array([7, 8, 9]),'sequence': 10, 't': 5}]`
     """
    detections = to_npy(detections)
    for i, det in enumerate(detections):
        det['sequence'] = sequences[i]
        det['t'] = t[i]
    return detections

def run_test_with_visualization(loader, model, dataset: str, log_every_n_batch=-1, name="", compile_detections=False, no_eval=False):
    """
    定义了一个名为 `run_test_with_visualization` 的函数，它接受一些参数来执行带有可视化的测试操作

    参数:
    - `loader`：可能是数据加载器，用于加载测试数据
    - `model`：要测试的模型
    - `dataset`：数据集的名称
    - `log_every_n_batch`：每多少个批次记录一次信息，默认为 -1
    - `name`：测试的名称，默认为空字符串
    - `compile_detections`：是否编译检测结果，默认为 `False`
    - `no_eval`：是否不进行评估，默认为 `False`
    """
    model.eval()
    # 将模型设置为评估模式

    print('run_test_with_visualization', compile_detections, no_eval)
    # 打印函数名称和一些相关的参数值

    if not no_eval:
        """
        如果进行评估，则创建一个 `DetectionBuffer` 对象用于计算平均精度等指标
        """
        mapcalc = DetectionBuffer(height=loader.dataset.height, width=loader.dataset.width, classes=loader.dataset.classes)
    counter = 0
    # 初始化计数器

    if compile_detections:
        """
        如果要编译检测结果，则初始化一个列表来存储编译后的检测结果
        """
        compiled_detections = []

    for i, data in enumerate(tqdm.tqdm(loader, desc=f"Testing {name}")):
        """
        使用 `tqdm` 库来显示加载数据的进度，并遍历数据

        例如，如果 `loader` 是一个包含 100 个数据样本的迭代器，这里会依次取出每个样本，并显示进度信息 "Testing <name>"
        """
        data = data.cuda(non_blocking=True)
        # 将数据移动到 GPU 上

        data_for_visualization = data.clone()
        # 复制数据用于可视化

        data = format_data(data)
        # 对数据进行某种格式化操作

        detections, targets = model(data.clone())
        # 使用模型对格式化后的数据进行预测，得到检测结果和目标
        print("targets", targets[0])
        print("targets.length ", len(targets))
        print("detections", detections[0])
        print("detections.length ", len(detections))
        print('class_names', loader.dataset.classes)


        if compile_detections:
            """
            如果要编译检测结果，将格式化后的检测结果添加到 `compiled_detections` 列表中
            """
            compiled_detections.extend(format_detections(data.sequence, data.t1, detections))

        print("log_every_n_batch",log_every_n_batch)
        if log_every_n_batch > 0 and counter % log_every_n_batch == 0:
            """
            如果 `log_every_n_batch` 大于 0 且当前计数器满足记录条件，则记录检测框的信息
            """
            log_bboxes(data_for_visualization, targets=targets, detections=detections, bidx=4, class_names=loader.dataset.classes, key="testing/evaluated_bboxes")

        if not no_eval:
            """
            如果进行评估，使用 `mapcalc` 对象来更新检测结果和目标
            """
            mapcalc.update(detections, targets, dataset, data.height[0], data.width[0])

        if i % 5 == 0:
            """
            每 5 个批次清理一次 GPU 缓存
            """
            torch.cuda.empty_cache()

        counter += 1
        # 计数器递增

    torch.cuda.empty_cache()
    # 再次清理 GPU 缓存

    data = None
    if not no_eval:
        """
        如果不是不进行评估，计算并获取评估指标数据
        """
        data = mapcalc.compute()

    return (data, compiled_detections) if compile_detections else data
    # 根据 `compile_detections` 的值返回相应的结果