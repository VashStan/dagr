# avoid matlab error on server
import os
import torch
import wandb
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

from torch_geometric.data import DataLoader
from dagr.utils.args import FLAGS

from dagr.data.dsec_data import DSEC
from dagr.data.augment import Augmentations

from dagr.model.networks.dagr import DAGR
from dagr.model.networks.ema import ModelEMA

from dagr.utils.logging import set_up_logging_directory, log_hparams
from dagr.utils.testing import run_test_with_visualization


if __name__ == '__main__':
    """
    当脚本作为主程序运行时执行以下代码块
    """
    import torch_geometric
    import random
    import numpy as np
    """
    导入所需的模块
    """

    seed = 42
    """
    设置随机数种子为 42
    """
    torch_geometric.seed.seed_everything(seed)
    torch.random.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    """
    为各种随机操作设置相同的种子，以确保可重复性
    """

    args = FLAGS()
    """
    获取命令行参数或其他方式定义的参数
    """

    output_directory = set_up_logging_directory(args.dataset, args.task, args.output_directory)
    """
    设置日志输出目录
    """

    project = f"low_latency-{args.dataset}-{args.task}"
    """
    定义项目名称
    """
    print(f"PROJECT: {project}")
    log_hparams(args)
    """
    打印项目名称并记录超参数
    """

    print("init datasets")
    dataset_path = args.dataset_directory.parent / args.dataset
    """
    获取数据集路径
    """

    test_dataset = DSEC(args.dataset_directory, "test", Augmentations.transform_testing, debug=False, min_bbox_diag=15, min_bbox_height=10)
    """
    初始化测试数据集
    """

    num_iters_per_epoch = 1
    """
    设置每个 epoch 的迭代次数
    """

    sampler = np.random.permutation(np.arange(len(test_dataset)))
    """
    对数据集的索引进行随机排列
    """
    test_loader = DataLoader(test_dataset, sampler=sampler, follow_batch=['bbox', 'bbox0'], batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)
    """
    创建数据加载器
    """

    print("init net")
    # load a dummy sample to get height, width
    model = DAGR(args, height=test_dataset.height, width=test_dataset.width)
    model = model.cuda()
    ema = ModelEMA(model)
    """
    初始化网络模型和相关对象，并将模型移动到 GPU
    """

    assert "checkpoint" in args
    """
    断言命令行参数中包含 'checkpoint'
    """
    checkpoint = torch.load(args.checkpoint)
    ema.ema.load_state_dict(checkpoint['ema'])
    ema.ema.cache_luts(radius=args.radius, height=test_dataset.height, width=test_dataset.width)
    """
    加载检查点并更新模型状态
    """

    with torch.no_grad():
        """
        在不计算梯度的上下文环境中
        """
        metrics = run_test_with_visualization(test_loader, ema.ema, dataset=args.dataset)
        """
        运行带有可视化的测试并获取评估指标
        """
        log_data = {f"testing/metric/{k}": v for k, v in metrics.items()}
        """
        构建要记录的指标数据
        """
        wandb.log(log_data)
        """
        使用 wandb 记录指标数据
        """
        print(metrics['mAP'])
        """
        打印平均精度（mAP）的值
        """
