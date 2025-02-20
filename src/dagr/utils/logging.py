import torch
import wandb
import os

from typing import List, Dict, Optional
from torch_geometric.data import Batch
from pathlib import PosixPath
from pprint import pprint
from pathlib import Path

from torch_geometric.data import Data


def set_up_logging_directory(dataset, task, output_directory):
    project = f"low_latency-{dataset}-{task}"
    print('project',project)

    output_directory = output_directory / dataset / task
    output_directory.mkdir(parents=True, exist_ok=True)
   # wandb.init(project=project, entity="rpg", save_code=True, dir=str(output_directory))

   # name = wandb.run.name
    name = 'test003'
    output_directory = output_directory / name
    output_directory.mkdir(parents=True, exist_ok=True)
    os.system(f"cp -r {os.path.join(os.path.dirname(__file__), '../../low_latency_object_detection')} {str(output_directory)}")

    return output_directory

def log_hparams(args):
    hparams = {k: str(v) if type(v) is PosixPath else v for k, v in vars(args).items()}
    pprint(hparams)
    #wandb.log(hparams)

def log_bboxes(data: Batch,
               targets: List[Dict[str, torch.Tensor]],
               detections: List[Dict[str, torch.Tensor]],
               class_names: List[str],
               bidx: int,
               key: str):

    print("targets" ,targets[0] )
    print("detections" ,detections[0])
    print('class_names',class_names)


    """
    用途：处理和记录边界框（Bounding Boxes）相关的数据

    输入含义：
    - `data`：一批数据
    - `targets`：目标边界框的列表，每个元素是一个包含边界框信息的字典
    - `detections`：检测到的边界框的列表，每个元素是一个包含边界框信息的字典
    - `class_names`：类别名称的列表
    - `bidx`：用于控制循环结束的索引
    - `key`：用于在记录时标识数据的键

    输出含义：
    - 无直接返回值，主要是将处理后的边界框数据通过 `wandb.log` 进行记录

    例子：
    假设 `data` 包含一些图像数据，`targets` 和 `Detections` 分别包含目标和检测到的边界框信息，`class_names` 是 ["cat", "dog"]，`bidx` 是 5，`key` 是 "bbox_data"。
    函数会处理这些边界框数据，并将其以指定的 `key` 记录到 `wandb` 中。
    """
    gt_bbox = []
    # 初始化用于存储真实边界框的列表
    det_bbox = []
    # 初始化用于存储检测到的边界框的列表
    images = []
    # 初始化用于存储图像的列表

    for b, datum in enumerate(data.to_data_list()):
        # 遍历数据列表中的每个元素
        image = visualize_events(datum)
        # 对每个数据元素进行某种可视化操作得到图像
        image = torch.cat([image, image], dim=1)
        # 对图像进行拼接操作
        images.append(image)

        if len(detections) > 0:
            det = detections[b]
            det = torch.cat([det['boxes'], det['labels'].view(-1, 1), det['scores'].view(-1, 1)], dim=-1)
            """
            对这个检测结果中的 'boxes' 、 'labels' （转换为列向量）和 'scores' （转换为列向量）沿着最后一个维度（-1）进行拼接
            例如，如果 det['boxes'] 是形状为 (m, 4) 的张量，det['labels'].view(-1,1) 是形状为 (m, 1) 的张量，det['scores'].view(-1,1) 是形状为 (m, 1) 的张量，那么拼接后的 det 形状将是 (m, 6)
            """
            det[:, [0, 2]] += b * datum.width
            """
            对拼接后的结果 det 的第 0 列和第 2 列的元素分别加上 b 乘以 datum.width 的值，以调整边界框的坐标。
            假设 b = 2，datum.width = 10，det[:, 0] 的原始值为 [5, 7, 9]，那么调整后的值将为 [5 + 2 * 10, 7 + 2 * 10, 9 + 2 * 10] = [25, 27, 29]
            """
            det_bbox.append(det)
            """
            将调整后的检测结果添加到 'det_bbox' 列表中
            """

        if len(targets) > 0:
            # 如果有目标边界框
            tar = targets[b]
            tar = torch.cat([tar['boxes'], tar['labels'].view(-1, 1), torch.ones_like(tar['labels'].view(-1, 1))], dim=-1)
            # 对目标边界框的相关张量进行拼接
            tar[:, [0, 2]] += b * datum.width
            tar[:, [1, 3]] += datum.height
            # 对边界框的坐标进行调整
            gt_bbox.append(tar)

        if b == bidx-1:
            # 如果达到指定的索引则结束循环
            break

    pred_bbox = torch.cat(det_bbox)
    # 拼接检测到的边界框
    gt_bbox = torch.cat(gt_bbox)
    # 拼接真实边界框
    images = torch.cat(images, dim=-1)
    # 拼接图像

    gt_bbox[:,[0,2]] /= (bidx * datum.width)
    gt_bbox[:,[1,3]] /= (2 * datum.height)

    pred_bbox[:,[0,2]] /= (bidx * datum.width)
    pred_bbox[:,[1,3]] /= (2 * datum.height)

    image = __convert_to_wandb_data(images.detach().float().cpu(),
                                    gt_bbox.detach().cpu(),
                                    pred_bbox.detach().cpu(),
                                    class_names)
    # 将处理后的图像、真实边界框和预测边界框转换为适合 `wandb` 的数据格式

    wandb.log({key: image})
    # 使用 `wandb.log` 记录数据

def visualize_events(data: Data)->torch.Tensor:
    x, y = data.pos[:,:2].long().t()
    p = data.x[:,0].long()

    if hasattr(data, "image"):
        image = data.image[0].clone()
    else:
        image = torch.full(size=(3, data.height, data.width), fill_value=255, device=p.device, dtype=torch.uint8)

    is_pos = p == 1
    image[:, y[is_pos], x[is_pos]] = torch.tensor([[0],[0],[255]], dtype=torch.uint8, device=p.device)
    image[:, y[~is_pos], x[~is_pos]] = torch.tensor([[255],[0],[0]], dtype=torch.uint8, device=p.device)

    return image

def __convert_to_wandb_data(image: torch.Tensor, gt: torch.Tensor, p: torch.Tensor, class_names: List[str])->wandb.Image:
    return wandb.Image(image, boxes={
        "predictions": __parse_bboxes(p, class_names, suffix="P"),
        "ground_truth": __parse_bboxes(gt, class_names)
    })

def __parse_bboxes(bboxes: torch.Tensor, class_names: List[str], suffix: str="GT"):
    # bbox N x 6 -> xyxycs
    return {
        "box_data": [__parse_bbox(bbox, class_names, suffix) for bbox in bboxes],
        "class_labels": dict(enumerate(class_names))
    }

def __parse_bbox(bbox: torch.Tensor, class_names: List[str], suffix: str="GT"):
    # bbox xyxycs
    return {
        "position": {
            "minX": float(bbox[0]),
            "minY": float(bbox[1]),
            "maxX": float(bbox[2]),
            "maxY": float(bbox[3])
        },
        "class_id": int(bbox[-2]),
        "scores": {
            "object score": float(bbox[-1])
        },
        "bbox_caption": f"{suffix} - {class_names[int(bbox[-2])]}"
    }


