import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from dagr.visualization.bbox_viz import draw_bbox_on_img
from dagr.visualization.event_viz import draw_events_on_image
from dsec_det.directory import DSECDirectory
from dsec_det.io import extract_from_h5_by_timewindow, extract_image_by_index, load_start_and_end_time
from dsec_det.preprocessing import compute_index
from sort.sort import Sort

if __name__ == '__main__':
    parser = argparse.ArgumentParser("""Visualization script to show bounding boxes""")
    parser.add_argument("--detections_folder", help="Path to folder with detections.", type=Path,
                        default="/data/scratch1/daniel/logs/dsec/detection/graceful-snowball-1298")
    parser.add_argument("--dataset_directory", help="Path to DSEC folder including which split.", type=Path,
                        default="/data/scratch1/daniel/datasets/DSEC_fragment/test")
    parser.add_argument("--vis_time_step_us", help="Number of microseconds to step each iteration.", type=int,
                        default=1000)
    parser.add_argument("--event_time_window_us", help="Length of sliding event time window for visualization.",
                        type=int, default=5000)
    parser.add_argument("--sequence",
                        help="Sequence to visualize. Must be an official DSEC sequence e.g. zurich_city_13_b",
                        default="zurich_city_13_b", type=str)
    parser.add_argument("--write_to_output",
                        help="Whether to save images in folder ${detections_folder}/visualization. Otherwise, just cv2.imshow is used.",
                        action="store_true")
    args = parser.parse_args()

    detections_file = 'log/dsec/detection/test111/detections_zurich_city_13_b.npy'
    detections = np.load(detections_file)

    print("detections", detections)
    print("detections type ", type(detections))
    print("detections.dtype.names", detections.dtype.names)

    # detections.dtype.names ('t', 'x', 'y', 'w', 'h', 'class_id', 'class_confidence')

    # tracking_results = sort_tracker(detections)
    # print('tracking_results', tracking_results)
    detection_timestamps = np.unique(detections['t'])

    dsec_directory = DSECDirectory(args.dataset_directory / args.sequence)


    t0, t1 = load_start_and_end_time(dsec_directory)

    vis_timestamps = np.arange(t0, t1, step=args.vis_time_step_us)
    # 找到对应的图片
    step_index_to_image_index = compute_index(dsec_directory.images.timestamps, vis_timestamps)
    step_index_to_boxes_index = compute_index(detection_timestamps, vis_timestamps)

    # Sort算法初始化
    mot_tracker = Sort()

    image_list = []  # 用于存储处理后的每一帧图像

    if args.write_to_output:
        output_path = args.detections_folder / "visualization_sort"
        output_path.mkdir(parents=True, exist_ok=True)
        # 定义视频相关参数
        fps = 60  # 帧率，可根据实际需求调整
        video_name = os.path.join(output_path, 'output_video.mp4')  # 输出视频文件名
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 视频编码格式，可按需更换


    for step, t in enumerate(vis_timestamps):

        # find most recent image
        image_index = step_index_to_image_index[step]
        image = extract_image_by_index(dsec_directory.images.image_files_distorted, image_index)

        # find events within time window [image_timestamps, t]
        events = extract_from_h5_by_timewindow(dsec_directory.events.event_file, t-args.event_time_window_us, t)

        # find most recent bounding boxes
        boxes_index = step_index_to_boxes_index[step]
        boxes_timestamp = detection_timestamps[boxes_index]
        # 从 Detections 中筛选出 't' 字段的值等于 boxes_timestamp 的那些检测结果，并将它们存储在 boxes 变量中
        boxes = detections[detections['t'] == boxes_timestamp]
        print('boxes', boxes)

        # draw them on one image
        scale = 2
        image = draw_events_on_image(image, events['x'], events['y'], events['p'])
        # # 增加代码 假设 img 是 draw_events_on_image 的输出
        # if not isinstance(image, np.ndarray):
        #     image = np.array(image)  # 如果 img 不是 numpy 数组，进行转换

        print("boxes['class_id']",boxes["class_id"])

        # image = draw_bbox_on_img(image, scale*boxes['x'], scale*boxes['y'], scale*boxes['w'], scale*boxes["h"],
        #                          boxes["class_id"], boxes['class_confidence'], conf=0.3, nms=0.65)
        #
        # image_list.append(image)





        boxes_sorted = []
        # 转换后的结果列表
        transformed_detections = []

        for det in boxes:

            x1 = det['x'] - det['w'] / 2
            y1 = det['y'] - det['h'] / 2
            x2 = det['x'] + det['w'] / 2
            y2 = det['y'] + det['h'] / 2
            confidence = det['class_confidence']
            class_id = det['class_id']
            if confidence < 0.5:
                continue
            transformed_detections.append([x1, y1, x2, y2, confidence])

        transformed_detections_numpy = np.array(transformed_detections)
        track_bbs_ids = mot_tracker.update(transformed_detections_numpy)
        # print('track_bbs_ids', track_bbs_ids)
        # print('track_bbs_ids type', type(track_bbs_ids))
        # print('track_bbs_ids shape', track_bbs_ids.shape)


        for track in track_bbs_ids:
            x1, y1, x2, y2, track_id = track

            # 计算目标的中心坐标 (x, y) 和宽度 (w) 和高度 (h)
            x = (x1 + x2) / 2
            y = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1

            # 假设置信度可以是一个固定值，或者来自其他信息
            confidence = 1.0  # 或者从其他来源获得，如 det['class_confidence']

            # 创建新的检测框字典，添加 track_id 到 class_id
            box_temp = {
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'class_confidence': confidence,
                'class_id': 1.0
            }

            print('box_temp', box_temp)
            boxes_sorted.append(box_temp)

            boxes_sort = pd.DataFrame(boxes_sorted)
            print('boxes_sort', boxes_sort)
            print("boxes_sort['class_id']",boxes_sort['class_id'])

            image = draw_bbox_on_img(image, scale * boxes_sort['x'], scale * boxes_sort['y'], scale * boxes_sort['w'],
                                     scale * boxes_sort["h"],
                                     boxes_sort['class_id'].to_numpy(), boxes_sort['class_confidence'], conf=0.3, nms=0.65)

            image_list.append(image)

            # if args.write_to_output:
            #     print('output_path', output_path)
            #     cv2.imwrite(str(output_path / ("%06d.png" % step)), image)
            #     pass

    # 打印转换后的格式
    # print(transformed_detections)

    # 获取图像尺寸（假设所有图像尺寸一致，取第一张图像的尺寸为例）
    height, width, _ = image_list[0].shape
    video_writer = cv2.VideoWriter(video_name, fourcc, fps, (width, height))
    # 将每一帧图像写入视频
    for img in image_list:
        video_writer.write(img)

    video_writer.release()
    print(f"视频已成功保存至 {video_name}")


