#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
# 文件: test_pedestrian_detection.py
# 描述: 行人检测模块测试脚本（纯Python版本）
# ==============================================================================

"""
行人检测模块测试脚本
====================

直接运行此脚本测试PedestrianDetector模块。

使用示例：
```bash
# 使用默认配置
python test_pedestrian_detection.py --image /path/to/test.jpg

# 指定检测器和输出目录
python test_pedestrian_detection.py \
    --image /path/to/test.jpg \
    --output ./results \
    --detector yolov8n \
    --confidence 0.25
```
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import cv2
import numpy as np

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# 将 habitat-baselines/ 目录加入 sys.path，使 Python 可按 habitat_baselines 导入
# （目录实际名为 habitat-baselines，代码中用 habitat_baselines 导入）
sys.path.insert(0, os.path.join(PROJECT_ROOT, "habitat-baselines"))

# 导入检测模块
try:
    from habitat_baselines.rl.ppo.brain.pedestrian_detection import (
        PedestrianDetector,
        DetectorType,
        DetectionResult,
    )
    MODULE_AVAILABLE = True
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保在项目根目录运行此脚本")
    MODULE_AVAILABLE = False


def draw_detections(image, detections, output_path, detector_name):
    """
    在图像上绘制检测结果

    Args:
        image: 输入图像
        detections: 检测结果列表
        output_path: 输出图像路径
        detector_name: 检测器名称
    """
    img = image.copy()

    for i, det in enumerate(detections):
        bbox = det.bbox
        score = det.score

        x1, y1, x2, y2 = [int(v) for v in bbox]

        # 根据警告等级选择颜色
        if det.relative_area > 0.15:
            color = (0, 0, 255)
        elif det.relative_area > 0.05:
            color = (0, 255, 255)
        else:
            color = (0, 255, 0)  # 绿色 - 远距离

        # 绘制边界框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # 绘制标签
        label = f"Ped-{i+1}: {score:.2f}"
        if det.relative_area > 0.15:
            label += " [DANGER]"
        elif det.relative_area > 0.05:
            label += " [CAUTION]"

        # 计算文本位置
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 绘制中心点
        cx, cy = int(det.center[0]), int(det.center[1])
        cv2.circle(img, (cx, cy), 5, color, -1)

    # 添加检测器信息
    info_text = f"Detector: {detector_name} | Detections: {len(detections)}"
    cv2.putText(img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # 保存图像
    cv2.imwrite(output_path, img)
    print(f"  [保存] 打框图像: {output_path}")


def save_results_json(results, output_path):
    """保存检测结果到JSON文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  [保存] JSON结果: {output_path}")


def save_results_csv(results, output_path):
    """保存检测结果到CSV文件"""
    import csv

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'x1', 'y1', 'x2', 'y2', 'score', 'center_x', 'center_y', 'width', 'height', 'relative_area'])

        for det in results['detections']:
            writer.writerow([
                det['id'],
                det['bbox'][0],
                det['bbox'][1],
                det['bbox'][2],
                det['bbox'][3],
                det['score'],
                det['center'][0],
                det['center'][1],
                det['width'],
                det['height'],
                det['relative_area'],
            ])
    print(f"  [保存] CSV结果: {output_path}")


def save_summary_txt(results, output_path):
    """保存简洁的文本摘要"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("         行人检测结果摘要\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"检测器: {results['detector_type']}\n")
        f.write(f"模型路径: {results['model_path']}\n")
        f.write(f"输入图像: {results['image_path']}\n\n")

        f.write("-" * 60 + "\n")
        f.write("检测性能:\n")
        f.write("-" * 60 + "\n")
        f.write(f"  检测耗时: {results['elapsed_ms']:.2f} ms\n")
        f.write(f"  检测数量: {results['num_detections']} 个行人\n")
        f.write(f"  是否检测到行人: {'是' if results['success'] else '否'}\n\n")

        f.write("-" * 60 + "\n")
        f.write("检测详情:\n")
        f.write("-" * 60 + "\n")

        if results['num_detections'] > 0:
            for det in results['detections']:
                f.write(f"\n  行人 #{det['id']}:\n")
                f.write(f"    边界框: [{det['bbox'][0]:.1f}, {det['bbox'][1]:.1f}, {det['bbox'][2]:.1f}, {det['bbox'][3]:.1f}]\n")
                f.write(f"    中心点: ({det['center'][0]:.1f}, {det['center'][1]:.1f})\n")
                f.write(f"    宽高: {det['width']:.1f} x {det['height']:.1f}\n")
                f.write(f"    置信度: {det['score']:.4f}\n")
                f.write(f"    相对面积: {det['relative_area']:.4f}\n")

                # 根据面积给出距离估算
                if det['relative_area'] > 0.15:
                    f.write(f"    距离估算: 很近 (< 3m)\n")
                elif det['relative_area'] > 0.05:
                    f.write(f"    距离估算: 中等 (3-5m)\n")
                else:
                    f.write(f"    距离估算: 较远 (> 5m)\n")
        else:
            f.write("  未检测到行人\n")

        f.write("\n" + "=" * 60 + "\n")

    print(f"  [保存] TXT摘要: {output_path}")


def format_results(result, detector_type, model_path, image_path):
    """格式化检测结果为字典"""
    detections = []
    for det in result.detection_list:
        detections.append({
            "id": len(detections),
            "bbox": det.bbox,
            "center": det.center,
            "width": det.width,
            "height": det.height,
            "score": det.score,
            "relative_area": det.relative_area,
        })

    return {
        "detector_type": detector_type,
        "model_path": model_path,
        "image_path": image_path,
        "elapsed_ms": result.elapsed_ms,
        "num_detections": result.num_detections,
        "success": result.has_pedestrian,
        "image_shape": list(result.image_shape),
        "detections": detections,
    }


def test_detector(image_path, detector_type, confidence, device, output_dir, image_name):
    """测试单个检测器"""
    print(f"\n{'='*60}")
    print(f"测试检测器: {detector_type}")
    print(f"{'='*60}")

    try:
        # 初始化检测器
        print("  初始化检测器...")
        start_init = time.perf_counter()
        detector = PedestrianDetector(
            detector_type=detector_type,
            device=device,
            confidence=confidence,
        )
        init_time = time.perf_counter() - start_init
        print(f"  初始化耗时: {init_time*1000:.2f}ms")

        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        print(f"  图像尺寸: {image_rgb.shape[1]}x{image_rgb.shape[0]}")

        # 执行检测
        print("  执行检测...")
        start_time = time.perf_counter()
        result = detector.detect(image_rgb, frame_id=0)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        print(f"\n  [结果] 检测到 {result.num_detections} 个行人")
        print(f"  [结果] 检测耗时: {elapsed_ms:.2f}ms")

        # 打印检测详情
        if result.num_detections > 0:
            print("\n  检测详情:")
            for i, det in enumerate(result.detection_list):
                distance_est = "很近" if det.relative_area > 0.15 else ("中等" if det.relative_area > 0.05 else "较远")
                print(f"    行人{i+1}: bbox=[{det.bbox[0]:.0f},{det.bbox[1]:.0f},{det.bbox[2]:.0f},{det.bbox[3]:.0f}] "
                      f"置信度={det.score:.3f} 面积={det.relative_area:.4f} → {distance_est}")

        # 格式化结果
        results = format_results(
            result,
            detector_type,
            str(detector.checkpoint_path or "auto"),
            image_path,
        )

        # 保存打框图像
        vis_image_path = os.path.join(output_dir, "images", f"{image_name}_{detector_type}_vis.jpg")
        draw_detections(image, result.detection_list, vis_image_path, detector_type)

        # 保存结果
        base_name = f"{image_name}_{detector_type}"
        save_results_json(results, os.path.join(output_dir, f"{base_name}_results.json"))
        save_results_csv(results, os.path.join(output_dir, f"{base_name}_results.csv"))
        save_summary_txt(results, os.path.join(output_dir, f"{base_name}_summary.txt"))

        # 清理
        detector.cleanup()

        return results

    except Exception as e:
        print(f"  [错误] {e}")
        import traceback
        traceback.print_exc()
        return None


def process_folder(folder_path, detector_type, confidence, device, output_dir):
    """
    批量处理文件夹中的所有图像。

    输出结构：
      output_dir/
      ├── <detector_type>/
      │   ├── images/          # 可视化打框图像
      │   │   └── <原图名>_vis.jpg
      │   └── results/
      │       ├── <原图名>_results.json
      │       ├── <原图名>_results.csv
      │       └── <原图名>_summary.txt
      └── <detector_type>_batch_summary.json
    """
    print(f"\n{'='*60}")
    print(f"批量检测: {folder_path}")
    print(f"检测器: {detector_type}")
    print(f"{'='*60}")

    # 收集图像
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_paths = []
    for root, _, files in os.walk(folder_path):
        for f in sorted(files):
            if Path(f).suffix.lower() in extensions:
                image_paths.append(os.path.join(root, f))

    if not image_paths:
        print(f"[错误] 文件夹中未找到图像: {folder_path}")
        return {}

    print(f"找到 {len(image_paths)} 张图像")

    det_dir = os.path.join(output_dir, detector_type)
    img_out = os.path.join(det_dir, "images")
    res_out = os.path.join(det_dir, "results")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(res_out, exist_ok=True)

    # 初始化检测器（只初始化一次）
    print("初始化检测器...")
    try:
        detector = PedestrianDetector(
            detector_type=detector_type,
            device=device,
            confidence=confidence,
        )
    except Exception as e:
        print(f"[错误] 初始化检测器失败: {e}")
        return {}

    batch_results = {}
    for i, img_path in enumerate(image_paths):
        rel = os.path.relpath(img_path, folder_path).replace(os.sep, '_')
        img_name = Path(rel).stem
        print(f"\n  [{i+1}/{len(image_paths)}] {rel}")

        try:
            image = cv2.imread(img_path)
            if image is None:
                print(f"    [跳过] 无法读取图像")
                continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            result = detector.detect(image_rgb, frame_id=i)

            if result.num_detections > 0:
                for j, det in enumerate(result.detection_list):
                    dist = "很近" if det.relative_area > 0.15 else ("中等" if det.relative_area > 0.05 else "较远")
                    print(f"    行人{j+1}: [{det.bbox[0]:.0f},{det.bbox[1]:.0f},{det.bbox[2]:.0f},{det.bbox[3]:.0f}] "
                          f"置信={det.score:.3f} {dist}")
            else:
                print(f"    无检测")

            formatted = format_results(result, detector_type,
                                      str(detector.checkpoint_path or "auto"), img_path)

            # 保存可视化图像
            vis_path = os.path.join(img_out, f"{img_name}_vis.jpg")
            draw_detections(image, result.detection_list, vis_path, detector_type)

            # 保存结果文件
            save_results_json(formatted, os.path.join(res_out, f"{img_name}_results.json"))
            save_results_csv(formatted, os.path.join(res_out, f"{img_name}_results.csv"))
            save_summary_txt(formatted, os.path.join(res_out, f"{img_name}_summary.txt"))

            batch_results[rel] = formatted

        except Exception as e:
            print(f"    [错误] {e}")

    detector.cleanup()

    # 保存汇总
    summary_path = os.path.join(output_dir, f"{detector_type}_batch_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            "folder": folder_path,
            "detector": detector_type,
            "total_images": len(image_paths),
            "processed": len(batch_results),
            "results": batch_results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n[汇总] 批量结果已保存: {summary_path}")

    return batch_results


def main():
    parser = argparse.ArgumentParser(
        description='行人检测模块测试 / 批量检测',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单张图像测试
  python test_pedestrian_detection.py --image test.jpg
  python test_pedestrian_detection.py --image test.jpg --detector yolov8s
  python test_pedestrian_detection.py --image test.jpg --detector all --output ./results

  # 批量检测（文件夹中所有图像，按检测器分目录）
  python test_pedestrian_detection.py --folder data/collect_data/val/scene_001/rgb/
  python test_pedestrian_detection.py --folder data/collect_data/val/7Ukhou1GxYi.basis/18/rgb/ --detector yolov8n --output ./results
  python test_pedestrian_detection.py --folder ./images --detector all --confidence 0.3
        """
    )
    parser.add_argument('--image', help='单张图像路径')
    parser.add_argument('--folder', help='批量检测：图像文件夹路径')
    parser.add_argument('--output', default='./ped_detection_results', help='输出根目录')
    parser.add_argument('--detector', default='yolov8n',
                        help='检测器: yolov8n, yolov8s, yolov8m, all (默认 yolov8n)')
    parser.add_argument('--confidence', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--device', default='cuda', help='设备: cuda, cpu')
    args = parser.parse_args()

    if not MODULE_AVAILABLE:
        print("错误: 无法导入PedestrianDetector模块")
        print("请确保在项目根目录运行此脚本")
        sys.exit(1)

    # 参数校验：--image 和 --folder 必须至少提供一个
    if not args.image and not args.folder:
        parser.error("必须提供 --image 或 --folder 参数")
    if args.image and args.folder:
        parser.error("--image 和 --folder 不能同时使用")

    # 确定检测器列表
    if args.detector == 'all':
        detectors = ['yolov8n', 'yolov8s']
    else:
        detectors = [args.detector]

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, "images"), exist_ok=True)

    # 批量检测模式
    if args.folder:
        if not os.path.isdir(args.folder):
            print(f"错误: 文件夹不存在: {args.folder}")
            sys.exit(1)
        for det in detectors:
            process_folder(args.folder, det, args.confidence, args.device, args.output)
        print("\n" + "="*60)
        print("批量检测完成！")
        print("="*60)
        print(f"\n输出目录: {args.output}")
        print("目录结构:")
        print(f"  {args.output}/")
        for det in detectors:
            print(f"  ├── {det}/")
            print(f"  │   ├── images/        # 打框图像")
            print(f"  │   ├── results/       # JSON/CSV/TXT 结果")
            print(f"  │   └── {det}_batch_summary.json")
        return

    # 单图检测模式
    if not os.path.exists(args.image):
        print(f"错误: 图像文件不存在: {args.image}")
        sys.exit(1)

    image_name = os.path.basename(args.image).rsplit('.', 1)[0]
    results_all = {}

    for det in detectors:
        result = test_detector(
            args.image,
            det,
            args.confidence,
            args.device,
            args.output,
            image_name,
        )
        if result:
            results_all[det] = result

    if results_all:
        summary = {
            "test_image": args.image,
            "detectors_tested": list(results_all.keys()),
            "results": results_all,
        }
        summary_path = os.path.join(args.output, f"{image_name}_all_results.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\n[汇总] 多检测器结果已保存: {summary_path}")

    print("\n" + "="*60)
    print("测试完成!")
    print("="*60)
    print(f"\n输出目录: {args.output}")
    print("生成的文件:")
    print("  - images/               : 打框图像")
    print("  - *_results.json       : 完整JSON结果")
    print("  - *_results.csv        : CSV格式结果")
    print("  - *_summary.txt        : 文本摘要")
    print("  - *_all_results.json   : 多检测器汇总")


if __name__ == "__main__":
    main()
