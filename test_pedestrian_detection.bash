#!/bin/bash
# ==============================================================================
# 文件: test_pedestrian_detection.bash
# 描述: 行人检测模块测试脚本
# ==============================================================================

"""
行人检测模块测试脚本
=====================

测试PedestrianDetector模块的效果：
1. 读取测试图像
2. 调用检测器进行行人检测
3. 保存打框图像
4. 保存检测结果（位置、置信度、耗时）

使用示例：
```bash
# 使用默认配置测试
bash test_pedestrian_detection.bash --image /path/to/test.jpg --output ./results

# 指定检测器类型
bash test_pedestrian_detection.bash \
    --image /path/to/test.jpg \
    --detector yolov8n \
    --output ./results

# 测试所有检测器
bash test_pedestrian_detection.bash \
    --image /path/to/test.jpg \
    --detector all \
    --output ./results
```
"""

# =============================================================================
# 参数解析
# =============================================================================

IMAGE_PATH=""
OUTPUT_DIR="./ped_detection_results"
DETECTOR="yolov8n"
CONFIDENCE=0.25
DEVICE="cuda"

while [[ $# -gt 0 ]]; do
    case $1 in
        --image)
            IMAGE_PATH="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --detector)
            DETECTOR="$2"
            shift 2
            ;;
        --confidence)
            CONFIDENCE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --image PATH      测试图像路径 (必需)"
            echo "  --output DIR      输出目录 (默认: ./ped_detection_results)"
            echo "  --detector TYPE   检测器类型 (默认: yolov8n)"
            echo "                    可选: yolov8n, yolov8s, yolov8m, rtdetr_r18, rtdetr_r50, all"
            echo "  --confidence VAL  置信度阈值 (默认: 0.25)"
            echo "  --device DEV      设备 (默认: cuda)"
            echo "  --help, -h        显示帮助"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 检查必需参数
if [ -z "${IMAGE_PATH}" ]; then
    echo "错误: 必须指定 --image 参数"
    echo "使用 --help 查看帮助"
    exit 1
fi

if [ ! -f "${IMAGE_PATH}" ]; then
    echo "错误: 图像文件不存在: ${IMAGE_PATH}"
    exit 1
fi

# =============================================================================
# 创建输出目录
# =============================================================================

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}/images"

# 获取图像名称（不含扩展名）
IMAGE_NAME=$(basename "${IMAGE_PATH}" | sed 's/\.[^.]*$//')

# =============================================================================
# 创建Python测试脚本
# =============================================================================

cat > "${OUTPUT_DIR}/run_detection.py" << 'PYTHON_SCRIPT'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
行人检测测试脚本
================

自动生成，用于调用PedestrianDetector并保存结果
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
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

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
            color = (0, 0, 255)  # 红色 - 近距离
        elif det.relative_area > 0.05:
            color = (0, 255, 255)  # 黄色 - 中距离
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
    print(f"  保存打框图像: {output_path}")


def save_results_json(results, output_path):
    """
    保存检测结果到JSON文件

    Args:
        results: 检测结果字典
        output_path: 输出文件路径
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  保存JSON结果: {output_path}")


def save_results_csv(results, output_path):
    """
    保存检测结果到CSV文件

    Args:
        results: 检测结果字典
        output_path: 输出文件路径
    """
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
    print(f"  保存CSV结果: {output_path}")


def save_summary_txt(results, output_path):
    """
    保存简洁的文本摘要

    Args:
        results: 检测结果字典
        output_path: 输出文件路径
    """
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

    print(f"  保存TXT摘要: {output_path}")


def format_results(result: DetectionResult, detector_type: str, model_path: str, image_path: str) -> dict:
    """
    格式化检测结果为字典

    Args:
        result: DetectionResult对象
        detector_type: 检测器类型
        model_path: 模型路径
        image_path: 图像路径

    Returns:
        结果字典
    """
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
    """
    测试单个检测器

    Args:
        image_path: 测试图像路径
        detector_type: 检测器类型
        confidence: 置信度阈值
        device: 计算设备
        output_dir: 输出目录
        image_name: 图像名称
    """
    print(f"\n{'='*60}")
    print(f"测试检测器: {detector_type}")
    print(f"{'='*60}")

    try:
        # 初始化检测器
        print("  初始化检测器...")
        detector = PedestrianDetector(
            detector_type=detector_type,
            device=device,
            confidence=confidence,
        )

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

        print(f"  检测结果: {result.num_detections} 个行人")
        print(f"  检测耗时: {elapsed_ms:.2f}ms")

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
        print(f"  错误: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='行人检测测试')
    parser.add_argument('--image', required=True, help='测试图像路径')
    parser.add_argument('--output', default='./ped_detection_results', help='输出目录')
    parser.add_argument('--detector', default='yolov8n', help='检测器类型')
    parser.add_argument('--confidence', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--device', default='cuda', help='设备')
    args = parser.parse_args()

    if not MODULE_AVAILABLE:
        print("错误: 无法导入PedestrianDetector模块")
        sys.exit(1)

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, "images"), exist_ok=True)

    image_name = os.path.basename(args.image).rsplit('.', 1)[0]

    # 确定要测试的检测器
    if args.detector == 'all':
        detectors = ['yolov8n', 'yolov8s', 'rtdetr_r18']
    else:
        detectors = [args.detector]

    results_all = {}

    # 测试每个检测器
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

    # 保存汇总报告
    if results_all:
        summary = {
            "test_image": args.image,
            "detectors_tested": list(results_all.keys()),
            "results": results_all,
        }
        summary_path = os.path.join(args.output, f"{image_name}_all_results.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\n汇总报告已保存: {summary_path}")

    print("\n" + "="*60)
    print("测试完成!")
    print("="*60)


if __name__ == "__main__":
    main()
PYTHON_SCRIPT

# =============================================================================
# 执行测试
# =============================================================================

echo ""
echo "============================================================"
echo "           行人检测模块测试"
echo "============================================================"
echo ""
echo "图像路径: ${IMAGE_PATH}"
echo "输出目录: ${OUTPUT_DIR}"
echo "检测器:   ${DETECTOR}"
echo "置信度:   ${CONFIDENCE}"
echo "设备:     ${DEVICE}"
echo "============================================================"
echo ""

# 运行Python测试脚本
python3 "${OUTPUT_DIR}/run_detection.py" \
    --image "${IMAGE_PATH}" \
    --output "${OUTPUT_DIR}" \
    --detector "${DETECTOR}" \
    --confidence "${CONFIDENCE}" \
    --device "${DEVICE}"

# =============================================================================
# 测试完成
# =============================================================================

echo ""
echo "============================================================"
echo "           测试结果输出"
echo "============================================================"
echo ""
echo "输出目录: ${OUTPUT_DIR}"
echo ""
echo "生成的文件:"
echo "  - images/          : 打框图像"
echo "  - *_results.json  : 完整JSON结果"
echo "  - *_results.csv   : CSV格式结果"
echo "  - *_summary.txt   : 文本摘要"
echo "  - *_all_results.json : 多检测器汇总"
echo ""
echo "============================================================"
