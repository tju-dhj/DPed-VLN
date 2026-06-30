#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成测试图像脚本
================

生成包含模拟行人的测试图像，用于测试行人检测模块。

运行示例：
```bash
python generate_test_image.py --output ./test_images
```
"""

import os
import argparse
import random

import cv2
import numpy as np


def generate_person_image(width, height, x, y, person_width, person_height, person_id):
    """
    在给定位置生成模拟行人图像

    Args:
        width, height: 画布尺寸
        x, y: 行人中心位置
        person_width, person_height: 行人框尺寸
        person_id: 行人ID
    """
    img = np.ones((height, width, 3), dtype=np.uint8) * 200  # 浅灰色背景

    # 计算行人边界框
    x1 = int(x - person_width / 2)
    y1 = int(y - person_height / 2)
    x2 = int(x + person_width / 2)
    y2 = int(y + person_height / 2)

    # 限制在图像范围内
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)

    # 绘制模拟行人（使用随机颜色）
    base_color = (
        random.randint(50, 100),   # B
        random.randint(100, 150),   # G
        random.randint(150, 200),   # R
    )

    # 身体（矩形）
    cv2.rectangle(img, (x1, y1), (x2, y2), base_color, -1)

    # 头部（圆形）
    head_radius = int(person_width / 3)
    head_center = (int((x1 + x2) / 2), int(y1 + head_radius))
    cv2.circle(img, head_center, head_radius, (220, 180, 160), -1)

    # 添加边框
    cv2.rectangle(img, (x1, y1), (x2, y2), (50, 50, 50), 2)

    # 添加编号标签
    label = f"Person-{person_id}"
    cv2.putText(img, label, (x1, y1 - 10),
                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return img, (x1, y1, x2, y2)


def generate_test_images(output_dir, num_images=5):
    """
    生成多张测试图像

    Args:
        output_dir: 输出目录
        num_images: 图像数量
    """
    os.makedirs(output_dir, exist_ok=True)

    image_sizes = [
        (640, 480, "640x480"),
        (800, 600, "800x600"),
        (1280, 720, "1280x720"),
        (1920, 1080, "1920x1080"),
    ]

    for i in range(num_images):
        # 随机选择图像尺寸
        width, height, size_name = random.choice(image_sizes)

        # 创建空白图像
        img = np.ones((height, width, 3), dtype=np.uint8) * 220

        # 添加背景纹理
        for _ in range(50):
            px, py = random.randint(0, width-1), random.randint(0, height-1)
            cv2.circle(img, (px, py), random.randint(1, 3), (180, 180, 180), -1)

        # 随机添加一些行人
        num_persons = random.randint(0, 4)
        annotations = []

        for p in range(num_persons):
            person_id = p + 1

            # 根据位置决定行人大小（远处小，近处大）
            y_base = random.randint(height // 4, height - 50)
            x_center = random.randint(width // 4, width * 3 // 4)

            # 位置越靠下，行人越大（模拟近处行人）
            height_ratio = y_base / height
            person_height = int(50 + height_ratio * 300)  # 50-350
            person_width = int(person_height * 0.4)

            # 生成行人
            person_img, bbox = generate_person_image(
                width, height, x_center, y_base,
                person_width, person_height, person_id
            )

            # 将行人叠加到主图像
            mask = np.all(person_img > [100, 100, 100], axis=-1)
            img[mask] = person_img[mask]

            annotations.append({
                "id": person_id,
                "bbox": list(bbox),
                "center": [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2],
            })

        # 保存图像
        img_name = f"test_scene_{i+1:02d}_{size_name}.jpg"
        img_path = os.path.join(output_dir, img_name)
        cv2.imwrite(img_path, img)
        print(f"生成图像: {img_path}")

        # 保存标注
        import json
        anno_name = f"test_scene_{i+1:02d}_{size_name}.json"
        anno_path = os.path.join(output_dir, anno_name)
        with open(anno_path, 'w') as f:
            json.dump({
                "image": img_name,
                "size": size_name,
                "num_persons": num_persons,
                "annotations": annotations,
            }, f, indent=2)
        print(f"保存标注: {anno_path}")

    print(f"\n生成完成! 共 {num_images} 张图像保存在: {output_dir}")


def generate_single_with_ground_truth(output_path, num_persons=3):
    """
    生成一张带有真实标注的测试图像

    Args:
        output_path: 输出路径
        num_persons: 行人数量
    """
    width, height = 1280, 720

    # 创建图像
    img = np.ones((height, width, 3), dtype=np.uint8) * 220

    # 添加背景纹理
    for _ in range(100):
        px, py = random.randint(0, width-1), random.randint(0, height-1)
        cv2.circle(img, (px, py), random.randint(1, 3), (180, 180, 180), -1)

    annotations = []

    for p in range(num_persons):
        person_id = p + 1

        # 位置
        x_center = width // (num_persons + 1) * (p + 1)
        y_base = height - 50 - p * 30  # 不同高度

        # 大小
        person_height = 200 - p * 30
        person_width = int(person_height * 0.4)

        # 生成行人
        person_img, bbox = generate_person_image(
            width, height, x_center, y_base,
            person_width, person_height, person_id
        )

        # 叠加
        mask = np.all(person_img > [100, 100, 100], axis=-1)
        img[mask] = person_img[mask]

        annotations.append({
            "id": person_id,
            "bbox": list(bbox),
            "area": (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
        })

    # 保存图像
    cv2.imwrite(output_path, img)
    print(f"生成图像: {output_path}")

    # 保存标注
    import json
    anno_path = output_path.rsplit('.', 1)[0] + '_ground_truth.json'
    with open(anno_path, 'w') as f:
        json.dump({
            "image": os.path.basename(output_path),
            "size": f"{width}x{height}",
            "annotations": annotations,
        }, f, indent=2)
    print(f"保存标注: {anno_path}")


def main():
    parser = argparse.ArgumentParser(description='生成测试图像')
    parser.add_argument('--output', '-o', default='./test_images',
                       help='输出目录')
    parser.add_argument('--num', '-n', type=int, default=5,
                       help='生成的图像数量')
    parser.add_argument('--single', '-s', action='store_true',
                       help='生成单张标准测试图像')
    parser.add_argument('--persons', '-p', type=int, default=3,
                       help='行人数量（单张模式）')
    args = parser.parse_args()

    if args.single:
        output_path = os.path.join(args.output, 'single_test.jpg')
        os.makedirs(args.output, exist_ok=True)
        generate_single_with_ground_truth(output_path, args.persons)
    else:
        generate_test_images(args.output, args.num)


if __name__ == "__main__":
    main()
