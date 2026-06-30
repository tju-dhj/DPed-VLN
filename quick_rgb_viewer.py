#!/usr/bin/env python3
"""
快速查看 agent_0_articulated_agent_jaw_rgb 数据
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

def quick_view_rgb(rgb_data, title="RGB Data"):
    """快速查看RGB数据"""
    
    # 检查数据
    if rgb_data is None:
        print("❌ RGB数据为空！")
        return
    
    print(f"🎨 正在显示: {title}")
    print(f"📊 数据形状: {rgb_data.shape}")
    print(f"📊 数据类型: {rgb_data.dtype}")
    print(f"📊 数值范围: [{rgb_data.min()}, {rgb_data.max()}]")
    
    # 转换为numpy数组
    if not isinstance(rgb_data, np.ndarray):
        rgb_data = np.array(rgb_data)
    
    # 确保数据类型正确
    if rgb_data.dtype != np.uint8:
        if rgb_data.max() <= 1.0:
            rgb_data = (rgb_data * 255).astype(np.uint8)
        else:
            rgb_data = rgb_data.astype(np.uint8)
    
    # 创建图像显示
    plt.figure(figsize=(12, 8))
    
    # 原始RGB图像
    plt.subplot(2, 2, 1)
    plt.imshow(rgb_data)
    plt.title('🎯 Original RGB Image', fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # 灰度图像
    plt.subplot(2, 2, 2)
    gray = cv2.cvtColor(rgb_data, cv2.COLOR_RGB2GRAY)
    plt.imshow(gray, cmap='gray')
    plt.title('⚫ Grayscale', fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # 颜色直方图
    plt.subplot(2, 2, 3)
    colors = ['red', 'green', 'blue']
    for i, color in enumerate(colors):
        hist = cv2.calcHist([rgb_data], [i], None, [256], [0, 256])
        plt.plot(hist, color=color, alpha=0.8, linewidth=2, label=f'{color.upper()}')
    plt.title('📊 Color Histogram', fontsize=14, fontweight='bold')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 边缘检测
    plt.subplot(2, 2, 4)
    edges = cv2.Canny(gray, 50, 150)
    plt.imshow(edges, cmap='hot')
    plt.title('🔥 Edge Detection', fontsize=14, fontweight='bold')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 保存图像
    output_path = 'rgb_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 图像已保存到: {output_path}")
    
    # 保存原始RGB图像
    original_path = 'rgb_original.jpg'
    cv2.imwrite(original_path, cv2.cvtColor(rgb_data, cv2.COLOR_RGB2BGR))
    print(f"✅ 原始RGB图像已保存到: {original_path}")
    
    print("🎉 可视化完成！")

# 使用示例
if __name__ == "__main__":
    # 这里替换为你的实际RGB数据
    # 示例：假设你的数据是这样的
    # rgb_data = observations['agent_0_articulated_agent_jaw_rgb']
    
    # 创建一个示例数据用于测试
    print("🚀 创建示例RGB数据...")
    sample_rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 添加一些模式让图像更有趣
    cv2.circle(sample_rgb, (320, 240), 100, (255, 0, 0), -1)  # 红色圆圈
    cv2.rectangle(sample_rgb, (100, 100), (200, 200), (0, 255, 0), -1)  # 绿色矩形
    cv2.putText(sample_rgb, 'AGENT RGB', (250, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    
    # 显示数据
    quick_view_rgb(sample_rgb, "Agent RGB Data")
