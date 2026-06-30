#!/usr/bin/env python3
"""
炫酷的RGB数据可视化脚本
用于显示和保存 agent_0_articulated_agent_jaw_rgb 数据
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import os
import sys
import time
from datetime import datetime
import argparse

# 炫酷输出工具
class CoolVisualizer:
    """炫酷的可视化工具类"""
    
    # ANSI颜色代码
    class Colors:
        RED = '\033[91m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        BLUE = '\033[94m'
        MAGENTA = '\033[95m'
        CYAN = '\033[96m'
        WHITE = '\033[97m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'
        END = '\033[0m'
        BG_RED = '\033[41m'
        BG_GREEN = '\033[42m'
        BG_YELLOW = '\033[43m'
        BG_BLUE = '\033[44m'
        BG_MAGENTA = '\033[45m'
        BG_CYAN = '\033[46m'
    
    def __init__(self):
        self.start_time = time.time()
    
    def colorize(self, text: str, color: str = None, bg_color: str = None, bold: bool = False) -> str:
        """为文本添加颜色"""
        if not sys.stdout.isatty():
            return text
        
        result = text
        if color:
            result = f"{color}{result}"
        if bg_color:
            result = f"{bg_color}{result}"
        if bold:
            result = f"{self.Colors.BOLD}{result}"
        result += self.Colors.END
        return result
    
    def print_header(self, title: str):
        """打印炫酷标题"""
        width = 80
        border = "═" * width
        title_line = f"║ {title.center(width-4)} ║"
        
        print(self.colorize("╔" + border[1:-1] + "╗", self.Colors.CYAN, bold=True))
        print(self.colorize(title_line, self.Colors.CYAN, bold=True))
        print(self.colorize("╚" + border[1:-1] + "╝", self.Colors.CYAN, bold=True))
        print()
    
    def print_success(self, message: str):
        """打印成功消息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"{self.colorize('✓', self.Colors.GREEN, bold=True)} "
              f"{self.colorize(f'[{timestamp}]', self.Colors.CYAN)} "
              f"{self.colorize(message, self.Colors.GREEN, bold=True)}")
    
    def print_info(self, message: str):
        """打印信息消息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"{self.colorize('ℹ', self.Colors.BLUE, bold=True)} "
              f"{self.colorize(f'[{timestamp}]', self.Colors.CYAN)} "
              f"{self.colorize(message, self.Colors.BLUE)}")
    
    def print_warning(self, message: str):
        """打印警告消息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"{self.colorize('⚠', self.Colors.YELLOW, bold=True)} "
              f"{self.colorize(f'[{timestamp}]', self.Colors.CYAN)} "
              f"{self.colorize(message, self.Colors.YELLOW, bold=True)}")
    
    def print_error(self, message: str):
        """打印错误消息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"{self.colorize('✗', self.Colors.RED, bold=True)} "
              f"{self.colorize(f'[{timestamp}]', self.Colors.CYAN)} "
              f"{self.colorize(message, self.Colors.RED, bold=True)}")
    
    def print_progress_bar(self, current: int, total: int, prefix: str = "Progress", 
                          suffix: str = "Complete", length: int = 50):
        """打印进度条"""
        if not sys.stdout.isatty():
            return
        
        percent = 100 * (current / float(total))
        filled_length = int(length * current // total)
        bar = '█' * filled_length + '░' * (length - filled_length)
        
        # 根据进度选择颜色
        if percent < 30:
            color = self.Colors.RED
        elif percent < 70:
            color = self.Colors.YELLOW
        else:
            color = self.Colors.GREEN
        
        print(f'\r{self.colorize(prefix, self.Colors.CYAN, bold=True)} '
              f'|{self.colorize(bar, color)}| '
              f'{self.colorize(f"{percent:.1f}%", self.Colors.WHITE, bold=True)} '
              f'{self.colorize(suffix, self.Colors.CYAN)}', end='\r')
        
        if current == total:
            print()

def analyze_rgb_data(rgb_data):
    """分析RGB数据并返回统计信息"""
    if rgb_data is None:
        return None
    
    # 转换为numpy数组
    if not isinstance(rgb_data, np.ndarray):
        rgb_data = np.array(rgb_data)
    
    # 基本统计信息
    stats = {
        'shape': rgb_data.shape,
        'dtype': rgb_data.dtype,
        'min_value': rgb_data.min(),
        'max_value': rgb_data.max(),
        'mean_value': rgb_data.mean(),
        'std_value': rgb_data.std(),
        'unique_colors': len(np.unique(rgb_data.reshape(-1, rgb_data.shape[-1]), axis=0))
    }
    
    # 颜色分布分析
    if len(rgb_data.shape) == 3:
        # 分别分析R、G、B通道
        for i, channel in enumerate(['R', 'G', 'B']):
            channel_data = rgb_data[:, :, i]
            stats[f'{channel}_mean'] = channel_data.mean()
            stats[f'{channel}_std'] = channel_data.std()
            stats[f'{channel}_min'] = channel_data.min()
            stats[f'{channel}_max'] = channel_data.max()
    
    return stats

def create_enhanced_visualization(rgb_data, output_path=None):
    """创建增强的可视化图像"""
    if rgb_data is None:
        return None
    
    # 转换为numpy数组
    if not isinstance(rgb_data, np.ndarray):
        rgb_data = np.array(rgb_data)
    
    # 确保数据类型正确
    if rgb_data.dtype != np.uint8:
        if rgb_data.max() <= 1.0:
            rgb_data = (rgb_data * 255).astype(np.uint8)
        else:
            rgb_data = rgb_data.astype(np.uint8)
    
    # 创建matplotlib图形
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('🎯 Agent RGB Data Analysis', fontsize=16, fontweight='bold')
    
    # 1. 原始RGB图像
    axes[0, 0].imshow(rgb_data)
    axes[0, 0].set_title('📸 Original RGB Image', fontweight='bold')
    axes[0, 0].axis('off')
    
    # 2. 灰度图像
    gray = cv2.cvtColor(rgb_data, cv2.COLOR_RGB2GRAY)
    axes[0, 1].imshow(gray, cmap='gray')
    axes[0, 1].set_title('⚫ Grayscale View', fontweight='bold')
    axes[0, 1].axis('off')
    
    # 3. 颜色直方图
    colors = ['red', 'green', 'blue']
    for i, color in enumerate(colors):
        hist = cv2.calcHist([rgb_data], [i], None, [256], [0, 256])
        axes[1, 0].plot(hist, color=color, alpha=0.7, label=f'{color.upper()} Channel')
    axes[1, 0].set_title('📊 Color Histogram', fontweight='bold')
    axes[1, 0].set_xlabel('Pixel Value')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 边缘检测
    edges = cv2.Canny(gray, 50, 150)
    axes[1, 1].imshow(edges, cmap='hot')
    axes[1, 1].set_title('🔥 Edge Detection', fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        return output_path
    
    return fig

def create_animated_visualization(rgb_sequence, output_path=None):
    """创建动画可视化"""
    if not rgb_sequence or len(rgb_sequence) == 0:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title('🎬 RGB Data Animation', fontsize=16, fontweight='bold')
    
    def animate(frame):
        ax.clear()
        if frame < len(rgb_sequence):
            rgb_data = rgb_sequence[frame]
            if isinstance(rgb_data, np.ndarray):
                ax.imshow(rgb_data)
                ax.set_title(f'Frame {frame + 1}/{len(rgb_sequence)}', fontweight='bold')
        ax.axis('off')
        return ax,
    
    anim = animation.FuncAnimation(fig, animate, frames=len(rgb_sequence), 
                                 interval=200, blit=False, repeat=True)
    
    if output_path:
        anim.save(output_path, writer='pillow', fps=5)
        return output_path
    
    return anim

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='炫酷的RGB数据可视化工具')
    parser.add_argument('--rgb_data', type=str, help='RGB数据文件路径')
    parser.add_argument('--output_dir', type=str, default='./rgb_visualization', 
                       help='输出目录')
    parser.add_argument('--show_stats', action='store_true', help='显示统计信息')
    parser.add_argument('--create_animation', action='store_true', help='创建动画')
    
    args = parser.parse_args()
    
    # 初始化炫酷输出工具
    visualizer = CoolVisualizer()
    
    # 显示标题
    visualizer.print_header("🎨 RGB数据可视化工具")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    visualizer.print_success(f"输出目录已创建: {args.output_dir}")
    
    # 示例RGB数据（如果用户没有提供）
    if not args.rgb_data:
        visualizer.print_info("使用示例RGB数据...")
        # 创建一个示例RGB图像
        sample_rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        # 添加一些模式
        cv2.circle(sample_rgb, (320, 240), 100, (255, 0, 0), -1)
        cv2.rectangle(sample_rgb, (100, 100), (200, 200), (0, 255, 0), -1)
        rgb_data = sample_rgb
    else:
        # 从文件加载RGB数据
        try:
            rgb_data = np.load(args.rgb_data)
            visualizer.print_success(f"成功加载RGB数据: {args.rgb_data}")
        except Exception as e:
            visualizer.print_error(f"加载RGB数据失败: {e}")
            return
    
    # 分析数据
    visualizer.print_info("正在分析RGB数据...")
    stats = analyze_rgb_data(rgb_data)
    
    if stats:
        visualizer.print_success("数据分析完成！")
        if args.show_stats:
            visualizer.print_info("📊 数据统计信息:")
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"   {key}: {value:.4f}")
                else:
                    print(f"   {key}: {value}")
    
    # 创建可视化
    visualizer.print_info("正在创建可视化...")
    
    # 保存原始图像
    original_path = os.path.join(args.output_dir, 'original_rgb.jpg')
    cv2.imwrite(original_path, cv2.cvtColor(rgb_data, cv2.COLOR_RGB2BGR))
    visualizer.print_success(f"原始图像已保存: {original_path}")
    
    # 创建增强可视化
    enhanced_path = os.path.join(args.output_dir, 'enhanced_analysis.png')
    create_enhanced_visualization(rgb_data, enhanced_path)
    visualizer.print_success(f"增强分析图已保存: {enhanced_path}")
    
    # 创建动画（如果有序列数据）
    if args.create_animation and isinstance(rgb_data, list):
        anim_path = os.path.join(args.output_dir, 'rgb_animation.gif')
        create_animated_visualization(rgb_data, anim_path)
        visualizer.print_success(f"动画已保存: {anim_path}")
    
    # 显示完成信息
    visualizer.print_success("🎉 可视化完成！")
    visualizer.print_info(f"所有文件已保存到: {args.output_dir}")

if __name__ == "__main__":
    main()
