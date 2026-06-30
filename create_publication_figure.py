#!/usr/bin/env python3
"""
生成适合学术期刊发表的高质量轨迹可视化图

特点:
- IEEE/Springer期刊标准
- 高分辨率矢量图
- 专业配色方案
- 清晰的图例和标注
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib import rc
import pathlib

# 设置学术期刊风格
plt.style.use('seaborn-v0_8-paper')
rc('font', **{'family': 'serif', 'serif': ['Times New Roman'], 'size': 11})
rc('text', usetex=False)  # 如果系统有LaTeX，设为True
rc('axes', linewidth=1.2)
rc('grid', linewidth=0.5, alpha=0.3)

def load_trajectory_data(trajectory_file):
    """加载轨迹数据"""
    with open(trajectory_file, 'r') as f:
        return json.load(f)

def extract_trajectories(trajectory_data):
    """提取轨迹数据"""
    robot_positions = []
    pedestrian_trajectories = {}
    
    for step_data in trajectory_data:
        if not step_data or 'robot' not in step_data:
            continue
        
        # 机器人轨迹
        robot_pos = step_data['robot']['position']
        robot_positions.append(robot_pos)
        
        # 行人轨迹
        for ped in step_data.get('pedestrians', []):
            ped_id = ped['id']
            ped_pos = ped['position']
            
            if ped_id not in pedestrian_trajectories:
                pedestrian_trajectories[ped_id] = []
            pedestrian_trajectories[ped_id].append(ped_pos)
    
    return {
        'robot': np.array(robot_positions),
        'pedestrians': {
            ped_id: np.array(positions) 
            for ped_id, positions in pedestrian_trajectories.items()
        }
    }

def create_publication_figure(trajectories, output_path, dpi=600):
    """
    创建学术期刊级别的图表
    
    符合IEEE和Springer期刊标准:
    - 单栏图: 3.5英寸宽
    - 双栏图: 7.16英寸宽
    - DPI: 300-600
    - 格式: EPS/PDF (矢量) 或高分辨率PNG
    """
    
    # 创建图表 - 双栏宽度
    fig, ax = plt.subplots(figsize=(7.16, 6), dpi=dpi)
    
    # 配色方案 - 专业且色盲友好
    robot_color = '#1f77b4'  # 蓝色
    robot_start_color = '#2ca02c'  # 绿色
    robot_goal_color = '#d62728'  # 红色
    pedestrian_colors = [
        '#ff7f0e',  # 橙色
        '#9467bd',  # 紫色
        '#8c564b',  # 棕色
        '#e377c2',  # 粉色
        '#7f7f7f',  # 灰色
        '#bcbd22',  # 黄绿色
    ]
    
    # 提取机器人轨迹
    robot_pos = trajectories['robot']
    robot_x = robot_pos[:, 0]
    robot_z = robot_pos[:, 2]
    
    # 绘制机器人轨迹
    ax.plot(robot_x, robot_z, 
            color=robot_color, 
            linewidth=2.5, 
            label='Robot Trajectory',
            zorder=3,
            alpha=0.9)
    
    # 起点标记
    ax.scatter(robot_x[0], robot_z[0], 
              s=200, 
              color=robot_start_color,
              marker='o',
              edgecolors='black',
              linewidths=2,
              label='Start',
              zorder=5)
    
    # 终点标记
    ax.scatter(robot_x[-1], robot_z[-1],
              s=200,
              color=robot_goal_color,
              marker='s',
              edgecolors='black',
              linewidths=2,
              label='Goal',
              zorder=5)
    
    # 添加方向箭头（每隔N步）
    arrow_interval = max(len(robot_pos) // 8, 1)
    for i in range(0, len(robot_pos) - 1, arrow_interval):
        if i + 1 < len(robot_pos):
            dx = robot_x[i + 1] - robot_x[i]
            dz = robot_z[i + 1] - robot_z[i]
            
            if np.sqrt(dx**2 + dz**2) > 0.1:  # 只在移动距离足够时绘制
                arrow = FancyArrowPatch(
                    (robot_x[i], robot_z[i]),
                    (robot_x[i] + dx * 0.8, robot_z[i] + dz * 0.8),
                    arrowstyle='->,head_width=0.4,head_length=0.6',
                    color=robot_color,
                    linewidth=1.5,
                    alpha=0.7,
                    zorder=4
                )
                ax.add_patch(arrow)
    
    # 绘制行人轨迹
    for idx, (ped_id, ped_pos) in enumerate(trajectories['pedestrians'].items()):
        if len(ped_pos) == 0:
            continue
        
        color = pedestrian_colors[idx % len(pedestrian_colors)]
        ped_x = ped_pos[:, 0]
        ped_z = ped_pos[:, 2]
        
        # 行人轨迹（虚线）
        ax.plot(ped_x, ped_z,
               linestyle='--',
               color=color,
               linewidth=2,
               label=f'Pedestrian {ped_id + 1}',
               alpha=0.8,
               zorder=2)
        
        # 行人起点
        ax.scatter(ped_x[0], ped_z[0],
                  s=100,
                  color=color,
                  marker='o',
                  edgecolors='black',
                  linewidths=1.5,
                  alpha=0.8,
                  zorder=4)
        
        # 行人终点
        ax.scatter(ped_x[-1], ped_z[-1],
                  s=100,
                  color=color,
                  marker='x',
                  linewidths=2.5,
                  zorder=4)
    
    # 设置坐标轴
    ax.set_xlabel('X Position (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Z Position (m)', fontsize=12, fontweight='bold')
    ax.set_title('Multi-Agent Navigation Trajectories', 
                fontsize=14, 
                fontweight='bold',
                pad=15)
    
    # 网格
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
    
    # 设置相等的坐标比例
    ax.set_aspect('equal', adjustable='box')
    
    # 图例 - 放在最佳位置
    legend = ax.legend(
        loc='best',
        fontsize=10,
        frameon=True,
        fancybox=True,
        shadow=True,
        framealpha=0.95,
        edgecolor='black',
        facecolor='white'
    )
    legend.get_frame().set_linewidth(1.2)
    
    # 添加统计信息文本框
    num_pedestrians = len(trajectories['pedestrians'])
    trajectory_length = len(robot_pos)
    
    textstr = f'Episode Statistics:\n'
    textstr += f'• Time steps: {trajectory_length}\n'
    textstr += f'• Pedestrians: {num_pedestrians}\n'
    textstr += f'• Robot path length: {calculate_path_length(robot_pos):.2f} m'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black', linewidth=1.2)
    ax.text(0.02, 0.98, textstr, 
           transform=ax.transAxes,
           fontsize=9,
           verticalalignment='top',
           bbox=props)
    
    # 美化刻度
    ax.tick_params(axis='both', which='major', labelsize=10, width=1.2, length=6)
    ax.tick_params(axis='both', which='minor', width=0.8, length=3)
    
    # 紧凑布局
    plt.tight_layout()
    
    # 保存多种格式
    base_path = pathlib.Path(output_path)
    
    # PNG - 高分辨率位图
    plt.savefig(base_path.with_suffix('.png'), 
               dpi=dpi, 
               bbox_inches='tight',
               facecolor='white',
               edgecolor='none')
    print(f"✓ 已保存PNG: {base_path.with_suffix('.png')}")
    
    # PDF - 矢量图（推荐用于期刊）
    plt.savefig(base_path.with_suffix('.pdf'),
               format='pdf',
               bbox_inches='tight',
               facecolor='white',
               edgecolor='none')
    print(f"✓ 已保存PDF: {base_path.with_suffix('.pdf')}")
    
    # EPS - 矢量图（某些期刊要求）
    try:
        plt.savefig(base_path.with_suffix('.eps'),
                   format='eps',
                   bbox_inches='tight',
                   facecolor='white',
                   edgecolor='none')
        print(f"✓ 已保存EPS: {base_path.with_suffix('.eps')}")
    except Exception as e:
        print(f"⚠ EPS保存失败: {e}")
    
    plt.close()

def calculate_path_length(positions):
    """计算路径总长度"""
    if len(positions) < 2:
        return 0.0
    
    diffs = np.diff(positions, axis=0)
    distances = np.sqrt(np.sum(diffs**2, axis=1))
    return np.sum(distances)

def create_simplified_figure(trajectories, output_path, dpi=600):
    """
    创建简化版本 - 更清晰的黑白图（适合某些期刊）
    """
    fig, ax = plt.subplots(figsize=(7.16, 6), dpi=dpi)
    
    robot_pos = trajectories['robot']
    robot_x = robot_pos[:, 0]
    robot_z = robot_pos[:, 2]
    
    # 机器人轨迹 - 粗实线
    ax.plot(robot_x, robot_z,
           color='black',
           linewidth=3,
           label='Robot',
           zorder=3)
    
    # 起点和终点
    ax.scatter(robot_x[0], robot_z[0],
              s=200,
              color='white',
              marker='o',
              edgecolors='black',
              linewidths=2.5,
              label='Start',
              zorder=5)
    
    ax.scatter(robot_x[-1], robot_z[-1],
              s=200,
              color='black',
              marker='s',
              edgecolors='black',
              linewidths=2.5,
              label='Goal',
              zorder=5)
    
    # 行人轨迹 - 虚线，不同样式
    linestyles = ['--', '-.', ':']
    for idx, (ped_id, ped_pos) in enumerate(trajectories['pedestrians'].items()):
        if len(ped_pos) == 0:
            continue
        
        style = linestyles[idx % len(linestyles)]
        ped_x = ped_pos[:, 0]
        ped_z = ped_pos[:, 2]
        
        ax.plot(ped_x, ped_z,
               linestyle=style,
               color='black',
               linewidth=2,
               label=f'Ped. {ped_id + 1}',
               alpha=0.7,
               zorder=2)
        
        ax.scatter(ped_x[0], ped_z[0],
                  s=80,
                  color='white',
                  marker='o',
                  edgecolors='black',
                  linewidths=1.5,
                  zorder=4)
    
    ax.set_xlabel('X Position (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Z Position (m)', fontsize=12, fontweight='bold')
    ax.set_title('Multi-Agent Navigation Trajectories',
                fontsize=14,
                fontweight='bold',
                pad=15)
    
    ax.grid(True, linestyle='--', alpha=0.4, linewidth=0.5, color='gray')
    ax.set_aspect('equal', adjustable='box')
    
    ax.legend(loc='best', fontsize=10, frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    
    base_path = pathlib.Path(output_path)
    plt.savefig(base_path.with_name(base_path.stem + '_bw.png'),
               dpi=dpi,
               bbox_inches='tight',
               facecolor='white')
    print(f"✓ 已保存黑白版: {base_path.with_name(base_path.stem + '_bw.png')}")
    
    plt.savefig(base_path.with_name(base_path.stem + '_bw.pdf'),
               format='pdf',
               bbox_inches='tight',
               facecolor='white')
    print(f"✓ 已保存黑白PDF: {base_path.with_name(base_path.stem + '_bw.pdf')}")
    
    plt.close()

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='生成学术期刊级别的轨迹可视化图',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python create_publication_figure.py \\
      --trajectory_file data/collect_data/train/scene/episode/trajectories/0.json \\
      --output publication_figure \\
      --dpi 600
        """
    )
    
    parser.add_argument('--trajectory_file', type=str, required=True,
                       help='轨迹数据文件路径')
    parser.add_argument('--output', type=str, default='publication_figure',
                       help='输出文件路径（不含扩展名）')
    parser.add_argument('--dpi', type=int, default=600,
                       help='图像分辨率 (默认: 600)')
    parser.add_argument('--bw', action='store_true',
                       help='同时生成黑白版本')
    
    args = parser.parse_args()
    
    # 加载数据
    print(f"正在加载轨迹数据: {args.trajectory_file}")
    trajectory_data = load_trajectory_data(args.trajectory_file)
    
    print(f"轨迹数据包含 {len(trajectory_data)} 个时间步")
    
    # 提取轨迹
    print("提取轨迹...")
    trajectories = extract_trajectories(trajectory_data)
    
    print(f"机器人轨迹点数: {len(trajectories['robot'])}")
    print(f"行人数量: {len(trajectories['pedestrians'])}")
    
    # 生成彩色版本
    print("\n生成学术期刊级别图表...")
    create_publication_figure(trajectories, args.output, dpi=args.dpi)
    
    # 生成黑白版本
    if args.bw:
        print("\n生成黑白版本...")
        create_simplified_figure(trajectories, args.output, dpi=args.dpi)
    
    print("\n✓ 完成！")
    print(f"\n生成的文件可直接用于:")
    print("  • IEEE期刊投稿")
    print("  • Springer期刊投稿")
    print("  • 会议论文")
    print("  • 学位论文")

if __name__ == "__main__":
    main()

