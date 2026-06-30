#!/usr/bin/env python3
"""
轨迹可视化脚本

用于可视化采集数据中机器人和行人的运动轨迹，生成适合论文使用的图表。
"""

import json
import pathlib
import argparse
from typing import List, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.collections import LineCollection
import matplotlib.animation as animation


def load_trajectory_data(episode_path: pathlib.Path) -> List[Dict]:
    """
    加载单个episode的轨迹数据
    
    Args:
        episode_path: episode文件夹路径
        
    Returns:
        轨迹数据列表
    """
    trajectory_file = episode_path / "trajectories" / "0.json"
    if not trajectory_file.exists():
        return []
    
    with open(trajectory_file, "r") as f:
        return json.load(f)


def extract_trajectories(trajectory_data: List[Dict]) -> Dict:
    """
    从轨迹数据中提取机器人和行人的位置序列
    
    Args:
        trajectory_data: 原始轨迹数据
        
    Returns:
        提取后的轨迹字典
    """
    robot_positions = []
    robot_rotations = []
    pedestrian_trajectories = {}
    
    for step_data in trajectory_data:
        if not step_data or 'robot' not in step_data:
            continue
            
        # 提取机器人数据
        robot_pos = step_data['robot'].get('position', [0, 0, 0])
        robot_rot = step_data['robot'].get('rotation', [1, 0, 0, 0])
        robot_positions.append(robot_pos)
        robot_rotations.append(robot_rot)
        
        # 提取行人数据
        for ped in step_data.get('pedestrians', []):
            ped_id = ped.get('id', 0)
            ped_pos = ped.get('position', [0, 0, 0])
            
            if ped_id not in pedestrian_trajectories:
                pedestrian_trajectories[ped_id] = []
            pedestrian_trajectories[ped_id].append(ped_pos)
    
    return {
        'robot': {
            'positions': np.array(robot_positions),
            'rotations': np.array(robot_rotations)
        },
        'pedestrians': {
            ped_id: np.array(positions) 
            for ped_id, positions in pedestrian_trajectories.items()
        }
    }


def plot_trajectories_2d(trajectories: Dict, 
                         output_path: pathlib.Path = None,
                         title: str = "Agent Trajectories",
                         show_arrows: bool = True,
                         show_grid: bool = True,
                         figsize: Tuple[int, int] = (10, 10),
                         dpi: int = 300):
    """
    绘制2D俯视图轨迹
    
    Args:
        trajectories: 轨迹数据
        output_path: 输出文件路径
        title: 图表标题
        show_arrows: 是否显示方向箭头
        show_grid: 是否显示网格
        figsize: 图表大小
        dpi: 图像分辨率
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # 提取机器人轨迹
    robot_pos = trajectories['robot']['positions']
    robot_x = robot_pos[:, 0]
    robot_z = robot_pos[:, 2]  # 使用z作为y坐标（俯视图）
    
    # 绘制机器人轨迹
    ax.plot(robot_x, robot_z, 'b-', linewidth=2, label='Robot', alpha=0.7)
    ax.scatter(robot_x[0], robot_z[0], c='green', s=100, marker='o', 
              label='Robot Start', zorder=5, edgecolors='black', linewidths=1.5)
    ax.scatter(robot_x[-1], robot_z[-1], c='red', s=100, marker='s', 
              label='Robot Goal', zorder=5, edgecolors='black', linewidths=1.5)
    
    # 绘制机器人方向箭头（每隔几步）
    if show_arrows and len(robot_pos) > 1:
        arrow_interval = max(len(robot_pos) // 10, 1)
        robot_rot = trajectories['robot']['rotations']
        
        for i in range(0, len(robot_pos), arrow_interval):
            if i >= len(robot_rot):
                break
            # 从四元数计算yaw角
            qw, qx, qy, qz = robot_rot[i]
            yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy**2 + qz**2))
            
            # 计算箭头方向
            dx = np.sin(yaw) * 0.3
            dz = np.cos(yaw) * 0.3
            
            arrow = FancyArrowPatch(
                (robot_x[i], robot_z[i]),
                (robot_x[i] + dx, robot_z[i] + dz),
                arrowstyle='->', mutation_scale=15, 
                color='blue', alpha=0.5, linewidth=1.5
            )
            ax.add_patch(arrow)
    
    # 绘制行人轨迹
    colors = plt.cm.rainbow(np.linspace(0, 1, len(trajectories['pedestrians'])))
    
    for idx, (ped_id, ped_pos) in enumerate(trajectories['pedestrians'].items()):
        if len(ped_pos) == 0:
            continue
            
        ped_x = ped_pos[:, 0]
        ped_z = ped_pos[:, 2]
        
        color = colors[idx]
        ax.plot(ped_x, ped_z, '--', color=color, linewidth=1.5, 
               label=f'Pedestrian {ped_id}', alpha=0.7)
        ax.scatter(ped_x[0], ped_z[0], c=[color], s=50, marker='o', 
                  zorder=4, edgecolors='black', linewidths=1)
        ax.scatter(ped_x[-1], ped_z[-1], c=[color], s=50, marker='x', 
                  zorder=4, linewidths=2)
    
    # 设置图表样式
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Z Position (m)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.set_aspect('equal', adjustable='box')
    
    if show_grid:
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"轨迹图已保存到: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_trajectories_with_heatmap(trajectories: Dict,
                                   output_path: pathlib.Path = None,
                                   title: str = "Agent Trajectories with Density",
                                   figsize: Tuple[int, int] = (12, 10),
                                   dpi: int = 300):
    """
    绘制带密度热力图的轨迹
    
    Args:
        trajectories: 轨迹数据
        output_path: 输出文件路径
        title: 图表标题
        figsize: 图表大小
        dpi: 图像分辨率
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # 收集所有位置用于热力图
    all_x = []
    all_z = []
    
    # 机器人轨迹
    robot_pos = trajectories['robot']['positions']
    robot_x = robot_pos[:, 0]
    robot_z = robot_pos[:, 2]
    all_x.extend(robot_x)
    all_z.extend(robot_z)
    
    # 行人轨迹
    for ped_pos in trajectories['pedestrians'].values():
        if len(ped_pos) > 0:
            all_x.extend(ped_pos[:, 0])
            all_z.extend(ped_pos[:, 2])
    
    # 绘制热力图
    if len(all_x) > 0:
        heatmap, xedges, yedges = np.histogram2d(all_x, all_z, bins=50)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        im = ax.imshow(heatmap.T, extent=extent, origin='lower', 
                      cmap='YlOrRd', alpha=0.5, aspect='auto')
        plt.colorbar(im, ax=ax, label='Density')
    
    # 绘制机器人轨迹
    ax.plot(robot_x, robot_z, 'b-', linewidth=2.5, label='Robot', alpha=0.8)
    ax.scatter(robot_x[0], robot_z[0], c='green', s=150, marker='o', 
              label='Start', zorder=5, edgecolors='black', linewidths=2)
    ax.scatter(robot_x[-1], robot_z[-1], c='red', s=150, marker='s', 
              label='Goal', zorder=5, edgecolors='black', linewidths=2)
    
    # 绘制行人轨迹
    colors = plt.cm.rainbow(np.linspace(0, 1, len(trajectories['pedestrians'])))
    for idx, (ped_id, ped_pos) in enumerate(trajectories['pedestrians'].items()):
        if len(ped_pos) > 0:
            ped_x = ped_pos[:, 0]
            ped_z = ped_pos[:, 2]
            ax.plot(ped_x, ped_z, '--', color=colors[idx], linewidth=1.5, 
                   label=f'Ped {ped_id}', alpha=0.7)
    
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Z Position (m)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"热力图已保存到: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_distance_over_time(trajectories: Dict,
                            output_path: pathlib.Path = None,
                            title: str = "Inter-Agent Distances Over Time",
                            figsize: Tuple[int, int] = (12, 6),
                            dpi: int = 300):
    """
    绘制机器人与行人之间的距离随时间变化
    
    Args:
        trajectories: 轨迹数据
        output_path: 输出文件路径
        title: 图表标题
        figsize: 图表大小
        dpi: 图像分辨率
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    robot_pos = trajectories['robot']['positions']
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(trajectories['pedestrians'])))
    
    for idx, (ped_id, ped_pos) in enumerate(trajectories['pedestrians'].items()):
        if len(ped_pos) == 0:
            continue
        
        # 计算每一步的距离
        min_len = min(len(robot_pos), len(ped_pos))
        distances = []
        
        for i in range(min_len):
            robot_p = robot_pos[i]
            ped_p = ped_pos[i]
            # 计算2D距离（忽略y轴高度）
            dist = np.sqrt((robot_p[0] - ped_p[0])**2 + (robot_p[2] - ped_p[2])**2)
            distances.append(dist)
        
        timesteps = np.arange(len(distances))
        ax.plot(timesteps, distances, color=colors[idx], linewidth=2, 
               label=f'Robot to Ped {ped_id}', alpha=0.8)
    
    # 添加安全距离线
    ax.axhline(y=0.5, color='r', linestyle='--', linewidth=1.5, 
              label='Safety Threshold (0.5m)', alpha=0.7)
    
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Distance (m)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"距离图已保存到: {output_path}")
    else:
        plt.show()
    
    plt.close()


def create_animation(trajectories: Dict,
                     output_path: pathlib.Path = None,
                     title: str = "Agent Trajectories Animation",
                     figsize: Tuple[int, int] = (10, 10),
                     fps: int = 10):
    """
    创建轨迹动画
    
    Args:
        trajectories: 轨迹数据
        output_path: 输出文件路径（.mp4或.gif）
        title: 动画标题
        figsize: 图表大小
        fps: 帧率
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    robot_pos = trajectories['robot']['positions']
    robot_x = robot_pos[:, 0]
    robot_z = robot_pos[:, 2]
    
    # 计算边界
    all_x = list(robot_x)
    all_z = list(robot_z)
    for ped_pos in trajectories['pedestrians'].values():
        if len(ped_pos) > 0:
            all_x.extend(ped_pos[:, 0])
            all_z.extend(ped_pos[:, 2])
    
    margin = 1.0
    x_min, x_max = min(all_x) - margin, max(all_x) + margin
    z_min, z_max = min(all_z) - margin, max(all_z) + margin
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(z_min, z_max)
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Z Position (m)')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # 初始化图形元素
    robot_trail, = ax.plot([], [], 'b-', linewidth=1.5, alpha=0.5, label='Robot Trail')
    robot_point = ax.scatter([], [], c='blue', s=100, marker='o', zorder=5)
    
    ped_trails = []
    ped_points = []
    colors = plt.cm.rainbow(np.linspace(0, 1, len(trajectories['pedestrians'])))
    
    for idx, ped_id in enumerate(trajectories['pedestrians'].keys()):
        trail, = ax.plot([], [], '--', color=colors[idx], linewidth=1, 
                        alpha=0.5, label=f'Ped {ped_id}')
        point = ax.scatter([], [], c=[colors[idx]], s=70, marker='o', zorder=4)
        ped_trails.append(trail)
        ped_points.append(point)
    
    ax.legend(loc='upper right')
    
    def init():
        robot_trail.set_data([], [])
        robot_point.set_offsets(np.empty((0, 2)))
        for trail in ped_trails:
            trail.set_data([], [])
        for point in ped_points:
            point.set_offsets(np.empty((0, 2)))
        return [robot_trail, robot_point] + ped_trails + ped_points
    
    def animate(frame):
        # 更新机器人
        robot_trail.set_data(robot_x[:frame+1], robot_z[:frame+1])
        robot_point.set_offsets([[robot_x[frame], robot_z[frame]]])
        
        # 更新行人
        for idx, (ped_id, ped_pos) in enumerate(trajectories['pedestrians'].items()):
            if frame < len(ped_pos):
                ped_x = ped_pos[:frame+1, 0]
                ped_z = ped_pos[:frame+1, 2]
                ped_trails[idx].set_data(ped_x, ped_z)
                ped_points[idx].set_offsets([[ped_pos[frame, 0], ped_pos[frame, 2]]])
        
        return [robot_trail, robot_point] + ped_trails + ped_points
    
    max_frames = len(robot_pos)
    anim = animation.FuncAnimation(fig, animate, init_func=init, 
                                  frames=max_frames, interval=1000//fps, 
                                  blit=True, repeat=True)
    
    if output_path:
        if str(output_path).endswith('.gif'):
            anim.save(output_path, writer='pillow', fps=fps)
        else:
            anim.save(output_path, writer='ffmpeg', fps=fps)
        print(f"动画已保存到: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="可视化agent轨迹")
    parser.add_argument("--episode_path", type=str, required=True,
                      help="Episode数据路径")
    parser.add_argument("--output_dir", type=str, default=None,
                      help="输出目录 (默认: episode_path/visualizations)")
    parser.add_argument("--plot_types", type=str, nargs='+', 
                      default=['trajectory', 'heatmap', 'distance'],
                      choices=['trajectory', 'heatmap', 'distance', 'animation'],
                      help="绘图类型")
    parser.add_argument("--dpi", type=int, default=300,
                      help="图像分辨率 (默认: 300)")
    parser.add_argument("--no_arrows", action='store_true',
                      help="不显示方向箭头")
    parser.add_argument("--fps", type=int, default=10,
                      help="动画帧率 (默认: 10)")
    
    args = parser.parse_args()
    
    episode_path = pathlib.Path(args.episode_path)
    
    if not episode_path.exists():
        print(f"错误: Episode路径不存在: {episode_path}")
        return
    
    # 加载轨迹数据
    print("正在加载轨迹数据...")
    trajectory_data = load_trajectory_data(episode_path)
    
    if not trajectory_data:
        print(f"错误: 未找到轨迹数据: {episode_path / 'trajectories' / '0.json'}")
        return
    
    print(f"已加载 {len(trajectory_data)} 个时间步的轨迹数据")
    
    # 提取轨迹
    trajectories = extract_trajectories(trajectory_data)
    
    print(f"机器人轨迹点数: {len(trajectories['robot']['positions'])}")
    print(f"行人数量: {len(trajectories['pedestrians'])}")
    for ped_id, ped_pos in trajectories['pedestrians'].items():
        print(f"  行人 {ped_id}: {len(ped_pos)} 个轨迹点")
    
    # 设置输出目录
    if args.output_dir:
        output_dir = pathlib.Path(args.output_dir)
    else:
        output_dir = episode_path / "visualizations"
    
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"\n输出目录: {output_dir}")
    
    episode_id = episode_path.name
    
    # 绘制不同类型的图
    if 'trajectory' in args.plot_types:
        print("\n绘制轨迹图...")
        plot_trajectories_2d(
            trajectories,
            output_path=output_dir / f"trajectory_{episode_id}.png",
            title=f"Agent Trajectories - Episode {episode_id}",
            show_arrows=not args.no_arrows,
            dpi=args.dpi
        )
    
    if 'heatmap' in args.plot_types:
        print("绘制热力图...")
        plot_trajectories_with_heatmap(
            trajectories,
            output_path=output_dir / f"heatmap_{episode_id}.png",
            title=f"Trajectory Density - Episode {episode_id}",
            dpi=args.dpi
        )
    
    if 'distance' in args.plot_types:
        print("绘制距离图...")
        plot_distance_over_time(
            trajectories,
            output_path=output_dir / f"distances_{episode_id}.png",
            title=f"Inter-Agent Distances - Episode {episode_id}",
            dpi=args.dpi
        )
    
    if 'animation' in args.plot_types:
        print("创建动画...")
        create_animation(
            trajectories,
            output_path=output_dir / f"animation_{episode_id}.mp4",
            title=f"Episode {episode_id}",
            fps=args.fps
        )
    
    print("\n✓ 完成!")


if __name__ == "__main__":
    main()

