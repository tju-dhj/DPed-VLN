#!/usr/bin/env python3
"""
轨迹数据使用示例

演示如何加载和使用采集的轨迹数据
"""

import json
import pathlib
import numpy as np
import matplotlib.pyplot as plt


def example_1_load_and_print():
    """示例1: 加载轨迹数据并打印基本信息"""
    print("="*80)
    print("示例1: 加载并查看轨迹数据")
    print("="*80)
    
    # 指定episode路径（请修改为实际路径）
    episode_path = pathlib.Path("data/collect_data/train/scene_001/episode_123")
    trajectory_file = episode_path / "trajectories" / "0.json"
    
    if not trajectory_file.exists():
        print(f"轨迹文件不存在: {trajectory_file}")
        print("请先运行数据采集，或修改路径为实际的episode路径")
        return
    
    # 加载轨迹数据
    with open(trajectory_file, "r") as f:
        trajectory_data = json.load(f)
    
    print(f"\n✓ 成功加载轨迹数据")
    print(f"总时间步数: {len(trajectory_data)}")
    
    # 查看第一步的数据结构
    if len(trajectory_data) > 0:
        first_step = trajectory_data[0]
        print(f"\n第一步数据结构:")
        print(f"  机器人位置: {first_step['robot']['position']}")
        print(f"  机器人旋转: {first_step['robot']['rotation']}")
        print(f"  行人数量: {len(first_step['pedestrians'])}")
        
        if len(first_step['pedestrians']) > 0:
            print(f"  第一个行人:")
            ped = first_step['pedestrians'][0]
            print(f"    ID: {ped['id']}")
            print(f"    位置: {ped['position']}")
            print(f"    旋转: {ped['rotation']}")
            print(f"    速度: {ped['velocity']}")
    
    # 统计行人数量变化
    pedestrian_counts = [len(step['pedestrians']) for step in trajectory_data]
    print(f"\n行人数量统计:")
    print(f"  最小: {min(pedestrian_counts)}")
    print(f"  最大: {max(pedestrian_counts)}")
    print(f"  平均: {np.mean(pedestrian_counts):.2f}")
    
    print("\n" + "="*80 + "\n")


def example_2_extract_robot_trajectory():
    """示例2: 提取并分析机器人轨迹"""
    print("="*80)
    print("示例2: 提取并分析机器人轨迹")
    print("="*80)
    
    episode_path = pathlib.Path("data/collect_data/train/scene_001/episode_123")
    trajectory_file = episode_path / "trajectories" / "0.json"
    
    if not trajectory_file.exists():
        print(f"轨迹文件不存在: {trajectory_file}")
        return
    
    with open(trajectory_file, "r") as f:
        trajectory_data = json.load(f)
    
    # 提取机器人位置
    robot_positions = np.array([step['robot']['position'] for step in trajectory_data])
    
    print(f"\n机器人轨迹分析:")
    print(f"  轨迹点数: {len(robot_positions)}")
    print(f"  起始位置: ({robot_positions[0][0]:.2f}, {robot_positions[0][1]:.2f}, {robot_positions[0][2]:.2f})")
    print(f"  结束位置: ({robot_positions[-1][0]:.2f}, {robot_positions[-1][1]:.2f}, {robot_positions[-1][2]:.2f})")
    
    # 计算轨迹长度
    total_length = 0.0
    for i in range(1, len(robot_positions)):
        dx = robot_positions[i][0] - robot_positions[i-1][0]
        dy = robot_positions[i][1] - robot_positions[i-1][1]
        dz = robot_positions[i][2] - robot_positions[i-1][2]
        total_length += np.sqrt(dx**2 + dy**2 + dz**2)
    
    print(f"  轨迹总长度: {total_length:.2f} 米")
    
    # 计算速度
    velocities = []
    for i in range(1, len(robot_positions)):
        dx = robot_positions[i][0] - robot_positions[i-1][0]
        dz = robot_positions[i][2] - robot_positions[i-1][2]
        speed = np.sqrt(dx**2 + dz**2)
        velocities.append(speed)
    
    if velocities:
        print(f"  平均速度: {np.mean(velocities):.3f} m/step")
        print(f"  最大速度: {np.max(velocities):.3f} m/step")
    
    print("\n" + "="*80 + "\n")


def example_3_extract_pedestrian_trajectories():
    """示例3: 提取所有行人轨迹"""
    print("="*80)
    print("示例3: 提取所有行人轨迹")
    print("="*80)
    
    episode_path = pathlib.Path("data/collect_data/train/scene_001/episode_123")
    trajectory_file = episode_path / "trajectories" / "0.json"
    
    if not trajectory_file.exists():
        print(f"轨迹文件不存在: {trajectory_file}")
        return
    
    with open(trajectory_file, "r") as f:
        trajectory_data = json.load(f)
    
    # 组织行人轨迹
    pedestrian_trajectories = {}
    
    for step in trajectory_data:
        for ped in step['pedestrians']:
            ped_id = ped['id']
            if ped_id not in pedestrian_trajectories:
                pedestrian_trajectories[ped_id] = {
                    'positions': [],
                    'rotations': [],
                    'velocities': []
                }
            pedestrian_trajectories[ped_id]['positions'].append(ped['position'])
            pedestrian_trajectories[ped_id]['rotations'].append(ped['rotation'])
            pedestrian_trajectories[ped_id]['velocities'].append(ped['velocity'])
    
    print(f"\n检测到 {len(pedestrian_trajectories)} 个行人")
    
    for ped_id, data in pedestrian_trajectories.items():
        positions = np.array(data['positions'])
        velocities = np.array(data['velocities'])
        
        print(f"\n行人 {ped_id}:")
        print(f"  轨迹点数: {len(positions)}")
        print(f"  起始位置: ({positions[0][0]:.2f}, {positions[0][1]:.2f}, {positions[0][2]:.2f})")
        print(f"  结束位置: ({positions[-1][0]:.2f}, {positions[-1][1]:.2f}, {positions[-1][2]:.2f})")
        
        # 计算轨迹长度
        length = 0.0
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dz = positions[i][2] - positions[i-1][2]
            length += np.sqrt(dx**2 + dz**2)
        
        print(f"  轨迹长度: {length:.2f} 米")
        
        # 计算平均速度
        speeds = [np.linalg.norm(v) for v in velocities]
        print(f"  平均速度: {np.mean(speeds):.3f} m/s")
    
    print("\n" + "="*80 + "\n")


def example_4_calculate_min_distances():
    """示例4: 计算机器人与行人之间的最小距离"""
    print("="*80)
    print("示例4: 计算机器人与行人之间的最小距离")
    print("="*80)
    
    episode_path = pathlib.Path("data/collect_data/train/scene_001/episode_123")
    trajectory_file = episode_path / "trajectories" / "0.json"
    
    if not trajectory_file.exists():
        print(f"轨迹文件不存在: {trajectory_file}")
        return
    
    with open(trajectory_file, "r") as f:
        trajectory_data = json.load(f)
    
    # 提取机器人位置
    robot_positions = np.array([step['robot']['position'] for step in trajectory_data])
    
    # 计算与每个行人的最小距离
    pedestrian_min_distances = {}
    
    for step_idx, step in enumerate(trajectory_data):
        robot_pos = robot_positions[step_idx]
        
        for ped in step['pedestrians']:
            ped_id = ped['id']
            ped_pos = np.array(ped['position'])
            
            # 计算2D距离（忽略高度）
            dx = robot_pos[0] - ped_pos[0]
            dz = robot_pos[2] - ped_pos[2]
            distance = np.sqrt(dx**2 + dz**2)
            
            if ped_id not in pedestrian_min_distances:
                pedestrian_min_distances[ped_id] = {
                    'min_distance': distance,
                    'min_step': step_idx,
                    'all_distances': []
                }
            else:
                if distance < pedestrian_min_distances[ped_id]['min_distance']:
                    pedestrian_min_distances[ped_id]['min_distance'] = distance
                    pedestrian_min_distances[ped_id]['min_step'] = step_idx
            
            pedestrian_min_distances[ped_id]['all_distances'].append(distance)
    
    print(f"\n机器人与行人的最近接近距离:")
    
    for ped_id, data in pedestrian_min_distances.items():
        min_dist = data['min_distance']
        min_step = data['min_step']
        avg_dist = np.mean(data['all_distances'])
        
        print(f"\n行人 {ped_id}:")
        print(f"  最小距离: {min_dist:.2f} 米 (在第 {min_step} 步)")
        print(f"  平均距离: {avg_dist:.2f} 米")
        
        # 安全性评估
        if min_dist < 0.3:
            print(f"  ⚠️  警告: 距离过近，可能存在碰撞风险")
        elif min_dist < 0.5:
            print(f"  ⚠️  注意: 距离较近，接近安全阈值")
        else:
            print(f"  ✓ 安全距离充足")
    
    print("\n" + "="*80 + "\n")


def example_5_simple_visualization():
    """示例5: 简单的轨迹可视化"""
    print("="*80)
    print("示例5: 简单的轨迹可视化")
    print("="*80)
    
    episode_path = pathlib.Path("data/collect_data/train/scene_001/episode_123")
    trajectory_file = episode_path / "trajectories" / "0.json"
    
    if not trajectory_file.exists():
        print(f"轨迹文件不存在: {trajectory_file}")
        return
    
    with open(trajectory_file, "r") as f:
        trajectory_data = json.load(f)
    
    # 提取机器人轨迹
    robot_positions = np.array([step['robot']['position'] for step in trajectory_data])
    robot_x = robot_positions[:, 0]
    robot_z = robot_positions[:, 2]
    
    # 提取行人轨迹
    pedestrian_trajectories = {}
    for step in trajectory_data:
        for ped in step['pedestrians']:
            ped_id = ped['id']
            if ped_id not in pedestrian_trajectories:
                pedestrian_trajectories[ped_id] = []
            pedestrian_trajectories[ped_id].append(ped['position'])
    
    # 创建图表
    plt.figure(figsize=(10, 10))
    
    # 绘制机器人轨迹
    plt.plot(robot_x, robot_z, 'b-', linewidth=2, label='机器人', alpha=0.7)
    plt.scatter(robot_x[0], robot_z[0], c='green', s=150, marker='o', 
               label='起点', zorder=5, edgecolors='black', linewidths=2)
    plt.scatter(robot_x[-1], robot_z[-1], c='red', s=150, marker='s', 
               label='终点', zorder=5, edgecolors='black', linewidths=2)
    
    # 绘制行人轨迹
    colors = plt.cm.rainbow(np.linspace(0, 1, len(pedestrian_trajectories)))
    for idx, (ped_id, positions) in enumerate(pedestrian_trajectories.items()):
        positions = np.array(positions)
        ped_x = positions[:, 0]
        ped_z = positions[:, 2]
        plt.plot(ped_x, ped_z, '--', color=colors[idx], linewidth=1.5, 
                label=f'行人 {ped_id}', alpha=0.7)
    
    plt.xlabel('X 位置 (米)', fontsize=12)
    plt.ylabel('Z 位置 (米)', fontsize=12)
    plt.title('Agent 轨迹可视化', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.axis('equal')
    
    output_file = "simple_trajectory_example.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ 轨迹图已保存到: {output_file}")
    print("\n" + "="*80 + "\n")


def main():
    """运行所有示例"""
    print("\n" + "="*80)
    print("轨迹数据使用示例集")
    print("="*80)
    print("\n说明: 这些示例演示如何加载和使用采集的轨迹数据")
    print("请确保已经运行数据采集，并修改episode_path为实际路径\n")
    
    # 运行示例（如果文件不存在，会自动跳过）
    example_1_load_and_print()
    example_2_extract_robot_trajectory()
    example_3_extract_pedestrian_trajectories()
    example_4_calculate_min_distances()
    example_5_simple_visualization()
    
    print("="*80)
    print("所有示例运行完成！")
    print("="*80)
    print("\n提示:")
    print("1. 修改示例中的 episode_path 为您的实际数据路径")
    print("2. 使用这些示例作为起点，开发自己的分析代码")
    print("3. 使用 visualize_trajectories.py 生成更专业的可视化")
    print("\n")


if __name__ == "__main__":
    main()

