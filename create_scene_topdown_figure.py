#!/usr/bin/env python3
"""
使用Habitat-sim渲染场景俯视图并叠加轨迹

特点:
- 加载真实的3D场景GLB文件
- 渲染高质量俯视图
- 叠加agent轨迹
- 学术期刊标准
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib import rc
from PIL import Image
import pathlib
import habitat_sim
import magnum as mn

# 学术期刊风格
plt.style.use('seaborn-v0_8-paper')
rc('font', **{'family': 'serif', 'serif': ['Times New Roman'], 'size': 11})
rc('text', usetex=False)
rc('axes', linewidth=1.5)

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
        
        robot_pos = step_data['robot']['position']
        robot_positions.append(robot_pos)
        
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

def render_scene_topdown(scene_file, trajectories, resolution=2048, render_agents=True, frame_index=0):
    """
    渲染场景的俯视图，可选择是否渲染agents
    
    Args:
        scene_file: GLB场景文件路径
        trajectories: 轨迹数据
        resolution: 渲染分辨率
        render_agents: 是否在场景中渲染agents（使用3D标记）
        frame_index: 要渲染哪一帧的agent位置（0=起点，-1=终点）
    
    Returns:
        渲染的俯视图图像和元数据
    """
    # 配置simulator
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = str(scene_file)
    # 启用物理引擎以支持articulated objects
    backend_cfg.enable_physics = True if render_agents else False
    backend_cfg.physics_config_file = "data/default.physics_config.json" if render_agents else ""
    
    # Agent配置
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    
    # 计算场景边界和合适的相机位置
    robot_pos = trajectories['robot']
    all_x = [robot_pos[:, 0]]
    all_z = [robot_pos[:, 2]]
    all_y = [robot_pos[:, 1]]
    
    for ped_pos in trajectories['pedestrians'].values():
        if len(ped_pos) > 0:
            all_x.append(ped_pos[:, 0])
            all_z.append(ped_pos[:, 2])
            all_y.append(ped_pos[:, 1])
    
    all_x = np.concatenate(all_x)
    all_z = np.concatenate(all_z)
    all_y = np.concatenate(all_y)
    
    # 计算轨迹中心点
    center_x = (all_x.min() + all_x.max()) / 2
    center_z = (all_z.min() + all_z.max()) / 2
    
    # 使用固定的相机高度和FOV，按照实际比例渲染场景
    y_camera = all_y.max() + 8.0  # 固定相机高度8米
    desired_fov = 90.0  # 固定90度FOV，标准广角视野
    
    # Sensor配置 - 俯视图
    sensor_spec = habitat_sim.CameraSensorSpec()
    sensor_spec.uuid = "topdown"
    sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    sensor_spec.resolution = [resolution, resolution]
    sensor_spec.position = mn.Vector3(0, 0, 0)
    sensor_spec.orientation = mn.Vector3(0, 0, 0)
    sensor_spec.hfov = desired_fov  # 设置水平FOV
    
    agent_cfg.sensor_specifications = [sensor_spec]
    
    # 创建simulator
    cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)
    
    # 设置相机位置和朝向（俯视）
    agent = sim.get_agent(0)
    agent_state = habitat_sim.AgentState()
    agent_state.position = np.array([center_x, y_camera, center_z], dtype=np.float32)
    
    # 相机向下看 (pitch = -90度)
    rotation_quat = mn.Quaternion.rotation(
        mn.Rad(-np.pi / 2), mn.Vector3(1, 0, 0)
    )
    agent_state.rotation = np.quaternion(
        rotation_quat.scalar,
        rotation_quat.vector[0],
        rotation_quat.vector[1],
        rotation_quat.vector[2]
    )
    
    agent.set_state(agent_state)
    
    # 如果需要渲染agents，在场景中添加articulated agents
    agent_positions = None
    agent_obj_ids = []
    
    if render_agents:
        # 获取指定帧的agent位置
        if frame_index < 0:
            frame_index = len(robot_pos) + frame_index
        frame_index = min(max(0, frame_index), len(robot_pos) - 1)
        
        # 定义模型路径
        robot_urdf = "data/robots/hab_spot_arm/urdf/hab_spot_arm.urdf"
        humanoid_urdf = "data/humanoids/humanoid_data/female_0/female_0.urdf"
        
        try:
            # 获取articulated object manager
            art_obj_mgr = sim.get_articulated_object_manager()
            
            # 加载机器人
            robot_pos_frame = robot_pos[frame_index]
            robot_obj = art_obj_mgr.add_articulated_object_from_urdf(
                robot_urdf,
                fixed_base=False
            )
            if robot_obj is not None:
                robot_obj.translation = mn.Vector3(
                    robot_pos_frame[0], 
                    robot_pos_frame[1], 
                    robot_pos_frame[2]
                )
                agent_obj_ids.append(robot_obj.object_id)
                print(f"  ✓ 已加载机器人模型")
            
            # 加载行人
            ped_count = 0
            for ped_id, ped_traj in trajectories['pedestrians'].items():
                if frame_index < len(ped_traj):
                    ped_pos_frame = ped_traj[frame_index]
                    ped_obj = art_obj_mgr.add_articulated_object_from_urdf(
                        humanoid_urdf,
                        fixed_base=False
                    )
                    if ped_obj is not None:
                        ped_obj.translation = mn.Vector3(
                            ped_pos_frame[0],
                            ped_pos_frame[1],
                            ped_pos_frame[2]
                        )
                        agent_obj_ids.append(ped_obj.object_id)
                        ped_count += 1
            
            print(f"  ✓ 已加载 {ped_count} 个行人模型")
                
        except Exception as e:
            print(f"  ⚠ 无法加载agent模型: {e}")
            print(f"  将使用标记代替...")
        
        # 记录位置用于标记（作为备份方案）
        agent_positions = {
            'robot': robot_pos[frame_index],
            'pedestrians': []
        }
        
        for ped_id, ped_traj in trajectories['pedestrians'].items():
            if frame_index < len(ped_traj):
                agent_positions['pedestrians'].append({
                    'id': ped_id,
                    'pos': ped_traj[frame_index]
                })
        
        print(f"  记录了 {len(agent_positions['pedestrians']) + 1} 个agent的位置")
    
    # 渲染场景（包含agents如果已加载）
    observations = sim.get_sensor_observations()
    topdown_image = observations["topdown"]
    
    # 清理articulated objects
    if len(agent_obj_ids) > 0:
        art_obj_mgr = sim.get_articulated_object_manager()
        for obj_id in agent_obj_ids:
            art_obj_mgr.remove_object_by_id(obj_id)
    
    sim.close()
    
    # 计算实际的像素到世界坐标的映射关系
    # 相机的视野范围（在地面上）
    fov_rad = np.deg2rad(desired_fov)
    ground_width = 2 * y_camera * np.tan(fov_rad / 2)
    
    # 计算图像extent，使其与轨迹坐标精确对应
    actual_extent = [
        center_x - ground_width / 2,  # x_min
        center_x + ground_width / 2,  # x_max
        center_z - ground_width / 2,  # z_min (bottom)
        center_z + ground_width / 2   # z_max (top)
    ]
    
    return {
        'image': topdown_image,
        'bounds': {
            'x_min': actual_extent[0],
            'x_max': actual_extent[1],
            'z_min': actual_extent[2],
            'z_max': actual_extent[3],
            'extent': actual_extent
        },
        'camera_height': y_camera,
        'ground_width': ground_width,
        'agent_positions': agent_positions  # 如果render_agents=True，包含agent位置
    }

def create_topdown_figure(trajectories, scene_file, output_path, dpi=600, debug=False, render_agents=True, frame_index=0):
    """
    创建带场景俯视图的学术期刊级别图表
    
    Args:
        render_agents: 是否在场景中渲染agents的3D标记
        frame_index: 要渲染哪一帧的agent位置（0=起点，-1=终点，或指定帧索引）
    """
    print("正在渲染场景俯视图...")
    scene_data = render_scene_topdown(scene_file, trajectories, resolution=2048, 
                                     render_agents=render_agents, frame_index=frame_index)
    print("✓ 场景渲染完成")
    
    if debug:
        print(f"\n调试信息:")
        print(f"  相机高度: {scene_data['camera_height']:.2f} m")
        print(f"  地面视野宽度: {scene_data['ground_width']:.2f} m")
        print(f"  渲染范围 X: [{scene_data['bounds']['x_min']:.2f}, {scene_data['bounds']['x_max']:.2f}]")
        print(f"  渲染范围 Z: [{scene_data['bounds']['z_min']:.2f}, {scene_data['bounds']['z_max']:.2f}]")
        
        robot_pos = trajectories['robot']
        print(f"\n  轨迹范围 X: [{robot_pos[:, 0].min():.2f}, {robot_pos[:, 0].max():.2f}]")
        print(f"  轨迹范围 Z: [{robot_pos[:, 2].min():.2f}, {robot_pos[:, 2].max():.2f}]")
        print(f"  轨迹范围 Y: [{robot_pos[:, 1].min():.2f}, {robot_pos[:, 1].max():.2f}]\n")
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(8, 7.5), dpi=dpi)
    
    # 显示场景俯视图
    # matplotlib的imshow默认原点在左上角，需要设置origin='lower'使其与轨迹坐标系一致
    ax.imshow(scene_data['image'],
             extent=scene_data['bounds']['extent'],
             alpha=0.5,  # 半透明
             aspect='equal',  # 保持纵横比
             origin='lower',  # 原点在左下角，与matplotlib坐标系一致
             zorder=0,
             interpolation='bilinear')
    
    bounds = scene_data['bounds']
    
    # 配色方案
    robot_color = '#0066CC'
    robot_start_color = '#00AA00'
    robot_goal_color = '#CC0000'
    pedestrian_colors = ['#FF8C00', '#8B008B', '#8B4513', '#DC143C', '#2F4F4F']
    
    # 提取机器人轨迹
    robot_pos = trajectories['robot']
    robot_x = robot_pos[:, 0]
    robot_z = robot_pos[:, 2]
    
    # 绘制机器人轨迹（带白色边框）
    ax.plot(robot_x, robot_z,
            color='white',
            linewidth=5.5,
            alpha=0.9,
            zorder=2,
            solid_capstyle='round')
    
    ax.plot(robot_x, robot_z,
            color=robot_color,
            linewidth=4,
            label='Robot Trajectory',
            alpha=1.0,
            zorder=3,
            solid_capstyle='round')
    
    # 起点和终点
    ax.scatter(robot_x[0], robot_z[0],
              s=350,
              color=robot_start_color,
              marker='o',
              edgecolors='white',
              linewidths=3.5,
              label='Start',
              zorder=6)
    
    ax.scatter(robot_x[-1], robot_z[-1],
              s=350,
              color=robot_goal_color,
              marker='s',
              edgecolors='white',
              linewidths=3.5,
              label='Goal',
              zorder=6)
    
    # 方向箭头
    arrow_interval = max(len(robot_pos) // 6, 1)
    for i in range(0, len(robot_pos) - 1, arrow_interval):
        if i + 1 < len(robot_pos):
            dx = robot_x[i + 1] - robot_x[i]
            dz = robot_z[i + 1] - robot_z[i]
            
            if np.sqrt(dx**2 + dz**2) > 0.15:
                # 白色边框
                arrow_bg = FancyArrowPatch(
                    (robot_x[i], robot_z[i]),
                    (robot_x[i] + dx * 0.7, robot_z[i] + dz * 0.7),
                    arrowstyle='->,head_width=0.6,head_length=0.8',
                    color='white',
                    linewidth=4,
                    alpha=0.9,
                    zorder=4
                )
                ax.add_patch(arrow_bg)
                
                # 箭头
                arrow = FancyArrowPatch(
                    (robot_x[i], robot_z[i]),
                    (robot_x[i] + dx * 0.7, robot_z[i] + dz * 0.7),
                    arrowstyle='->,head_width=0.6,head_length=0.8',
                    color=robot_color,
                    linewidth=2.8,
                    alpha=1.0,
                    zorder=5
                )
                ax.add_patch(arrow)
    
    # 绘制行人轨迹
    for idx, (ped_id, ped_pos) in enumerate(trajectories['pedestrians'].items()):
        if len(ped_pos) == 0:
            continue
        
        color = pedestrian_colors[idx % len(pedestrian_colors)]
        ped_x = ped_pos[:, 0]
        ped_z = ped_pos[:, 2]
        
        # 白色边框
        ax.plot(ped_x, ped_z,
               linestyle='--',
               color='white',
               linewidth=4.5,
               alpha=0.9,
               zorder=2,
               dashes=(6, 3))
        
        # 行人轨迹
        ax.plot(ped_x, ped_z,
               linestyle='--',
               color=color,
               linewidth=3,
               label=f'Pedestrian {ped_id + 1}',
               alpha=1.0,
               zorder=3,
               dashes=(6, 3))
        
        # 起点和终点
        ax.scatter(ped_x[0], ped_z[0],
                  s=180,
                  color=color,
                  marker='o',
                  edgecolors='white',
                  linewidths=3,
                  alpha=1.0,
                  zorder=5)
        
        ax.scatter(ped_x[-1], ped_z[-1],
                  s=180,
                  color=color,
                  marker='x',
                  linewidths=4,
                  zorder=5)
    
    # 设置坐标轴
    ax.set_xlabel('X Position (m)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Z Position (m)', fontsize=13, fontweight='bold')
    ax.set_title('Multi-Agent Navigation in Real 3D Environment',
                fontsize=15,
                fontweight='bold',
                pad=20)
    
    ax.set_xlim(bounds['x_min'], bounds['x_max'])
    ax.set_ylim(bounds['z_min'], bounds['z_max'])
    
    # 网格
    ax.grid(True, linestyle=':', alpha=0.4, linewidth=0.8, color='gray', zorder=1)
    
    # 相等比例
    ax.set_aspect('equal', adjustable='box')
    
    # 图例
    legend = ax.legend(
        loc='upper right',
        fontsize=11,
        frameon=True,
        fancybox=True,
        shadow=True,
        framealpha=0.95,
        edgecolor='black',
        facecolor='white'
    )
    legend.get_frame().set_linewidth(1.5)
    
    # 比例尺
    add_scale_bar(ax, bounds)
    
    # 如果有agent位置信息，在图上绘制agent标记
    if scene_data['agent_positions'] is not None:
        agent_pos_data = scene_data['agent_positions']
        
        # 绘制机器人位置（大一点的标记）
        robot_pos_frame = agent_pos_data['robot']
        ax.scatter(robot_pos_frame[0], robot_pos_frame[2],
                  s=600,
                  color=robot_color,
                  marker='D',  # 菱形
                  edgecolors='white',
                  linewidths=4,
                  alpha=0.9,
                  zorder=8,
                  label='Robot Position')
        
        # 绘制行人位置
        for ped_data in agent_pos_data['pedestrians']:
            ped_pos_frame = ped_data['pos']
            ped_id = ped_data['id']
            color = pedestrian_colors[ped_id % len(pedestrian_colors)]
            
            ax.scatter(ped_pos_frame[0], ped_pos_frame[2],
                      s=400,
                      color=color,
                      marker='o',
                      edgecolors='white',
                      linewidths=3,
                      alpha=0.9,
                      zorder=8)
        
        print(f"✓ 已在图中标记 {len(agent_pos_data['pedestrians']) + 1} 个agent的当前位置")
    
    # 统计信息
    num_pedestrians = len(trajectories['pedestrians'])
    trajectory_length = len(robot_pos)
    path_length = calculate_path_length(robot_pos)
    
    scene_name = pathlib.Path(scene_file).parent.name.split('-')[1]
    
    textstr = f'Scene: {scene_name}\n'
    textstr += f'Time steps: {trajectory_length}\n'
    textstr += f'Pedestrians: {num_pedestrians}\n'
    textstr += f'Path length: {path_length:.1f} m'
    
    props = dict(boxstyle='round,pad=0.6',
                facecolor='white',
                alpha=0.95,
                edgecolor='black',
                linewidth=1.5)
    ax.text(0.02, 0.02, textstr,
           transform=ax.transAxes,
           fontsize=10,
           verticalalignment='bottom',
           bbox=props,
           zorder=10)
    
    ax.tick_params(axis='both', which='major', labelsize=11, width=1.5, length=7)
    
    plt.tight_layout()
    
    # 保存
    base_path = pathlib.Path(output_path)
    
    plt.savefig(base_path.with_suffix('.png'),
               dpi=dpi,
               bbox_inches='tight',
               facecolor='white')
    print(f"✓ 已保存PNG: {base_path.with_suffix('.png')}")
    
    plt.savefig(base_path.with_suffix('.pdf'),
               format='pdf',
               bbox_inches='tight',
               facecolor='white')
    print(f"✓ 已保存PDF: {base_path.with_suffix('.pdf')}")
    
    try:
        plt.savefig(base_path.with_suffix('.eps'),
                   format='eps',
                   bbox_inches='tight',
                   facecolor='white')
        print(f"✓ 已保存EPS: {base_path.with_suffix('.eps')}")
    except:
        pass
    
    plt.close()

def add_scale_bar(ax, bounds, length_m=2.0):
    """添加比例尺"""
    x_start = bounds['x_min'] + (bounds['x_max'] - bounds['x_min']) * 0.05
    y_pos = bounds['z_min'] + (bounds['z_max'] - bounds['z_min']) * 0.08
    
    ax.plot([x_start, x_start + length_m], [y_pos, y_pos],
           color='black', linewidth=4.5, solid_capstyle='butt', zorder=10)
    
    ax.text(x_start + length_m / 2, y_pos + 0.2,
           f'{length_m:.0f} m',
           ha='center', va='bottom',
           fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.95, edgecolor='none'),
           zorder=10)

def calculate_path_length(positions):
    """计算路径总长度"""
    if len(positions) < 2:
        return 0.0
    diffs = np.diff(positions, axis=0)
    distances = np.sqrt(np.sum(diffs**2, axis=1))
    return np.sum(distances)

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='使用真实3D场景渲染俯视图并叠加轨迹'
    )
    
    parser.add_argument('--trajectory_file', type=str, required=True,
                       help='轨迹数据文件路径')
    parser.add_argument('--scene_file', type=str, required=True,
                       help='场景GLB文件路径 (例如: .../1EiJpeRNEs1.basis.glb)')
    parser.add_argument('--output', type=str, default='scene_topdown_figure',
                       help='输出文件路径（不含扩展名）')
    parser.add_argument('--dpi', type=int, default=600,
                       help='图像分辨率 (默认: 600)')
    parser.add_argument('--debug', action='store_true',
                       help='显示调试信息（比例尺、坐标范围等）')
    parser.add_argument('--render_agents', action='store_true', default=True,
                       help='在场景中渲染agents的3D标记（默认：True）')
    parser.add_argument('--no_render_agents', dest='render_agents', action='store_false',
                       help='不渲染agents标记')
    parser.add_argument('--frame_index', type=int, default=0,
                       help='渲染哪一帧的agent位置（0=起点，-1=终点，默认：0）')
    
    args = parser.parse_args()
    
    # 加载轨迹数据
    print(f"正在加载轨迹数据: {args.trajectory_file}")
    trajectory_data = load_trajectory_data(args.trajectory_file)
    print(f"✓ 轨迹数据包含 {len(trajectory_data)} 个时间步")
    
    # 提取轨迹
    print("提取轨迹...")
    trajectories = extract_trajectories(trajectory_data)
    print(f"✓ 机器人轨迹点数: {len(trajectories['robot'])}")
    print(f"✓ 行人数量: {len(trajectories['pedestrians'])}")
    
    # 生成场景俯视图
    print(f"\n正在加载场景: {args.scene_file}")
    create_topdown_figure(
        trajectories,
        args.scene_file,
        args.output,
        dpi=args.dpi,
        debug=args.debug,
        render_agents=args.render_agents,
        frame_index=args.frame_index
    )
    
    print("\n✓ 完成！")
    print("生成的图表展示了真实3D场景中的agent轨迹")

if __name__ == "__main__":
    main()

