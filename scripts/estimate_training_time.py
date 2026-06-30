#!/usr/bin/env python3
"""
估算训练时间
根据配置参数计算预期的训练时间
"""

import argparse
from typing import Dict


def estimate_training_time(
    total_num_steps: int,
    num_environments: int,
    num_steps: int,
    ppo_epoch: int,
    num_mini_batch: int,
    steps_per_second: float = None
) -> Dict[str, float]:
    """
    估算训练时间
    
    Args:
        total_num_steps: 总训练步数
        num_environments: 环境数量
        num_steps: 每次rollout的步数
        ppo_epoch: PPO更新轮数
        num_mini_batch: mini batch数量
        steps_per_second: 每秒步数（如果提供，用于更精确的估算）
    
    Returns:
        包含各种时间估算的字典
    """
    # 计算每个update的步数
    steps_per_update = num_environments * num_steps
    
    # 计算总update数
    total_updates = total_num_steps // steps_per_update
    
    # 计算每个update的样本数
    samples_per_update = num_environments * num_steps
    
    # 计算每个update的PPO更新次数
    ppo_updates_per_rollout = ppo_epoch * num_mini_batch
    
    print("=" * 70)
    print("训练时间估算")
    print("=" * 70)
    print(f"\n配置参数:")
    print(f"  总训练步数 (total_num_steps): {total_num_steps:,}")
    print(f"  环境数量 (num_environments): {num_environments}")
    print(f"  每次rollout步数 (num_steps): {num_steps}")
    print(f"  PPO更新轮数 (ppo_epoch): {ppo_epoch}")
    print(f"  Mini batch数量 (num_mini_batch): {num_mini_batch}")
    
    print(f"\n计算:")
    print(f"  每个update的步数: {steps_per_update:,} (num_environments × num_steps)")
    print(f"  总update数: {total_updates:,} (total_num_steps ÷ steps_per_update)")
    print(f"  每个update的样本数: {samples_per_update:,}")
    print(f"  每个rollout的PPO更新次数: {ppo_updates_per_rollout}")
    
    # 如果没有提供steps_per_second，使用经验值估算
    if steps_per_second is None:
        # 基于经验值：单环境大约0.5-2 FPS（取决于场景复杂度）
        # 这里使用保守估计：1 FPS（每秒1步）
        estimated_fps = 1.0 * num_environments  # 假设每个环境1 FPS
        print(f"\n⚠️  未提供实际FPS，使用经验估算值: {estimated_fps:.2f} FPS")
    else:
        estimated_fps = steps_per_second
        print(f"\n使用提供的FPS: {estimated_fps:.2f} FPS")
    
    # 计算时间
    # 每个rollout的时间 = num_steps / fps_per_env
    fps_per_env = estimated_fps / num_environments if num_environments > 0 else estimated_fps
    time_per_rollout = num_steps / fps_per_env  # 秒
    
    # 每个update的时间 = rollout时间 + PPO更新时间
    # PPO更新时间通常比rollout时间短得多，这里粗略估算为rollout时间的0.1-0.3倍
    ppo_time_factor = 0.2  # PPO更新时间约为rollout时间的20%
    time_per_update = time_per_rollout * (1 + ppo_time_factor * ppo_updates_per_rollout)
    
    # 总时间
    total_time_seconds = total_updates * time_per_update
    total_time_hours = total_time_seconds / 3600
    total_time_days = total_time_hours / 24
    
    print(f"\n时间估算:")
    print(f"  每个环境FPS: {fps_per_env:.2f}")
    print(f"  每个rollout时间: {time_per_rollout:.2f} 秒 ({time_per_rollout/60:.2f} 分钟)")
    print(f"  每个update时间: {time_per_update:.2f} 秒 ({time_per_update/60:.2f} 分钟)")
    print(f"  总训练时间: {total_time_seconds/3600:.2f} 小时 ({total_time_days:.2f} 天)")
    
    # 提供不同FPS下的估算
    print(f"\n不同FPS下的时间估算:")
    for fps in [0.5, 1.0, 1.5, 2.0]:
        fps_per_env = fps / num_environments if num_environments > 0 else fps
        time_per_rollout = num_steps / fps_per_env
        time_per_update = time_per_rollout * (1 + ppo_time_factor * ppo_updates_per_rollout)
        total_time = total_updates * time_per_update
        print(f"  {fps:.1f} FPS: {total_time/3600:.2f} 小时 ({total_time/3600/24:.2f} 天)")
    
    print("=" * 70)
    
    return {
        'total_updates': total_updates,
        'total_time_hours': total_time_hours,
        'total_time_days': total_time_days,
        'time_per_update': time_per_update,
        'estimated_fps': estimated_fps
    }


def compare_configs():
    """比较原始配置和修改后配置的训练时间"""
    print("\n" + "=" * 70)
    print("配置对比")
    print("=" * 70)
    
    # 原始配置
    print("\n【原始配置】")
    original = estimate_training_time(
        total_num_steps=10000000,
        num_environments=1,
        num_steps=128,
        ppo_epoch=2,
        num_mini_batch=1,
        steps_per_second=None
    )
    
    # 修改后配置
    print("\n【修改后配置（加速版）】")
    modified = estimate_training_time(
        total_num_steps=2000000,
        num_environments=1,
        num_steps=64,
        ppo_epoch=1,
        num_mini_batch=1,
        steps_per_second=None
    )
    
    # 计算加速比
    speedup = original['total_time_hours'] / modified['total_time_hours']
    print(f"\n加速比: {speedup:.2f}x")
    print(f"节省时间: {original['total_time_hours'] - modified['total_time_hours']:.2f} 小时")
    print(f"  ({original['total_time_days'] - modified['total_time_days']:.2f} 天)")


def main():
    parser = argparse.ArgumentParser(description='估算训练时间')
    parser.add_argument('--total_num_steps', type=int, default=2000000,
                       help='总训练步数')
    parser.add_argument('--num_environments', type=int, default=1,
                       help='环境数量')
    parser.add_argument('--num_steps', type=int, default=64,
                       help='每次rollout的步数')
    parser.add_argument('--ppo_epoch', type=int, default=1,
                       help='PPO更新轮数')
    parser.add_argument('--num_mini_batch', type=int, default=1,
                       help='Mini batch数量')
    parser.add_argument('--steps_per_second', type=float, default=None,
                       help='实际观察到的每秒步数（用于更精确的估算）')
    parser.add_argument('--compare', action='store_true',
                       help='比较原始配置和修改后配置')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_configs()
    else:
        estimate_training_time(
            total_num_steps=args.total_num_steps,
            num_environments=args.num_environments,
            num_steps=args.num_steps,
            ppo_epoch=args.ppo_epoch,
            num_mini_batch=args.num_mini_batch,
            steps_per_second=args.steps_per_second
        )


if __name__ == "__main__":
    main()
















