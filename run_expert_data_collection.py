#!/usr/bin/env python3

"""
专家数据采集运行脚本

这个脚本用于启动专家数据采集任务，收集专家轨迹数据用于训练。
"""

import argparse
import os
import sys
from pathlib import Path

# 添加项目路径到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from habitat_baselines.run import execute_exp


def main():
    parser = argparse.ArgumentParser(description="专家数据采集")
    parser.add_argument(
        "--config-name",
        type=str,
        default="social_nav_v2/expert_data_collection_v2",
        help="配置文件名称",
    )
    parser.add_argument(
        "--data-folder",
        type=str,
        default="expert_data",
        help="数据保存目录",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="数据集分割",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=1000,
        help="最大采集episode数",
    )
    parser.add_argument(
        "--max-steps-per-episode",
        type=int,
        default=500,
        help="每个episode最大步数",
    )
    parser.add_argument(
        "--num-environments",
        type=int,
        default=4,
        help="并行环境数量",
    )
    parser.add_argument(
        "--opts",
        nargs="*",
        help="覆盖配置选项",
    )
    
    args = parser.parse_args()
    
    # 构建配置覆盖选项
    opts = [
        f"expert_data_collection.data_folder={args.data_folder}",
        f"expert_data_collection.split={args.split}",
        f"expert_data_collection.max_episodes={args.max_episodes}",
        f"expert_data_collection.max_steps_per_episode={args.max_steps_per_episode}",
        f"habitat_baselines.num_environments={args.num_environments}",
    ]
    
    if args.opts:
        opts.extend(args.opts)
    
    # 创建数据保存目录
    data_folder = Path(args.data_folder)
    data_folder.mkdir(parents=True, exist_ok=True)
    
    print(f"开始专家数据采集...")
    print(f"配置文件: {args.config_name}")
    print(f"数据保存目录: {args.data_folder}")
    print(f"数据集分割: {args.split}")
    print(f"最大episode数: {args.max_episodes}")
    print(f"每个episode最大步数: {args.max_steps_per_episode}")
    print(f"并行环境数量: {args.num_environments}")
    
    # 执行实验
    execute_exp(
        config_name=args.config_name,
        run_type="train",
        opts=opts,
    )


if __name__ == "__main__":
    main()
