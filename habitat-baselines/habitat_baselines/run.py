#!/usr/bin/env python3
import multiprocessing as mp
mp.set_start_method('spawn', force=True)
import os
print(f"Executing run.py from: {os.path.abspath(__file__)}")
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
# 抑制Habitat渲染相关的警告输出
os.environ["GLOG_minloglevel"] = "2"  # 抑制INFO级别以下的日志
os.environ["MAGNUM_LOG"] = "quiet"   # 抑制Magnum渲染引擎的日志
os.environ["HABITAT_SIM_LOG"] = "quiet"  # 抑制Habitat Sim的日志

import random
import sys

# 获取当前文件的目录（动态获取，不硬编码）
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（habitat-baselines目录）
project_root = os.path.dirname(current_dir)
# 获取Falcon-main根目录（habitat-baselines的父目录）
falcon_root = os.path.dirname(project_root)
# 获取habitat-lab目录（通常与habitat-baselines同级）
habitat_lab_dir = os.path.join(falcon_root, "habitat-lab")

# 动态添加路径到 sys.path（避免硬编码用户路径）
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if habitat_lab_dir not in sys.path and os.path.exists(habitat_lab_dir):
    sys.path.insert(0, habitat_lab_dir)
if falcon_root not in sys.path:
    sys.path.append(falcon_root)
from typing import TYPE_CHECKING

import gc
import hydra
import numpy as np
import torch

from habitat.config.default import patch_config
from habitat.config.default_structured_configs import register_hydra_plugin
from habitat_baselines.config.default_structured_configs import (
    HabitatBaselinesConfigPlugin,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig

## for import functions related to falcon
try:
    import falcon
except ImportError:
    # 如果falcon模块不在标准位置，尝试从falcon_root导入
    if falcon_root not in sys.path:
        sys.path.insert(0, falcon_root)
    try:
        import falcon
    except ImportError:
        # 如果仍然无法导入，给出警告但不阻止运行
        print(f"Warning: Could not import falcon module. Some features may not work.")
        print(f"Tried paths: {falcon_root}, {sys.path[:5]}")

@hydra.main(
    version_base=None,
    config_path="config",
    config_name="pointnav/ppo_pointnav_example",
)
def main(cfg: "DictConfig"):
    cfg = patch_config(cfg)
    execute_exp(cfg, "eval" if cfg.habitat_baselines.evaluate else "train")


def execute_exp(config: "DictConfig", run_type: str) -> None:
    """This function runs the specified config with the specified runtype
    Args:
    config: Habitat.config
    runtype: str {train or eval}
    """
     #设置随机种子
    random.seed(config.habitat.seed)
    np.random.seed(config.habitat.seed)
    torch.manual_seed(config.habitat.seed)
    #设置PyTorch单线程
    if (
        config.habitat_baselines.force_torch_single_threaded
        and torch.cuda.is_available()
    ):
        torch.set_num_threads(1)
    
    # GPU内存优化设置 + 确定性训练
    if torch.cuda.is_available():
        # 设置内存分配策略
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")
        # 启用内存池以减少碎片
        torch.cuda.empty_cache()
        # 固定所有CUDA随机种子
        torch.cuda.manual_seed(config.habitat.seed)
        torch.cuda.manual_seed_all(config.habitat.seed)
        # 启用确定性算法（代价：略微降低训练速度）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    #获取训练器
    from habitat_baselines.common.baseline_registry import baseline_registry

    trainer_init = baseline_registry.get_trainer(
        config.habitat_baselines.trainer_name
    )
    assert (
        trainer_init is not None
    ), f"{config.habitat_baselines.trainer_name} is not supported"
    trainer = trainer_init(config)

    if run_type == "train":
        trainer.train()
    elif run_type == "eval":
        trainer.eval()


if __name__ == "__main__":
    register_hydra_plugin(HabitatBaselinesConfigPlugin)
    if "--exp-config" in sys.argv or "--run-type" in sys.argv:
        raise ValueError(
            "The API of run.py has changed to be compatible with hydra.\n"
            "--exp-config is now --config-name and is a config path inside habitat-baselines/habitat_baselines/config/. \n"
            "--run-type train is replaced with habitat_baselines.evaluate=False (default) and --run-type eval is replaced with habitat_baselines.evaluate=True.\n"
            "instead of calling:\n\n"
            "python -u -m habitat_baselines.run --exp-config habitat-baselines/habitat_baselines/config/<path-to-config> --run-type train/eval\n\n"
            "You now need to do:\n\n"
            "python -u -m habitat_baselines.run --config-name=<path-to-config> habitat_baselines.evaluate=False/True\n"
        )
    main()