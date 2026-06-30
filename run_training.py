#!/usr/bin/env python3
"""
训练入口脚本 - 在导入任何 habitat 模块之前修复 editable 安装路径
"""

import sys
import os

# ========== 修复 editable 安装路径 ==========
_correct_habitat_lab_path = "/share/home/u19666033/dhj/DPed_pro/habitat-lab/habitat"
_correct_habitat_baselines_path = "/share/home/u19666033/dhj/DPed_pro/habitat-baselines/habitat_baselines"

# 方法1: 尝试修改 editable finder 的 MAPPING
try:
    import __editable___habitat_lab_0_3_1_finder as _habitat_finder
    if hasattr(_habitat_finder, 'MAPPING') and 'habitat' in _habitat_finder.MAPPING:
        _old_path = _habitat_finder.MAPPING['habitat']
        if _old_path != _correct_habitat_lab_path:
            _habitat_finder.MAPPING['habitat'] = _correct_habitat_lab_path
            print(f"[run_training] Fixed habitat path: {_old_path} -> {_correct_habitat_lab_path}", flush=True)
except ImportError:
    print("[run_training] No editable habitat_lab finder found, using sys.path", flush=True)

try:
    import __editable___habitat_baselines_0_3_1_finder as _baselines_finder
    if hasattr(_baselines_finder, 'MAPPING') and 'habitat_baselines' in _baselines_finder.MAPPING:
        _old_path = _baselines_finder.MAPPING['habitat_baselines']
        if _old_path != _correct_habitat_baselines_path:
            _baselines_finder.MAPPING['habitat_baselines'] = _correct_habitat_baselines_path
            print(f"[run_training] Fixed habitat_baselines path: {_old_path} -> {_correct_habitat_baselines_path}", flush=True)
except ImportError:
    print("[run_training] No editable habitat_baselines finder found, using sys.path", flush=True)

# 方法2: 将正确路径添加到 sys.path 的最前面（作为备用）
sys.path.insert(0, "habitat-lab")
sys.path.insert(0, "habitat-baselines")
# ========== 修复完成 ==========

# 验证路径
print(f"[run_training] sys.path[0]: {sys.path[0]}", flush=True)
print(f"[run_training] sys.path[1]: {sys.path[1]}", flush=True)

# 现在导入并运行 habitat_baselines
if __name__ == "__main__":
    from habitat_baselines.run import main
    main()












