#!/bin/bash
# 测试NaVILA消融配置是否正确

source /share/home/u14004/.bashrc
conda activate falcon

cd /share/home/u14004/dhj/Falcon-main

echo "========================================================================"
echo "测试 Ablation v2 (人静止) 配置"
echo "========================================================================"
echo ""

# 只验证配置加载，不实际运行
python -c "
from omegaconf import OmegaConf
import sys

try:
    cfg = OmegaConf.load('habitat-baselines/habitat_baselines/config/dynamic_vlnce/navila_falcon_hm3d_v1_ablation_static_only.yaml')
    print('✅ 配置文件加载成功！')
    print('')
    print('关键配置检查：')
    print(f'  - num_agent_types: {cfg.habitat_baselines.rl.agent.num_agent_types}')
    print(f'  - reward_measure: {cfg.habitat.task.reward_measure}')
    
    # 检查人类agent动作速度
    for i in range(1, 7):
        action_key = f'agent_{i}_oracle_nav_randcoord_action_obstacle'
        if action_key in cfg.habitat.task.actions:
            lin_speed = cfg.habitat.task.actions[action_key].lin_speed
            ang_speed = cfg.habitat.task.actions[action_key].ang_speed
            print(f'  - {action_key}: lin_speed={lin_speed}, ang_speed={ang_speed}')
    
    sys.exit(0)
except Exception as e:
    print(f'❌ 配置文件加载失败: {e}')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "✅ 配置验证通过！可以运行实验了。"
    echo "========================================================================"
    echo ""
    echo "运行命令："
    echo "  sbatch main_slurm_navila_v1_ablation_static_only.bash"
else
    echo ""
    echo "========================================================================"
    echo "❌ 配置验证失败，请检查配置文件。"
    echo "========================================================================"
fi

