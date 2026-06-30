#!/bin/bash
# 对比 Baseline v2 和 Ablation (无人) 配置的差异

echo "========================================================================"
echo "配置对比：Baseline v2 vs. Ablation (无人)"
echo "========================================================================"
echo ""

CONFIG_DIR="/share/home/u14004/dhj/Falcon-main/habitat-baselines/habitat_baselines/config/dynamic_vlnce"
BASELINE="${CONFIG_DIR}/dynamic_vlnce_hm3d_train_v2.yaml"
ABLATION="${CONFIG_DIR}/dynamic_vlnce_hm3d_train_v2_ablation_no_human.yaml"

echo "📁 文件路径："
echo "  Baseline: ${BASELINE}"
echo "  Ablation: ${ABLATION}"
echo ""

echo "========================================================================"
echo "🔍 关键差异对比"
echo "========================================================================"
echo ""

# 1. 奖励函数
echo "1️⃣  奖励函数 (reward_measure)"
echo "--------------------------------------------------------------------"
echo "Baseline:"
grep -A1 "reward_measure:" "${BASELINE}" | head -2 || echo "  未找到"
echo ""
echo "Ablation:"
grep -A1 "reward_measure:" "${ABLATION}" | head -2 || echo "  未找到"
echo ""

# 2. 人相关惩罚
echo "2️⃣  人相关惩罚 (collide_human_penalty, close_to_human_penalty)"
echo "--------------------------------------------------------------------"
echo "Baseline:"
grep -E "collide_human_penalty:|close_to_human_penalty:|trajectory_cover_penalty:" "${BASELINE}" || echo "  未找到"
echo ""
echo "Ablation:"
grep -E "collide_human_penalty:|close_to_human_penalty:|trajectory_cover_penalty:" "${ABLATION}" || echo "  未找到"
echo ""

# 3. 辅助任务
echo "3️⃣  辅助任务 (auxiliary_losses)"
echo "--------------------------------------------------------------------"
echo "Baseline:"
grep -A15 "auxiliary_losses:" "${BASELINE}" | head -16 || echo "  未找到"
echo ""
echo "Ablation:"
grep -A3 "auxiliary_losses:" "${ABLATION}" | head -4 || echo "  未找到"
echo ""

# 4. 人传感器
echo "4️⃣  人传感器 (human_num_sensor, oracle_humanoid_future_trajectory)"
echo "--------------------------------------------------------------------"
echo "Baseline:"
grep -E "human_num_sensor|oracle_humanoid_future_trajectory" "${BASELINE}" || echo "  未找到"
echo ""
echo "Ablation:"
grep -E "human_num_sensor|oracle_humanoid_future_trajectory" "${ABLATION}" || echo "  未找到（已注释）"
echo ""

# 5. 保存目录
echo "5️⃣  保存目录 (checkpoint_folder, tensorboard_dir)"
echo "--------------------------------------------------------------------"
echo "Baseline:"
grep "checkpoint_folder:" "${BASELINE}" || echo "  未找到"
grep "tensorboard_dir:" "${BASELINE}" || echo "  未找到"
echo ""
echo "Ablation:"
grep "checkpoint_folder:" "${ABLATION}" || echo "  未找到"
grep "tensorboard_dir:" "${ABLATION}" || echo "  未找到"
echo ""

echo "========================================================================"
echo "✅ 对比完成！"
echo "========================================================================"
echo ""
echo "📝 完整差异请查看："
echo "  diff ${BASELINE} ${ABLATION}"
echo ""
echo "🚀 运行实验："
echo "  Baseline: sbatch main_slurm_rl_train_v2.bash"
echo "  Ablation: sbatch main_slurm_rl_train_v2_ablation_no_human.bash"
echo ""

