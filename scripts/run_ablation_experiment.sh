#!/bin/bash
# 快速启动消融实验

echo "========================================================================"
echo "🚀 启动消融实验：移除动态行人模块"
echo "========================================================================"
echo ""

cd /share/home/u14004/dhj/Falcon-main

# 创建日志目录
mkdir -p slurm_logs/rl_v2_ablation_no_human
echo "✅ 创建日志目录: slurm_logs/rl_v2_ablation_no_human"
echo ""

# 显示配置对比
echo "📋 配置对比："
echo "--------------------------------------------------------------------"
bash scripts/compare_v2_configs.sh | grep -A5 "关键差异对比" | tail -4
echo ""

# 提交训练任务
echo "========================================================================"
echo "🎯 提交SLURM任务"
echo "========================================================================"
echo ""

read -p "是否提交消融实验训练任务？(y/n): " confirm
if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
    echo "提交中..."
    JOB_ID=$(sbatch main_slurm_rl_train_v2_ablation_no_human.bash | awk '{print $4}')
    echo ""
    echo "✅ 任务已提交！"
    echo "   Job ID: ${JOB_ID}"
    echo "   输出日志: slurm_logs/rl_v2_ablation_no_human/${JOB_ID}_rl_v2_ablation_no_human.out"
    echo "   错误日志: slurm_logs/rl_v2_ablation_no_human/${JOB_ID}_rl_v2_ablation_no_human.err"
    echo ""
    echo "📊 监控命令："
    echo "   tail -f slurm_logs/rl_v2_ablation_no_human/${JOB_ID}_rl_v2_ablation_no_human.out"
    echo ""
    echo "📈 TensorBoard:"
    echo "   tensorboard --logdir=evaluation-vln/dynamic_vlnce_clip_rl_v2_ablation_no_human/hm3d/tb"
    echo ""
else
    echo "❌ 取消提交"
    echo ""
    echo "手动提交命令："
    echo "   sbatch main_slurm_rl_train_v2_ablation_no_human.bash"
fi

echo "========================================================================"
echo "📚 更多信息请查看: ABLATION_STUDY_NO_HUMAN.md"
echo "========================================================================"

