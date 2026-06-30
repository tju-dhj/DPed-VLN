#!/bin/bash
#
# 批量评估提交脚本 (tmux单卡版)
# 从第20个checkpoint开始，每5个评估一次
#

# 配置
CKPT_DIR="/share/home/u19666033/dhj/DPed_pro/evaluation-vln/dped_pro_clip_rl_v2_6actions/hm3d/checkpoints"
OUTPUT_BASE="/share/home/u19666033/dhj/DPed_pro/evaluation-vln/dped_pro_batch_val"
SCRIPT_DIR="/share/home/u19666033/dhj/DPed_pro/sbatch"

# 创建输出目录
mkdir -p "${OUTPUT_BASE}"

echo "========================================"
echo "批量评估脚本 (tmux单卡版)"
echo "========================================"
echo "评估策略: 从ckpt.20开始，每5个评估一次"
echo "即: ckpt.20, 25, 30, 35, 40..."
echo ""

# 生成要评估的checkpoint列表 (20, 25, 30, 35, ...)
EVAL_LIST=""
for i in $(seq 20 5 45); do
    ckpt_path="${CKPT_DIR}/ckpt.${i}.pth"
    if [ -f "${ckpt_path}" ]; then
        EVAL_LIST="${EVAL_LIST} ${i}"
        echo "  ✓ ckpt.${i} (存在)"
    else
        echo "  ✗ ckpt.${i} (不存在)"
    fi
done

echo ""
echo "========================================"
echo "将在以下tmux窗口中逐个运行:"
echo ""

# 生成tmux命令
for ckpt_num in ${EVAL_LIST}; do
    OUTPUT_DIR="${OUTPUT_BASE}/ckpt.${ckpt_num}/checkpoints"
    mkdir -p "${OUTPUT_DIR}"
    
    CMD="cd /share/home/u19666033/dhj/DPed_pro && \
python -u -m habitat-baselines.habitat_baselines.run \
    --config-name=DPed_pro/eval/DPed_rl_val_6action_normalized \
    habitat_baselines.eval_ckpt_path_dir=${CKPT_DIR}/ckpt.${ckpt_num}.pth \
    habitat_baselines.checkpoint_folder=${OUTPUT_DIR} \
    habitat.dataset.data_path=/share/home/u19666033/dhj/DPed_pro/data/DPed_pro/val_evalfast/{scene}.json.gz"
    
    echo "========================================"
    echo "ckpt.${ckpt_num}"
    echo "========================================"
    echo "输出目录: ${OUTPUT_DIR}"
    echo ""
    echo "运行命令:"
    echo "${CMD}"
    echo ""
    
    # 输出用于tmux发送的命令格式
    echo "---TMUX_CMD_START---"
    echo "${CMD}"
    echo "---TMUX_CMD_END---"
    echo ""
done

echo "========================================"
echo "使用方法:"
echo "1. tmux new -s eval_batch"
echo "2. 粘贴并运行上面的命令"
echo "3. 完成后按 Ctrl+B D 分离"
echo "4. 完成后回来运行第5个间隔的命令"
echo "========================================"
