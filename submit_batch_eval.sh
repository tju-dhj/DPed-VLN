#!/bin/bash
#
# 批量评估提交脚本
# 自动提交所有checkpoint的评估作业，并生成汇总表格
#

set -euo pipefail

# 配置
CKPT_DIR="/share/home/u19666033/dhj/DPed_pro/evaluation-vln/dped_pro_clip_rl_v2_6actions/hm3d/checkpoints"
OUTPUT_BASE="/share/home/u19666033/dhj/DPed_pro/evaluation-vln/dped_pro_batch_val"
LOG_DIR="/share/home/u19666033/dhj/DPed_pro/slurm_logs/batch_eval"
SCRIPT_DIR="/share/home/u19666033/dhj/DPed_pro/sbatch"
RESULTS_FILE="${OUTPUT_BASE}/batch_eval_summary.csv"

# 创建目录
mkdir -p "${OUTPUT_BASE}"
mkdir -p "${LOG_DIR}"

echo "========================================"
echo "批量评估提交脚本"
echo "========================================"

# 查找所有checkpoint
echo "扫描checkpoint目录..."
CKPTS=$(ls ${CKPT_DIR}/ckpt.*.pth 2>/dev/null | grep -oP 'ckpt\.\d+' | sort -t. -k2 -n | uniq)

if [ -z "${CKPTS}" ]; then
    echo "错误: 未找到checkpoint文件!"
    exit 1
fi

echo "找到以下checkpoint:"
echo "${CKPTS}"
echo ""

# 初始化结果文件
echo "ckpt_num,submitted,completed,sr,status" > "${RESULTS_FILE}"

# 提交作业并记录
JOB_IDS=""
for ckpt in ${CKPTS}; do
    ckpt_num=$(echo ${ckpt} | sed 's/ckpt\.//')
    
    echo "提交 ckpt.${ckpt_num}..."
    
    # 提交作业
    job_id=$(sbatch ${SCRIPT_DIR}/eval_single_ckpt.bash ${ckpt_num} 2>&1 | grep -oP '\d+' | tail -1)
    
    echo "${ckpt_num},${job_id},pending,N/A,submitted" >> "${RESULTS_FILE}"
    JOB_IDS="${JOB_IDS} ${job_id}"
    
    echo "  作业ID: ${job_id}"
    
    # 避免提交过快
    sleep 1
done

echo ""
echo "========================================"
echo "已提交 ${#JOB_IDS[@]} 个评估作业"
echo "========================================"
echo "作业ID: ${JOB_IDS}"
echo ""
echo "监控命令:"
echo "  squeue -u u19666033 | grep batch_eval"
echo ""
echo "汇总文件: ${RESULTS_FILE}"
echo ""
echo "完成后查看结果:"
echo "  cat ${RESULTS_FILE}"
echo "========================================"
