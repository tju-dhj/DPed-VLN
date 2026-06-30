#!/bin/bash
#
# 自动批量评估脚本
# - 评估完一个checkpoint后自动评估下一个
# - 如果没有新checkpoint则等待新ckpt生成
# - 评估完成后自动保存SR指标
#
# 用法: bash auto_batch_eval.sh
#

# 配置
CKPT_DIR="/share/home/u19666033/dhj/DPed_pro/evaluation-vln/dped_pro_clip_rl_v2_6actions/hm3d/checkpoints"
OUTPUT_BASE="/share/home/u19666033/dhj/DPed_pro/evaluation-vln/dped_pro_batch_val"
EVAL_CONFIG="DPed_pro/eval/DPed_rl_val_6action_normalized"
# 注意：data_path 已在YAML配置中定义，不要在命令行覆盖

# 评估范围
START_CKPT=20
END_CKPT=45
INTERVAL=5

# 已评估的checkpoint记录文件
TRACKING_FILE="${OUTPUT_BASE}/evaluated_checkpoints.txt"
RESULTS_FILE="${OUTPUT_BASE}/eval_results.csv"

# 创建输出目录
mkdir -p "${OUTPUT_BASE}"

# 初始化追踪文件
touch "${TRACKING_FILE}"

# 初始化结果文件（如果不存在）
if [ ! -f "${RESULTS_FILE}" ]; then
    echo "checkpoint,sr,spl,stl,psc,num_steps,collision,reward" > "${RESULTS_FILE}"
fi

get_evaluated_ckpts() {
    # 获取已评���的checkpoint编号
    if [ -f "${TRACKING_FILE}" ]; then
        cat "${TRACKING_FILE}" | grep -v '^#' | grep -v '^$'
    fi
}

is_evaluated() {
    local ckpt_num=$1
    grep -q "^${ckpt_num}$" "${TRACKING_FILE}"
}

mark_evaluated() {
    local ckpt_num=$1
    echo "${ckpt_num}" >> "${TRACKING_FILE}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ckpt.${ckpt_num} 已评估" >> "${OUTPUT_BASE}/eval_log.txt"
}

get_next_ckpt() {
    # 获取下一个需要评估的checkpoint
    local evaluated=$(get_evaluated_ckpts | sort -n)
    
    for ckpt in $(seq ${START_CKPT} ${INTERVAL} ${END_CKPT}); do
        if ! is_evaluated "${ckpt}"; then
            echo "${ckpt}"
            return 0
        fi
    done
    return 1
}

wait_for_new_checkpoint() {
    # 等待新的checkpoint出现
    echo ""
    echo "========================================"
    echo "所有计划checkpoint已评估完毕！"
    echo "进入监控模式：等待新checkpoint生成..."
    echo "========================================"
    echo ""
    
    # 获取当前最大的checkpoint编号
    local max_ckpt=$(ls "${CKPT_DIR}"/ckpt.*.pth 2>/dev/null | grep -oP 'ckpt\.\d+' | sed 's/ckpt\.//' | sort -n | tail -1)
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 当前最大ckpt: ${max_ckpt}"
    
    while true; do
        sleep 60  # 每分钟检查一次
        
        local new_max=$(ls "${CKPT_DIR}"/ckpt.*.pth 2>/dev/null | grep -oP 'ckpt\.\d+' | sed 's/ckpt\.//' | sort -n | tail -1)
        
        if [ ! -z "${new_max}" ] && [ "${new_max}" -gt "${max_ckpt}" ]; then
            echo ""
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] 检测到新checkpoint: ckpt.${new_max}"
            echo "${new_max}" >> "${TRACKING_FILE}"
            echo "${new_max}"
            return 0
        fi
    done
}

extract_sr_from_log() {
    # 从评估日志中提取SR指标
    local log_file=$1
    local sr="" spl="" stl="" psc="" num_steps="" collision="" reward=""
    
    # 使用grep提取最后的指标值（评估结束时的平均值）
    sr=$(grep -oP 'Average episode success: \K[0-9.]+' "${log_file}" | tail -1)
    spl=$(grep -oP 'Average episode spl: \K[0-9.]+' "${log_file}" | tail -1)
    stl=$(grep -oP 'Average episode stl: \K[0-9.]+' "${log_file}" | tail -1)
    psc=$(grep -oP 'Average episode psc: \K[0-9.]+' "${log_file}" | tail -1)
    num_steps=$(grep -oP 'Average episode num_steps: \K[0-9.]+' "${log_file}" | tail -1)
    collision=$(grep -oP 'Average episode human_collision: \K[0-9.]+' "${log_file}" | tail -1)
    reward=$(grep -oP 'Average episode reward: \K[0-9.]+' "${log_file}" | tail -1)
    
    # 如果找到值就返回
    if [ ! -z "${sr}" ]; then
        echo "${sr},${spl},${stl},${psc},${num_steps},${collision},${reward}"
        return 0
    fi
    
    return 1
}

save_results() {
    local ckpt_num=$1
    local output_dir="${OUTPUT_BASE}/ckpt.${ckpt_num}"
    
    # 查找最近的日志文件
    local log_file=""
    local eval_output_dir="${output_dir}/checkpoints"
    
    if [ -d "${eval_output_dir}" ]; then
        # 查找.log或.out文件
        local latest_log=$(ls -t "${eval_output_dir}"/*.log "${eval_output_dir}"/*.out 2>/dev/null | head -1)
        if [ -f "${latest_log}" ]; then
            log_file="${latest_log}"
        fi
    fi
    
    echo ""
    echo "    提取指标: log_file=${log_file}"
    
    # 提取SR并保存
    if [ -f "${log_file}" ]; then
        local metrics=$(extract_sr_from_log "${log_file}")
        if [ ! -z "${metrics}" ]; then
            echo "${ckpt_num},${metrics}" >> "${RESULTS_FILE}"
            echo "    ✓ 已保存到 ${RESULTS_FILE}"
            echo "    SR: $(echo ${metrics} | cut -d',' -f1)"
            return 0
        fi
    fi
    
    echo "    ⚠ 无法提取指标"
    return 1
}

run_eval() {
    local ckpt_num=$1
    local ckpt_path="${CKPT_DIR}/ckpt.${ckpt_num}.pth"
    local output_dir="${OUTPUT_BASE}/ckpt.${ckpt_num}/checkpoints"
    local eval_log="${OUTPUT_BASE}/ckpt.${ckpt_num}/eval_output.log"
    
    # 检查checkpoint是否存在
    if [ ! -f "${ckpt_path}" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 错误: ${ckpt_path} 不存在!"
        return 1
    fi
    
    mkdir -p "${output_dir}"
    
    echo ""
    echo "========================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')]"
    echo "开始评估: ckpt.${ckpt_num}"
    echo "Checkpoint: ${ckpt_path}"
    echo "输出目录: ${output_dir}"
    echo "日志文件: ${eval_log}"
    echo "========================================"
    echo ""
    
    # 记录开始时间
    local start_time=$(date +%s)
    
    # 运行评估（同时输出到日志）
    cd /share/home/u19666033/dhj/DPed_pro
    
    python -u -m habitat-baselines.habitat_baselines.run \
        --config-name="${EVAL_CONFIG}" \
        habitat_baselines.eval_ckpt_path_dir="${ckpt_path}" \
        habitat_baselines.checkpoint_folder="${output_dir}" \
        2>&1 | tee "${eval_log}"
    
    local exit_code=${PIPESTATUS[0]}
    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))
    local elapsed_min=$((elapsed / 60))
    
    echo ""
    echo "========================================"
    if [ ${exit_code} -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ckpt.${ckpt_num} 评估完成! (用时: ${elapsed_min}分钟)"
        mark_evaluated "${ckpt_num}"
        
        # 从追踪文件中删除已完成的（确保不重复）
        sed -i "/^${ckpt_num}$/d" "${TRACKING_FILE}"
        echo "${ckpt_num}" >> "${TRACKING_FILE}"
        
        # 保存SR指标
        echo ""
        echo "    正在提取并保存指标..."
        save_results "${ckpt_num}"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ckpt.${ckpt_num} 评估失败! (exit code: ${exit_code}, 用时: ${elapsed_min}分钟)"
    fi
    echo "========================================"
    echo ""
    
    return ${exit_code}
}

# 主循环
echo ""
echo "========================================"
echo "自动批量评估脚本"
echo "========================================"
echo "Checkpoint目录: ${CKPT_DIR}"
echo "输出目录: ${OUTPUT_BASE}"
echo "评估范围: ckpt.${START_CKPT} 到 ckpt.${END_CKPT}，间隔${INTERVAL}"
echo "已评估记录: ${TRACKING_FILE}"
echo "结果保存: ${RESULTS_FILE}"
echo "========================================"
echo ""

# 激活conda环境提示
echo "请确保在运行此脚本前已激活conda环境: conda activate falcon"
echo ""

# 主循环
while true; do
    # 获取下一个checkpoint
    next_ckpt=$(get_next_ckpt)
    
    if [ -z "${next_ckpt}" ]; then
        # 所有计划的checkpoint都已评估完
        next_ckpt=$(wait_for_new_checkpoint)
    fi
    
    if [ -z "${next_ckpt}" ]; then
        echo "错误: 无法获取checkpoint编号"
        sleep 10
        continue
    fi
    
    # 运行评估
    run_eval "${next_ckpt}"
    
    # 评估失败时等待一下再重试
    if [ $? -ne 0 ]; then
        echo "评估失败，10秒后重试..."
        sleep 10
    fi
    
    # 短暂休息
    sleep 5
done