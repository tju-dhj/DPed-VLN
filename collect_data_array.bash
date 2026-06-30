#!/bin/bash
#SBATCH --job-name=dhj_falcon_collect
#SBATCH --output=slurm_logs/collect_%A_%a.out
#SBATCH --error=slurm_logs/collect_%A_%a.err
#SBATCH --wckey=p19666033
#SBATCH -A p_p19666033
#SBATCH -p L40
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --gres=gpu:l40:1
#SBATCH --array=0-7              # 8个独立任务，每个任务1张GPU
#SBATCH --time=48:00:00

# ============================================
# 说明: 
# - 每个任务独立申请1张GPU
# - 哪张GPU先分配到就先开始运行
# - 8个任务会采集不同的数据chunk
# ============================================

# 激活conda环境
source /share/apps/miniconda3/etc/profile.d/conda.sh
conda activate falcon

# 进入项目目录
cd /share/home/u19666033/dhj/falcon_collect_data/Falcon-main

# 创建日志目录
mkdir -p slurm_logs

# ============================================
# 配置参数
# ============================================
TOTAL_EPISODES=100000
NUM_TASKS=8
EPISODES_PER_TASK=$((TOTAL_EPISODES / NUM_TASKS))

# 当前任务ID
TASK_ID=$SLURM_ARRAY_TASK_ID

# 输出目录（每个任务独立的chunk）
OUTPUT_DIR="/share/home/u19666033/dhj/falcon_collect_data/Falcon-main/data/collect_data/train_chunk_${TASK_ID}"

# 随机种子（确保每个任务采集不同的数据）
RANDOM_SEED=$((42 + TASK_ID * 10000))

# ============================================
# 打印任务信息
# ============================================
echo "========================================"
echo "数据采集任务 - Chunk ${TASK_ID}"
echo "========================================"
echo "Job ID:          $SLURM_JOB_ID"
echo "Array Job ID:    $SLURM_ARRAY_JOB_ID"
echo "Task ID:         $TASK_ID"
echo "Node:            $SLURM_NODELIST"
echo "GPU:             $CUDA_VISIBLE_DEVICES"
echo "----------------------------------------"
echo "本任务Episodes:  $EPISODES_PER_TASK"
echo "输出目录:        $OUTPUT_DIR"
echo "随机种子:        $RANDOM_SEED"
echo "========================================"
echo "开始时间: $(date)"
echo "========================================"

# ============================================
# 运行数据采集
# ============================================
python -u -m habitat_baselines.habitat_baselines.run \
    --config-name=dynamic_vlnce/collect_data_multi.yaml \
    expert_data_collection.data_folder="$OUTPUT_DIR" \
    expert_data_collection.max_episodes=$EPISODES_PER_TASK \
    habitat.seed=$RANDOM_SEED

# ============================================
# 完成信息
# ============================================
EXIT_CODE=$?
echo "========================================"
echo "结束时间: $(date)"
echo "退出代码: $EXIT_CODE"
echo "========================================"

# 统计采集的数据
if [ -d "$OUTPUT_DIR" ]; then
    echo "统计采集的数据..."
    TOTAL_COUNT=0
    if [ -d "$OUTPUT_DIR/train" ]; then
        for scene_dir in "$OUTPUT_DIR/train"/*; do
            if [ -d "$scene_dir" ]; then
                SCENE_COUNT=$(find "$scene_dir" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
                TOTAL_COUNT=$((TOTAL_COUNT + SCENE_COUNT))
            fi
        done
    fi
    echo "已采集Episodes: $TOTAL_COUNT / $EPISODES_PER_TASK"
fi

exit $EXIT_CODE

