#!/bin/bash
#SBATCH --job-name=dhj_falcon_split
#SBATCH --output=slurm_logs/split_%A_%a.out
#SBATCH --error=slurm_logs/split_%A_%a.err
#SBATCH --wckey=p19666033
#SBATCH -A p_p19666033
#SBATCH -p L40
#SBATCH --ntasks=1               # 明确指定1个任务
#SBATCH --nodes=1                # 明确指定1个节点
#SBATCH --gres=gpu:l40:1
#SBATCH --array=0-4              # 5个任务（0,1,2,3,4）
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8        # 每个任务的CPU数

# ============================================
# Episode定义文件划分版本
# 每个Task处理不同的episode子集，避免重复采集
# ============================================
# 使用前先运行: python scripts/split_episodes.py
# ============================================

# 激活conda环境
source /share/apps/miniconda3/etc/profile.d/conda.sh
conda activate falcon

# 进入项目目录
cd /share/home/u19666033/dhj/falcon_collect_data/Falcon-main

# 设置Python路径，确保使用正确的代码路径
export PYTHONPATH=/share/home/u19666033/dhj/falcon_collect_data/Falcon-main/habitat-baselines:/share/home/u19666033/dhj/falcon_collect_data/Falcon-main/habitat-lab:$PYTHONPATH

# 创建日志目录
mkdir -p slurm_logs

# ============================================
# 配置参数
# ============================================
NUM_SPLITS=5
TASK_ID=$SLURM_ARRAY_TASK_ID

# 每个Task使用不同的episode子集
EPISODE_CONTENT_DIR="/share/home/u19666033/dhj/falcon_collect_data/Falcon-main/data/datasets/pointnav/social-hm3d/train/content_split_${TASK_ID}"

# 输出目录
OUTPUT_DIR="/share/home/u19666033/dhj/falcon_collect_data/Falcon-main/data/collect_data/train_split_${TASK_ID}"

# 随机种子（仍然使用不同的种子以增加数据多样性）
RANDOM_SEED=$((42 + TASK_ID * 10000))

# ============================================
# 检查episode子集是否存在
# ============================================
if [ ! -d "$EPISODE_CONTENT_DIR" ]; then
    echo "========================================"
    echo "错误: Episode子集目录不存在!"
    echo "========================================"
    echo "目录: $EPISODE_CONTENT_DIR"
    echo ""
    echo "请先运行划分脚本:"
    echo "  python scripts/split_episodes.py"
    echo "========================================"
    exit 1
fi

# 统计这个split的episode文件数
NUM_EPISODE_FILES=$(find "$EPISODE_CONTENT_DIR" -name "*.json.gz" | wc -l)

# ============================================
# 打印任务信息
# ============================================
echo "========================================"
echo "数据采集任务 - Split ${TASK_ID}"
echo "========================================"
echo "Job ID:          $SLURM_JOB_ID"
echo "Array Job ID:    $SLURM_ARRAY_JOB_ID"
echo "Task ID:         $TASK_ID"
echo "Node:            $SLURM_NODELIST"
echo "GPU:             $CUDA_VISIBLE_DEVICES"
echo "----------------------------------------"
echo "Episode子集:     $EPISODE_CONTENT_DIR"
echo "Episode文件数:   $NUM_EPISODE_FILES"
echo "输出目录:        $OUTPUT_DIR"
echo "随机种子:        $RANDOM_SEED"
echo "========================================"
echo "开始时间: $(date)"
echo "========================================"

# ============================================
# 创建临时配置文件覆盖
# ============================================
# 需要修改配置以使用指定的episode子集
CONFIG_OVERRIDE="habitat.dataset.content_scenes=['*']"
CONFIG_OVERRIDE="$CONFIG_OVERRIDE habitat.dataset.data_path='$EPISODE_CONTENT_DIR/{split}/{content_scene}.json.gz'"

# ============================================
# 运行数据采集
# ============================================
# 设置环境变量，强制单GPU模式，禁用分布式
export HABITAT_ENV_DEBUG=1
export CUDA_VISIBLE_DEVICES=0
# 重要：取消SLURM相关环境变量，避免触发分布式训练
unset SLURM_NTASKS
unset SLURM_JOB_NUM_NODES

# 注意: 这里不设置max_episodes，让它采集这个split中的所有episodes
python -u -m habitat_baselines.habitat_baselines.run \
    --config-name=dynamic_vlnce/collect_data_multi.yaml \
    expert_data_collection.data_folder="$OUTPUT_DIR" \
    habitat.dataset.content_scenes=['*'] \
    habitat.dataset.data_path="$EPISODE_CONTENT_DIR"'/{split}/{content_scene}.json.gz' \
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
    echo "已采集Episodes: $TOTAL_COUNT"
    
    # 计算预期的episodes数量
    EXPECTED_COUNT=0
    for ep_file in "$EPISODE_CONTENT_DIR"/*.json.gz; do
        if [ -f "$ep_file" ]; then
            COUNT=$(python3 -c "import gzip,json; print(len(json.load(gzip.open('$ep_file'))['episodes']))" 2>/dev/null || echo 0)
            EXPECTED_COUNT=$((EXPECTED_COUNT + COUNT))
        fi
    done
    
    echo "预期Episodes: $EXPECTED_COUNT"
    
    if [ $TOTAL_COUNT -eq $EXPECTED_COUNT ]; then
        echo "✅ 数据采集完整"
    else
        echo "⚠️  采集数量与预期不符"
    fi
fi

exit $EXIT_CODE

