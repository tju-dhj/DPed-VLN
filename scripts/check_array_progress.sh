#!/bin/bash
# 检查Job Array数据采集进度

cd /share/home/u19666033/dhj/falcon_collect_data/Falcon-main

echo "========================================"
echo "数据采集进度检查"
echo "时间: $(date)"
echo "========================================"

# 检查SLURM任务状态
echo ""
echo "📊 任务状态:"
echo ""
squeue -u $USER -o "%.10i %.12j %.8T %.10M %.6D %.15R" --sort=i

# 统计各个chunk的数据
echo ""
echo "========================================"
echo "📁 各Chunk数据统计:"
echo "========================================"
TOTAL_EPISODES=0
COMPLETED_TASKS=0
RUNNING_TASKS=0

for i in {0..7}; do
    chunk_dir="data/collect_data/train_chunk_${i}"
    
    if [ -d "$chunk_dir" ]; then
        EPISODE_COUNT=0
        if [ -d "$chunk_dir/train" ]; then
            for scene_dir in "$chunk_dir/train"/*; do
                if [ -d "$scene_dir" ]; then
                    SCENE_EPISODES=$(find "$scene_dir" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
                    EPISODE_COUNT=$((EPISODE_COUNT + SCENE_EPISODES))
                fi
            done
        fi
        
        TOTAL_EPISODES=$((TOTAL_EPISODES + EPISODE_COUNT))
        
        # 检查任务状态
        STATUS="❓"
        if [ $EPISODE_COUNT -gt 0 ]; then
            if [ $EPISODE_COUNT -ge 12000 ]; then
                STATUS="✅"
                COMPLETED_TASKS=$((COMPLETED_TASKS + 1))
            else
                STATUS="🔄"
                RUNNING_TASKS=$((RUNNING_TASKS + 1))
            fi
        fi
        
        printf "  Chunk %d: %s %6d episodes\n" $i "$STATUS" $EPISODE_COUNT
    else
        printf "  Chunk %d: ⏸️  未开始\n" $i
    fi
done

echo "  ========================================"
printf "  总计:      %6d episodes\n" $TOTAL_EPISODES
printf "  完成任务:  %d/8\n" $COMPLETED_TASKS
printf "  运行中:    %d\n" $RUNNING_TASKS

# 显示磁盘使用
echo ""
echo "========================================"
echo "💾 磁盘使用:"
echo "========================================"
if [ -d "data/collect_data" ]; then
    du -sh data/collect_data/train_chunk_* 2>/dev/null | while read size dir; do
        chunk_name=$(basename "$dir")
        printf "  %-20s %s\n" "$chunk_name" "$size"
    done
    echo "  ========================================"
    du -sh data/collect_data
fi

echo ""
echo "========================================"
echo "💡 常用命令:"
echo "========================================"
echo "  查看任务:     squeue -u \$USER"
echo "  查看日志:     tail -f slurm_logs/collect_JOBID_TASKID.out"
echo "  取消所有:     scancel -u \$USER -n dhj_falcon_collect"
echo "  重新检查:     bash scripts/check_array_progress.sh"
echo "========================================"

