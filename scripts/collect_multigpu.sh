#!/bin/bash
# 多GPU并行数据采集脚本
# 使用方法: bash scripts/collect_multigpu.sh [GPU数量]
export PYTHONPATH="/share/home/u19666033/dhj/falcon_collect_data/Falcon-main:$PYTHONPATH"
# 清除旧路径相关模块缓存（关键！）
find /share/home/u19666033/dhj/falcon_collect_data/Falcon-main -name "*.pyc" -delete
find /share/home/u19666033/dhj/falcon_collect_data/Falcon-main -name "__pycache__" -type d -exec rm -rf {} +

# 清除 Python 模块缓存（运行时）
python -c "import sys; [sys.modules.pop(k) for k in list(sys.modules) if 'habitat' in k or 'Falcon' in k]; print('Cleaned cached modules')"
# 切换到项目根目录
cd /share/home/u19666033/dhj/falcon_collect_data/Falcon-main

# 从命令行参数获取GPU数量，默认为4
NUM_GPUS=${1:-4}
TOTAL_EPISODES=100000
EPISODES_PER_GPU=$((TOTAL_EPISODES / NUM_GPUS))

echo "========================================"
echo "启动多GPU并行数据采集"
echo "GPU数量: $NUM_GPUS"
echo "总Episodes: $TOTAL_EPISODES"
echo "每GPU Episodes: $EPISODES_PER_GPU"
echo "当前目录: $(pwd)"
echo "========================================"

# 创建日志目录
mkdir -p logs/multigpu

# 为每个GPU启动一个独立进程
for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    # 设置每个GPU的输出目录
    OUTPUT_DIR="/share/home/u19666033/dhj/falcon_collect_data/Falcon-main/data/collect_data/train_part_$((GPU_ID + 1))"
    
    echo "----------------------------------------"
    echo "启动GPU $GPU_ID:"
    echo "  目标Episodes: $EPISODES_PER_GPU"
    echo "  输出目录: $OUTPUT_DIR"
    echo "  日志文件: logs/multigpu/gpu_${GPU_ID}.log"
    
    # 在后台启动进程（使用与您测试时相同的命令和配置文件）
    CUDA_VISIBLE_DEVICES=$GPU_ID python -u -m habitat_baselines.habitat_baselines.run \
        --config-name=dynamic_vlnce/collect_data_multi.yaml \
        habitat_baselines.torch_gpu_id=0 \
        habitat.simulator.habitat_sim_v0.gpu_device_id=0 \
        expert_data_collection.data_folder="$OUTPUT_DIR" \
        expert_data_collection.max_episodes=$EPISODES_PER_GPU \
        habitat.seed=$((42 + GPU_ID * 10000)) \
        > logs/multigpu/gpu_${GPU_ID}.log 2>&1 &
    
    # 保存进程ID
    PID=$!
    echo $PID > logs/multigpu/gpu_${GPU_ID}.pid
    echo "  进程ID: $PID"
    
    # 短暂等待，避免同时初始化导致资源竞争
    sleep 15
done

echo "========================================"
echo "所有进程已启动完成！"
echo ""
echo "📊 监控命令:"
echo "  查看所有日志:    tail -f logs/multigpu/gpu_*.log"
echo "  查看GPU 0日志:   tail -f logs/multigpu/gpu_0.log"
echo "  查看进程状态:    ps aux | grep habitat_baselines.run"
echo ""
echo "🛑 控制命令:"
echo "  停止所有进程:    kill \$(cat logs/multigpu/*.pid)"
echo "  强制停止:        kill -9 \$(cat logs/multigpu/*.pid)"
echo ""
echo "📁 数据输出目录:"
for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    echo "  GPU $GPU_ID: data/collect_data/train_part_$((GPU_ID + 1))/"
done
echo "========================================"

