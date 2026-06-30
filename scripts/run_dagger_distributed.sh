#!/bin/bash
# 分布式DAgger训练启动脚本
# 使用方法: bash scripts/run_dagger_distributed.sh

# 设置CUDA设备可见性（可选，如果不设置则使用所有GPU）
# export CUDA_VISIBLE_DEVICES=0,1

# 设置分布式训练参数
export MASTER_ADDR=${MASTER_ADDR:-"localhost"}

# 自动查找可用端口（如果29500被占用）
if [ -z "$MASTER_PORT" ]; then
    # 使用Python来检测可用端口（更可靠）
    # 尝试从29500开始找可用端口
    PORT_FOUND=0
    for port in 29500 29501 29502 29503 29504 29505; do
        # 使用Python的socket库检查端口是否可用
        python3 -c "
import socket
import sys
port = $port
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(('localhost', port))
    s.close()
    sys.exit(0)  # 端口可用
except Exception:
    sys.exit(1)  # 端口被占用
" 2>/dev/null
        
        if [ $? -eq 0 ]; then
            export MASTER_PORT=$port
            echo "检测到可用端口: $port"
            PORT_FOUND=1
            break
        fi
    done
    
    if [ $PORT_FOUND -eq 0 ]; then
        # 如果所有常用端口都被占用，使用随机端口
        RANDOM_PORT=$((29500 + RANDOM % 1000))
        export MASTER_PORT=$RANDOM_PORT
        echo "警告: 常用端口被占用，使用随机端口: $MASTER_PORT"
    fi
else
    echo "使用指定端口: $MASTER_PORT"
fi

# 获取GPU数量（从SLURM或CUDA_VISIBLE_DEVICES）
if [ -n "$SLURM_STEP_GPUS" ]; then
    # 如果使用SLURM，从SLURM_STEP_GPUS获取GPU数量
    NUM_GPUS=$(echo $SLURM_STEP_GPUS | tr ',' '\n' | wc -l)
elif [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    # 如果设置了CUDA_VISIBLE_DEVICES，从中获取GPU数量
    NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
else
    # 默认使用所有可用GPU
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
fi

echo "========================================="
echo "启动分布式DAgger训练"
echo "GPU数量: $NUM_GPUS"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "========================================="

# 设置 NCCL 超时和错误处理
# 增加超时时间到 60 分钟（3600秒），避免进程同步超时
# 注意：PyTorch的默认超时是10分钟（600秒），需要设置足够大的值
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-3600}
export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}

# 设置PyTorch分布式超时（秒），默认是10分钟（600秒），设置为60分钟
export TORCH_DISTRIBUTED_TIMEOUT=${TORCH_DISTRIBUTED_TIMEOUT:-3600}
export TORCH_NCCL_BLOCKING_WAIT=${TORCH_NCCL_BLOCKING_WAIT:-1}
export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-INFO}

# NCCL 调试级别（WARN 减少日志，INFO 用于详细调试，DEBUG 用于详细排查）
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}

# 启用NCCL跟踪缓冲区以获取更详细的错误信息
export TORCH_NCCL_TRACE_BUFFER_SIZE=${TORCH_NCCL_TRACE_BUFFER_SIZE:-1048576}

# 如果 InfiniBand 有问题，可以禁用（单机多卡通常不需要IB）
# export NCCL_IB_DISABLE=1

# 如果网络接口有问题，可以指定
# export NCCL_SOCKET_IFNAME=eth0

# 单机多卡训练，使用P2P通信
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}
export NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-0}

# 设置NCCL通信后端（单机多卡推荐使用nccl）
export NCCL_BACKEND=${NCCL_BACKEND:-nccl}

echo "NCCL 配置:"
echo "  NCCL_TIMEOUT=$NCCL_TIMEOUT (秒)"
echo "  TORCH_DISTRIBUTED_TIMEOUT=$TORCH_DISTRIBUTED_TIMEOUT (秒)"
echo "  NCCL_ASYNC_ERROR_HANDLING=$NCCL_ASYNC_ERROR_HANDLING"
echo "  NCCL_DEBUG=$NCCL_DEBUG"
echo "  TORCH_NCCL_BLOCKING_WAIT=$TORCH_NCCL_BLOCKING_WAIT"
echo "  TORCH_DISTRIBUTED_DEBUG=$TORCH_DISTRIBUTED_DEBUG"
echo "  TORCH_NCCL_TRACE_BUFFER_SIZE=$TORCH_NCCL_TRACE_BUFFER_SIZE"

# 使用torchrun启动分布式训练
# torchrun会自动设置RANK, WORLD_SIZE, LOCAL_RANK等环境变量
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    -m habitat-baselines.habitat_baselines.run \
    --config-name=dynamic_vlnce/dynamic_vlnce_hm3d_dagger_train \
    habitat_baselines.il.distributed.enabled=True

# 检查是否真的成功（torchrun会在失败时返回非零退出码）
if [ $? -eq 0 ]; then
    echo "训练完成"
else
    echo "训练失败，请查看上面的错误信息"
    exit 1
fi


