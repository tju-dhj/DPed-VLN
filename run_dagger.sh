#!/bin/bash
# DAgger 训练启动脚本
# 使用 run_training.py 入口点确保正确的 habitat-lab 路径

cd /share/home/u14004/dhj/Falcon-main

# 打印路径信息以便调试
echo "=== Environment Info ==="
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-0,1}"
echo "Working directory: $(pwd)"
echo "========================"

# 运行训练命令
# 使用 torchrun 进行分布式训练，通过 run_training.py 入口点
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1} torchrun --nproc_per_node=${NPROC:-2} \
    /share/home/u14004/dhj/Falcon-main/run_training.py \
    --config-name=dynamic_vlnce/dynamic_vlnce_hm3d_initial_il_train_v2.yaml \
    habitat_baselines.il.distributed.enabled=True \
    "$@"

