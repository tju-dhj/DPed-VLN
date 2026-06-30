#!/bin/bash
# 启动Qwen模型的vLLM API服务

export MODEL_PATH="/share/home/u19666033/dhj/models/Qwen3.6-27B"
export PORT=8000

python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --served-model-name qwen \
    --port $PORT \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 2048
