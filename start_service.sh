#!/bin/bash

# 设置环境变量
export PYTHONPATH=/archive/liugu/ds/ktransformers
export MODEL_PATH=/archive/liugu/ds/DeepSeek-R1-Distill-Llama-70B
export GGUF_PATH=/archive/liugu/ds/DeepSeek-R1-Distill-Llama-70B-GGUF

# 在206上运行的命令
if [ "$(hostname)" = "compute06" ]; then
    echo "Starting services on compute06 (206)..."
    
    # 启动 KV Cache 服务器
    echo "Starting KV Cache Server..."
    python -m ktransformers.kvcache_server --port 29600 &
    sleep 2  # 等待KV Cache服务器启动
    
    # 启动模型分片服务
    echo "Starting Model Sharder..."
    python -m ktransformers.model_sharder \
        --model_path $MODEL_PATH \
        --gguf_path $GGUF_PATH \
        --base_port 29500 \
        --num_workers 8

# 在login上运行的命令
elif [ "$(hostname)" = "login" ]; then
    echo "Starting services on login (100)..."
    
    # 启动中央服务器
    echo "Starting Central Server..."
    python -m ktransformers.central_server \
        --port 29400 \
        --num_workers 8 \
        --worker_base_port 29500 \
        --kvcache_port 29600 \
        --worker_host 10.21.22.206
    
else
    echo "Unknown host. This script should be run on either compute06 or login."
    exit 1
fi 