#!/bin/bash

# 设置环境变量
export PYTHONPATH=/archive/liugu/ds/ktransformers
export MODEL_PATH=/archive/liugu/ds/DeepSeek-R1-Distill-Llama-70B
export GGUF_PATH=/archive/liugu/ds/DeepSeek-R1-Distill-Llama-70B-GGUF

# 终止使用特定端口的进程
kill_port_process() {
    port=$1
    pid=$(lsof -t -i:$port 2>/dev/null)
    if [ -n "$pid" ]; then
        echo "正在终止使用端口 $port 的进程 (PID: $pid)..."
        kill -9 $pid
        sleep 1
    fi
}

# 在206上运行的命令
if [ "$(hostname)" = "compute06" ]; then
    echo "Starting services on compute06 (206)..."
    
    # 终止占用端口的进程
    echo "检查并终止占用端口的进程..."
    kill_port_process 29600  # KV Cache 服务器端口
    for port in $(seq 29500 29507); do
        kill_port_process $port  # Worker进程端口
    done
    
    # 启动 KV Cache 服务器，并传入模型路径
    echo "Starting KV Cache Server with model loader..."
    python -m ktransformers.kvcache_server \
        --port 29600 \
        --model_path $MODEL_PATH \
        --gguf_path $GGUF_PATH &
    sleep 5  # 等待KV Cache服务器和模型加载器启动
    
    # 启动worker进程，它们将从KV Cache服务器获取模型参数
    echo "Starting Worker processes..."
    for i in $(seq 0 7); do
        worker_port=$((29500 + i))
        echo "Starting Worker $i on port $worker_port..."
        python -m ktransformers.worker \
            --worker_address=10.21.22.206:$worker_port \
            --worker_id=$i \
            --num_workers=8 &
    done

# 在login上运行的命令
elif [ "$(hostname)" = "login" ]; then
    echo "Starting services on login (100)..."
    
    # 终止占用端口的进程
    echo "检查并终止占用端口的进程..."
    kill_port_process 29400  # 中央服务器端口
    
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