#!/bin/bash

# 设置路径
PYTHONPATH=/archive/liugu/ds/ktransformers
MODEL_PATH=/archive/liugu/ds/DeepSeek-R1-Distill-Llama-70B/
GGUF_PATH=/archive/liugu/ds/DeepSeek-R1-Distill-Llama-70B-GGUF/

# 获取主机名
HOSTNAME=$(hostname)

# 杀死已有进程函数
kill_existing_processes() {
    echo "检测并杀死已存在的进程..."
    
    # 杀死 KV Cache Server 进程 (端口 29600)
    KVCACHE_PID=$(netstat -tunlp 2>/dev/null | grep ":29600" | awk '{print $7}' | cut -d'/' -f1)
    if [ ! -z "$KVCACHE_PID" ]; then
        echo "正在杀死 KV Cache Server 进程 (PID: $KVCACHE_PID)..."
        kill -9 $KVCACHE_PID
    fi
    
    # 杀死 Worker 进程 (端口 29500-29507)
    for PORT in $(seq 29500 29507); do
        WORKER_PID=$(netstat -tunlp 2>/dev/null | grep ":$PORT" | awk '{print $7}' | cut -d'/' -f1)
        if [ ! -z "$WORKER_PID" ]; then
            echo "正在杀死 Worker 进程 (端口: $PORT, PID: $WORKER_PID)..."
            kill -9 $WORKER_PID
        fi
    done
    
    # 杀死所有 python 含有 ktransformers 的进程
    echo "杀死所有相关 Python 进程..."
    ps aux | grep "python.*ktransformers" | grep -v grep | awk '{print $2}' | xargs -r kill -9
    
    # 等待进程完全终止
    sleep 2
    echo "所有旧进程已终止"
}

# 启动计算服务器上的服务
start_compute_services() {
    echo "在计算服务器上启动服务..."
    
    # 启动 KV Cache 服务器
    echo "启动 KV Cache 服务器..."
    PYTHONPATH=$PYTHONPATH python -m ktransformers.kvcache_server --server_address 10.21.22.206:29600 --num_workers 8 > kvcache_server.log 2>&1 &
    
    # 等待 KV Cache 服务器启动
    sleep 3
    
    # 启动 Worker 进程
    echo "启动 Worker 进程..."
    PYTHONPATH=$PYTHONPATH python -m ktransformers.worker --base_port 29500 --num_workers 8 --kvcache_address 10.21.22.206:29600 > worker.log 2>&1 &
    
    echo "所有服务已启动"
    echo "KV Cache 服务器日志: kvcache_server.log"
    echo "Worker 进程日志: worker.log"
}

# 启动登录节点上的客户端
start_login_client() {
    echo "在登录节点上启动客户端..."
    
    PYTHONPATH=$PYTHONPATH python -m ktransformers.local_chat \
        --model_path $MODEL_PATH \
        --gguf_path $GGUF_PATH \
        --worker_address "10.21.22.206:29500" \
        --max_new_tokens 50
}

# 检查GPU状态（仅在计算节点上）
check_gpu_status() {
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU 状态:"
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,temperature.gpu --format=csv,noheader
    else
        echo "无法检查 GPU 状态（nvidia-smi 不可用）"
    fi
}

# 根据主机名执行不同操作
if [[ "$HOSTNAME" == *"compute06"* ]]; then
    echo "在计算服务器上运行..."
    kill_existing_processes
    check_gpu_status
    start_compute_services
    
    # 显示各个进程的状态
    echo "服务状态:"
    ps aux | grep "python.*ktransformers" | grep -v grep
    
    echo "端口监听状态:"
    netstat -tunlp 2>/dev/null | grep -E ":(29500|29501|29502|29503|29504|29505|29506|29507|29600)"
    
elif [[ "$HOSTNAME" == *"login"* ]]; then
    echo "在登录节点上运行..."
    start_login_client
else
    echo "未知主机: $HOSTNAME"
    echo "请在 compute06 或 login 节点上运行此脚本"
    exit 1
fi 