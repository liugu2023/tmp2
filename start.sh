#!/bin/bash

# 设置路径
PYTHONPATH=/archive/liugu/ds/ktransformers
MODEL_PATH=/archive/liugu/ds/DeepSeek-R1-Distill-Llama-70B/
GGUF_PATH=/archive/liugu/ds/DeepSeek-R1-Distill-Llama-70B-GGUF/

# 获取主机名和IP地址
HOSTNAME=$(hostname)
SCHEDULER_IP="10.21.22.100"  # 保留IP地址用于网络连接
COMPUTE_IP="10.21.22.206"    # 保留IP地址用于网络连接

# 检查是哪个节点 (使用主机名)
is_scheduler_node() {
    [[ "$HOSTNAME" == *"login"* ]] && return 0 || return 1
}

is_compute_node() {
    [[ "$HOSTNAME" == *"compute06"* ]] && return 0 || return 1
}

# 杀死已有进程函数
kill_existing_processes() {
    echo "检测并杀死已存在的进程..."
    
    # 杀死嵌入服务器进程 (端口 29700)
    EMBEDDING_PID=$(netstat -tunlp 2>/dev/null | grep ":29700" | awk '{print $7}' | cut -d'/' -f1)
    if [ ! -z "$EMBEDDING_PID" ]; then
        echo "正在杀死嵌入服务器进程 (PID: $EMBEDDING_PID)..."
        kill -9 $EMBEDDING_PID
    fi
    
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
    
    # 杀死调度器进程 (端口 29400)
    SCHEDULER_PID=$(netstat -tunlp 2>/dev/null | grep ":29400" | awk '{print $7}' | cut -d'/' -f1)
    if [ ! -z "$SCHEDULER_PID" ]; then
        echo "正在杀死调度器进程 (PID: $SCHEDULER_PID)..."
        kill -9 $SCHEDULER_PID
    fi
    
    # 杀死所有 python 含有 ktransformers 的进程
    echo "杀死所有相关 Python 进程..."
    ps aux | grep "python.*ktransformers" | grep -v grep | awk '{print $2}' | xargs -r kill -9
    
    # 等待进程完全终止
    sleep 2
    echo "所有旧进程已终止"
}

# 启动计算服务器上的服务
start_compute_services() {
    echo "在计算服务器 (compute06) 上启动服务..."
    
    # 启动嵌入服务器 (使用GPU 0)
    echo "启动嵌入服务器..."
    PYTHONPATH=$PYTHONPATH python -m ktransformers.embedding_server --server_address ${COMPUTE_IP}:29700 --model_path $MODEL_PATH --skip_tokenizer true --cuda_visible_devices 0 > embedding_server.log 2>&1 &
    
    # 等待嵌入服务器启动
    sleep 3
    
    # 启动 KV Cache 服务器 (使用GPU 1)
    echo "启动 KV Cache 服务器..."
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$PYTHONPATH python -m ktransformers.kvcache_server --server_address ${COMPUTE_IP}:29600 --num_workers 8 > kvcache_server.log 2>&1 &
    
    # 等待 KV Cache 服务器启动
    sleep 3
    
    # 启动 Worker 进程 (workers分布在两块GPU上)
    echo "启动 Worker 进程..."
    PYTHONPATH=$PYTHONPATH python -m ktransformers.worker --base_port 29500 --num_workers 8 --kvcache_address ${COMPUTE_IP}:29600 --embedding_address ${COMPUTE_IP}:29700 > worker.log 2>&1 &
    
    echo "所有服务已启动"
    echo "KV Cache 服务器日志: kvcache_server.log"
    echo "嵌入服务器日志: embedding_server.log"
    echo "Worker 进程日志: worker.log"
}

# 启动调度节点上的服务
start_scheduler_services() {
    echo "在调度节点 (login) 上启动服务..."
    
    # 启动调度器
    echo "启动任务调度器..."
    PYTHONPATH=$PYTHONPATH python -m ktransformers.scheduler \
        --server_address ${SCHEDULER_IP}:29400 \
        --compute_address ${COMPUTE_IP}:29500 \
        --model_path $MODEL_PATH > scheduler.log 2>&1 &
    
    echo "调度服务已启动"
    echo "调度器日志: scheduler.log"
}

# 启动客户端（在任何节点上）
start_client() {
    echo "在客户端节点 (${HOSTNAME}) 上启动客户端..."
    
    PYTHONPATH=$PYTHONPATH python -m ktransformers.local_chat \
        --model_path $MODEL_PATH \
        --gguf_path $GGUF_PATH \
        --scheduler_address "${SCHEDULER_IP}:29400" \
        --max_new_tokens 50
    
    echo "客户端已退出"
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

# 显示帮助信息
show_help() {
    echo "使用方法: $0 [选项]"
    echo "选项:"
    echo "  --compute   启动计算节点服务 (在 compute06 上运行)"
    echo "  --scheduler 启动调度节点服务 (在 login 上运行)"
    echo "  --client    启动客户端 (可在任何内网节点上运行)"
    echo "  --help      显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 --compute   # 在计算节点上启动KV缓存和Worker服务"
    echo "  $0 --scheduler # 在调度节点上启动调度服务"
    echo "  $0 --client    # 在任何节点上启动客户端"
}

# 默认参数处理
if [ $# -eq 0 ]; then
    # 根据主机名自动检测模式
    if is_compute_node; then
        echo "检测到计算节点 (compute06)..."
        kill_existing_processes
        check_gpu_status
        start_compute_services
        
        # 显示各个进程的状态
        echo "服务状态:"
        ps aux | grep "python.*ktransformers" | grep -v grep
        
        echo "端口监听状态:"
        netstat -tunlp 2>/dev/null | grep -E ":(29500|29501|29502|29503|29504|29505|29506|29507|29600|29700)"
    elif is_scheduler_node; then
        echo "检测到调度节点 (login)..."
        kill_existing_processes
        start_scheduler_services
        
        # 显示进程状态
        echo "服务状态:"
        ps aux | grep "python.*ktransformers" | grep -v grep
        
        echo "端口监听状态:"
        netstat -tunlp 2>/dev/null | grep -E ":(29400)"
    else
        # 默认作为客户端
        echo "未检测到特定节点，以客户端模式运行..."
        start_client
    fi
else
    # 处理命令行参数
    case "$1" in
        --compute)
            if ! is_compute_node; then
                echo "警告：当前节点不是计算节点 (compute06)，服务可能无法正确运行"
                read -p "是否继续？ (y/n): " confirm
                [[ $confirm != "y" ]] && exit 1
            fi
            kill_existing_processes
            check_gpu_status
            start_compute_services
            ;;
        --scheduler)
            if ! is_scheduler_node; then
                echo "警告：当前节点不是调度节点 (login)，服务可能无法正确运行"
                read -p "是否继续？ (y/n): " confirm
                [[ $confirm != "y" ]] && exit 1
            fi
            kill_existing_processes
            start_scheduler_services
            ;;
        --client)
            start_client
            ;;
        --help)
            show_help
            ;;
        *)
            echo "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
fi 