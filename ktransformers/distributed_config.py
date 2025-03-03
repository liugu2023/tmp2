WORKER_CONFIGS = {
    "worker1": {  # 206机器
        "address": "10.21.22.206:29500",
        "gpus": ["cuda:0", "cuda:1"],
        "cpu_threads": 10
    }
}

# 模型分片策略 - 在206的两张P100上分配
MODEL_SHARD_STRATEGY = {
    "worker1": {  # 206机器
        "layers": [
            "model.embed_tokens",
            "model.layers.0",
            "model.layers.1",
            "model.layers.2",
            "model.layers.3",
            "model.layers.4",
            "model.layers.5",
            "model.layers.6",
            "model.layers.7",
            "model.layers.8",
            "model.layers.9",
            "model.layers.10",
            "model.layers.11",
            "model.layers.12",
            "model.layers.13",
            "model.layers.14",
            "model.norm"
        ],
        "memory": {
            0: "12GiB",  # 第一张P100
            1: "12GiB",  # 第二张P100
            'cpu': "138GiB"
        }
    }
}

# 通信配置
COMMUNICATION_CONFIG = {
    "timeout": 3600,
    "heartbeat_interval": 30,
    "retry_attempts": 3
} 