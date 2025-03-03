WORKER_CONFIGS = {
    # 第一台机器（206，有2张P100）
    "worker1": {
        "address": "10.21.22.206:29500",
        "gpus": ["cuda:0", "cuda:1"],
        "cpu_threads": 10
    },
    # 第二台机器（207，有2张P100）
    "worker2": {
        "address": "10.21.22.207:29500",
        "gpus": ["cuda:0", "cuda:1"],
        "cpu_threads": 10
    }
}

# 模型分片策略 - 手动分配17个数据组
MODEL_SHARD_STRATEGY = {
    "worker1": {  # 206机器 - 前9个数据组
        "layers": [
            "model.embed_tokens",
            "model.layers.0",
            "model.layers.1",
            "model.layers.2",
            "model.layers.3",
            "model.layers.4",
            "model.layers.5",
            "model.layers.6",
            "model.layers.7"
        ],
        "memory": {
            0: "12GiB",  # 第一张P100
            1: "12GiB",  # 第二张P100
            'cpu': "138GiB"
        }
    },
    "worker2": {  # 207机器 - 后8个数据组
        "layers": [
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
    "timeout": 3600,  # 通信超时时间（秒）
    "heartbeat_interval": 30,  # 心跳间隔（秒）
    "retry_attempts": 3  # 重试次数
} 