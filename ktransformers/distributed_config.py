WORKER_CONFIGS = {
    # 第一台机器（有1张P100）
    "worker1": {
        "address": "10.21.22.206:29500",
        "gpus": ["cuda:0"],  # 本地 P100
        "cpu_threads": 10
    },
    # 第二台机器（有2张P100）
    "worker2": {
        "address": "10.21.22.xxx:29500",  # 替换为实际IP
        "gpus": ["cuda:0", "cuda:1"],  # 两张 P100
        "cpu_threads": 10
    }
}

# 模型分片策略
MODEL_SHARD_STRATEGY = {
    "worker1": {
        "layers": "0:23",  # 分配前24层
        "memory": {
            0: "12GiB",
            'cpu': "138GiB"
        }
    },
    "worker2": {
        "layers": "24:47",  # 分配后24层
        "memory": {
            0: "12GiB",
            1: "12GiB",
            'cpu': "138GiB"
        }
    }
} 