from .resource_manager import ResourceManager
from .task_scheduler import DistributedTaskScheduler
from .worker import DistributedWorker
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--gguf_path", type=str, required=True)
    args = parser.parse_args()

    # 初始化资源管理器
    resource_manager = ResourceManager()
    resource_manager.initialize_default_resources()
    
    # 创建工作节点
    workers = []
    
    # CPU worker
    cpu_worker = DistributedWorker(
        "10.21.22.100", 
        "cpu",
        model_path=args.model_path,
        gguf_path=args.gguf_path
    )
    workers.append(cpu_worker)
    
    # GPU workers
    for gpu_id in range(2):
        gpu_worker = DistributedWorker(
            "10.21.22.206",
            "gpu",
            device_id=gpu_id,
            model_path=args.model_path,
            gguf_path=args.gguf_path
        )
        workers.append(gpu_worker)
    
    # 初始化所有worker
    for worker in workers:
        worker.initialize()
    
    # 简单的负载均衡：轮询分配请求
    current_worker = 0
    
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit"]:
                break
                
            # 轮询选择worker
            worker = workers[current_worker]
            current_worker = (current_worker + 1) % len(workers)
            
            # 生成响应
            response = worker.generate_response(user_input)
            print(f"Assistant: {response}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            continue

if __name__ == "__main__":
    main() 