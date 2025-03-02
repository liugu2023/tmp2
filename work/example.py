from .resource_manager import ResourceManager
from .task_scheduler import DistributedTaskScheduler
from .worker import DistributedWorker

def main():
    # 初始化资源管理器
    resource_manager = ResourceManager()
    resource_manager.initialize_default_resources()
    
    # 初始化任务调度器
    scheduler = DistributedTaskScheduler(resource_manager)
    
    # 创建工作节点
    cpu_worker = DistributedWorker("10.21.22.100", "cpu")
    gpu_worker1 = DistributedWorker("10.21.22.206", "gpu", 0)
    gpu_worker2 = DistributedWorker("10.21.22.206", "gpu", 1)
    
    # 示例任务
    def example_task(x):
        return x * 2
    
    # 在不同设备上执行任务
    result_cpu = cpu_worker.execute_task(example_task, 10)
    result_gpu1 = gpu_worker1.execute_task(example_task, torch.tensor([1, 2, 3]))
    result_gpu2 = gpu_worker2.execute_task(example_task, torch.tensor([4, 5, 6]))

if __name__ == "__main__":
    main() 