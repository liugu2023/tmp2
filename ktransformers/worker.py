import torch
import zmq
import pickle
from typing import Dict, Any, List
import logging
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, GenerationConfig, TextStreamer
import time
import os
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model_and_tokenizer(model_path: str, gguf_path: str):
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    logger.info("Loading model config...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    torch.set_default_dtype(config.torch_dtype)
    
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
        device_map="auto"
    )
    model.eval()
    
    return model, tokenizer

class ModelWorker:
    def __init__(self, address: str):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        host, port = address.split(':')
        self.socket.bind(f"tcp://0.0.0.0:{port}")
        self.socket.setsockopt(zmq.RCVHWM, 0)
        self.socket.setsockopt(zmq.SNDHWM, 0)
        
        self.model = None
        self.tokenizer = None
        self.num_processes = 12  # 使用12个进程
        self.process_pool = None
        self.generation_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue()
        logger.info(f"Worker started at {address}, listening on all interfaces")

    def load_model(self, model_path: str, gguf_path: str):
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        logger.info("Loading model config...")
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        
        # 更新配置
        config.attn_implementation = "flash_attention_2"  # 新的 Flash Attention 2 配置
        torch.set_default_dtype(config.torch_dtype)
        
        # 初始化分布式环境
        if not torch.distributed.is_initialized():
            logger.info("Initializing distributed environment...")
            torch.distributed.init_process_group(
                backend='nccl',
                init_method='tcp://localhost:29501',
                world_size=1,
                rank=0
            )
        
        logger.info("Loading model...")
        # 设置双 GPU 的内存分配和并行配置
        max_memory = {
            0: "12GiB",    # 第一张 GPU 使用 12GB
            1: "12GiB",    # 第二张 GPU 使用 12GB
            'cpu': "138GiB" # 剩余放在 CPU 内存
        }
        
        # 设置 CPU 线程数为物理核心数
        num_physical_cores = 6
        num_threads = 12  # 总线程数
        
        # 设置 PyTorch 线程数
        torch.set_num_threads(num_threads)  # 使用所有逻辑核心
        
        # 设置 OpenMP 线程数
        os.environ["OMP_NUM_THREADS"] = str(num_physical_cores)  # OpenMP 使用物理核心数
        os.environ["MKL_NUM_THREADS"] = str(num_physical_cores)  # MKL 使用物理核心数
        
        logger.info("CPU threading configuration:")
        logger.info(f"- Physical cores: {num_physical_cores}")
        logger.info(f"- Total threads: {num_threads}")
        logger.info(f"- PyTorch threads: {torch.get_num_threads()}")
        logger.info(f"- OpenMP threads: {os.environ.get('OMP_NUM_THREADS')}")
        logger.info(f"- MKL threads: {os.environ.get('MKL_NUM_THREADS')}")
        
        # 添加并行配置
        model_kwargs = {
            "device_map": None,  # 不使用自动设备映射
            "max_memory": max_memory,
            "low_cpu_mem_usage": True,
        }
        
        # 设置并行环境
        if torch.cuda.is_available():
            # 启用 CUDA 优化
            torch.backends.cudnn.benchmark = True  # 启用 cuDNN 自动调优
            torch.backends.cudnn.enabled = True
            torch.backends.cuda.matmul.allow_tf32 = True  # 允许使用 TF32
            torch.backends.cudnn.allow_tf32 = True
        
        logger.info("Applying optimization configurations...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            **model_kwargs
        )
        
        # 初始化时将模型放在 CPU 上
        self.model = self.model.cpu()
        self.model.eval()
        
        # 应用模型优化
        if hasattr(self.model, "config"):
            self.model.config.use_cache = True  # 启用 KV 缓存
            if hasattr(self.model.config, "pretraining_tp"):
                self.model.config.pretraining_tp = 1  # 设置张量并行度
        
        # 输出优化配置信息
        logger.info("Model optimization settings:")
        logger.info("- Using FP16 precision")
        logger.info("- Flash Attention 2 enabled")
        logger.info("- CPU threads:")
        logger.info(f"- Physical cores: {num_physical_cores}")
        logger.info(f"- Total threads: {num_threads}")
        logger.info(f"- PyTorch threads: {torch.get_num_threads()}")
        logger.info(f"- OpenMP threads: {os.environ.get('OMP_NUM_THREADS')}")
        logger.info(f"- MKL threads: {os.environ.get('MKL_NUM_THREADS')}")
        logger.info("- Device mapping:")
        for name, device in self.model.hf_device_map.items():
            logger.info(f"  {name}: {device}")
        
        logger.info("Model loaded successfully")

    # 修改 CustomStreamer 类定义
    class CustomStreamer(TextStreamer):
        def __init__(self, tokenizer, on_text_chunk=None, **kwargs):
            super().__init__(tokenizer, **kwargs)
            self.on_text_chunk = on_text_chunk

        def on_finalized_text(self, text: str, stream_end: bool = False):
            if self.on_text_chunk:
                self.on_text_chunk(text, stream_end)

    def generate_single(self, input_data: Dict[str, Any], process_id: int) -> Dict[str, Any]:
        try:
            logger.info(f"Process {process_id}: Starting generation...")
            
            # 设置当前进程使用的 GPU
            gpu_id = process_id % 2  # 在两个 GPU 之间分配
            device = f'cuda:{gpu_id}'
            torch.cuda.set_device(device)
            
            # 将模型移动到当前进程的 GPU
            model = self.model.to(device)
            
            # 确保模型在正确的设备上
            for param in model.parameters():
                if param.device != torch.device(device):
                    param.data = param.data.to(device)
            
            # 准备输入数据
            input_ids = torch.tensor(input_data['input_ids'], device=device)
            attention_mask = torch.tensor(input_data['attention_mask'], device=device)
            
            with torch.no_grad(), torch.amp.autocast('cuda'):
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=input_data.get('max_new_tokens', 512),
                    temperature=input_data.get('temperature', 0.6),
                    top_p=input_data.get('top_p', 0.9),
                    do_sample=input_data.get('do_sample', True),
                    streamer=None,  # 并行模式下暂时不使用 streamer
                    use_cache=True,
                    num_beams=1,
                    synced_gpus=False,
                    repetition_penalty=1.0,
                    length_penalty=1.0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # 清理 GPU 内存
            del model
            torch.cuda.empty_cache()
            
            return {
                'process_id': process_id,
                'status': 'success',
                'output_ids': outputs.cpu().numpy().tolist()
            }
        except Exception as e:
            logger.error(f"Process {process_id} error: {str(e)}")
            return {
                'process_id': process_id,
                'status': 'error',
                'error': str(e)
            }

    def generate_parallel(self, input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """并行生成多个结果"""
        start_time = time.time()
        results = []
        
        # 创建线程池
        with ThreadPoolExecutor(max_workers=self.num_processes) as executor:
            # 提交所有任务
            future_to_id = {
                executor.submit(self.generate_single, input_data, i): i 
                for i in range(self.num_processes)
            }
            
            # 收集结果
            for future in as_completed(future_to_id):
                process_id = future_to_id[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Process {process_id} completed")
                except Exception as e:
                    logger.error(f"Process {process_id} failed: {str(e)}")
        
        end_time = time.time()
        total_time = end_time - start_time
        logger.info(f"All processes completed in {total_time:.2f} seconds")
        
        return results

    def generate(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            logger.info("Starting parallel generation...")
            results = self.generate_parallel(input_data)
            
            # 统计成功的生成结果
            successful_results = [r for r in results if r['status'] == 'success']
            if not successful_results:
                raise Exception("No successful generations")
            
            # 计算统计信息
            total_tokens = sum(len(r['output_ids'][0]) for r in successful_results)
            avg_tokens = total_tokens / len(successful_results)
            
            logger.info(f"Parallel generation completed:")
            logger.info(f"- Successful generations: {len(successful_results)}/{self.num_processes}")
            logger.info(f"- Average tokens per generation: {avg_tokens:.2f}")
            
            # 返回所有结果
            return {
                'status': 'success',
                'results': results
            }
            
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }

    def run(self):
        logger.info("Worker ready to receive requests")
        try:
            while True:
                try:
                    message = self.socket.recv_pyobj()
                    command = message.get('command')
                    
                    if command == 'load_model':
                        self.load_model(message['model_path'], message['gguf_path'])
                        self.socket.send_pyobj({'status': 'success'})
                    
                    elif command == 'generate':
                        result = self.generate(message['data'])
                        self.socket.send_pyobj(result)
                    
                    elif command == 'shutdown':
                        logger.info("Received shutdown command")
                        self.socket.send_pyobj({'status': 'shutdown'})
                        # 清理资源
                        if hasattr(self, 'model'):
                            del self.model
                        if hasattr(self, 'tokenizer'):
                            del self.tokenizer
                        torch.cuda.empty_cache()
                        logger.info("Resources cleaned up")
                        break
                    
                except Exception as e:
                    logger.error(f"Error processing message: {str(e)}")
                    self.socket.send_pyobj({'status': 'error', 'error': str(e)})
        finally:
            # 确保资源被清理
            logger.info("Cleaning up worker resources...")
            self.socket.close()
            self.context.term()
            logger.info("Worker shutdown complete")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker_address', type=str, required=True)
    args = parser.parse_args()
    
    worker = ModelWorker(args.worker_address)
    worker.run()

if __name__ == '__main__':
    main() 