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
        self.num_gpu_processes = 2  # GPU 进程数
        self.num_cpu_processes = 8  # CPU 进程数
        self.num_processes = self.num_gpu_processes + self.num_cpu_processes
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
        # 设置双 GPU 的内存分配
        max_memory = {
            0: "12GiB",    # 第一张 GPU 使用 12GB
            1: "12GiB",    # 第二张 GPU 使用 12GB
            'cpu': "138GiB" # 剩余放在 CPU 内存
        }
        
        # 添加并行配置
        model_kwargs = {
            "device_map": "auto",  # 恢复自动设备映射
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
        
        self.model.eval()
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
            if process_id < self.num_gpu_processes:
                # GPU 进程
                logger.info(f"Process {process_id}: Starting generation on GPU {process_id}")
                device = f'cuda:{process_id}'
            else:
                # CPU 进程
                logger.info(f"Process {process_id}: Starting generation on CPU")
                device = 'cpu'
            
            # 准备输入数据
            input_ids = torch.tensor(input_data['input_ids'], device=device)
            attention_mask = torch.tensor(input_data['attention_mask'], device=device)
            
            # 如果是 CPU 进程，使用 float32 以提高数值稳定性
            dtype = torch.float16 if 'cuda' in str(device) else torch.float32
            
            with torch.no_grad():
                # GPU 使用 amp，CPU 不使用
                if 'cuda' in str(device):
                    with torch.amp.autocast('cuda'):
                        outputs = self.model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=input_data.get('max_new_tokens', 512),
                            temperature=input_data.get('temperature', 0.6),
                            top_p=input_data.get('top_p', 0.9),
                            do_sample=input_data.get('do_sample', True),
                            streamer=None,
                            use_cache=True,
                            num_beams=1,
                            synced_gpus=False,
                            repetition_penalty=1.0,
                            length_penalty=1.0,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id
                        )
                else:
                    # CPU 生成使用较小的 batch size 和较少的 tokens
                    outputs = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=min(input_data.get('max_new_tokens', 512), 256),  # 限制 CPU 生成长度
                        temperature=input_data.get('temperature', 0.6),
                        top_p=input_data.get('top_p', 0.9),
                        do_sample=input_data.get('do_sample', True),
                        streamer=None,
                        use_cache=True,
                        num_beams=1,
                        repetition_penalty=1.0,
                        length_penalty=1.0,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
            
            return {
                'process_id': process_id,
                'device': str(device),
                'status': 'success',
                'output_ids': outputs.cpu().numpy().tolist()
            }
        except Exception as e:
            logger.error(f"Process {process_id} error: {str(e)}")
            return {
                'process_id': process_id,
                'device': str(device) if 'device' in locals() else 'unknown',
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
            
            # 分别计算 GPU 和 CPU 的统计信息
            gpu_results = [r for r in successful_results if 'cuda' in r['device']]
            cpu_results = [r for r in successful_results if r['device'] == 'cpu']
            
            # 计算统计信息
            gpu_tokens = sum(len(r['output_ids'][0]) for r in gpu_results) if gpu_results else 0
            cpu_tokens = sum(len(r['output_ids'][0]) for r in cpu_results) if cpu_results else 0
            
            logger.info(f"Parallel generation completed:")
            logger.info(f"- GPU generations: {len(gpu_results)}/{self.num_gpu_processes}")
            logger.info(f"- CPU generations: {len(cpu_results)}/{self.num_cpu_processes}")
            if gpu_results:
                logger.info(f"- Average GPU tokens: {gpu_tokens/len(gpu_results):.2f}")
            if cpu_results:
                logger.info(f"- Average CPU tokens: {cpu_tokens/len(cpu_results):.2f}")
            
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