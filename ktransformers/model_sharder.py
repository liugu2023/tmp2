import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import logging
import os
import zmq
import time
from typing import Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SharedModelComponents:
    """用于在多个进程间共享模型组件的类"""
    def __init__(self, model_path: str):
        logger.info("Loading shared model components...")
        
        # 确保使用spawn方法创建进程
        mp.set_start_method('spawn', force=True)
        
        # 加载tokenizer (不需要放在共享内存中，每个进程可以快速加载)
        self.tokenizer_path = model_path
        
        # 加载配置
        logger.info("Loading model config...")
        self.config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        
        # 从safetensors或bin文件中加载embedding和lm_head权重
        logger.info("Loading embedding and lm_head weights...")
        import safetensors.torch
        import torch
        
        # 尝试从safetensors加载
        embed_loaded = False
        for filename in os.listdir(model_path):
            if filename.endswith('.safetensors'):
                try:
                    weights = safetensors.torch.load_file(os.path.join(model_path, filename))
                    if 'model.embed_tokens.weight' in weights:
                        # 创建并加载embedding层
                        self.embed_tokens = torch.nn.Embedding(
                            self.config.vocab_size, 
                            self.config.hidden_size
                        )
                        self.embed_tokens.weight.data.copy_(weights['model.embed_tokens.weight'])
                        self.embed_tokens.share_memory()
                        
                        # 创建并加载lm_head层
                        self.lm_head = torch.nn.Linear(
                            self.config.hidden_size, 
                            self.config.vocab_size, 
                            bias=False
                        )
                        if 'lm_head.weight' in weights:
                            self.lm_head.weight.data.copy_(weights['lm_head.weight'])
                        else:
                            # 如果没有单独的lm_head权重，使用tied weights
                            self.lm_head.weight = self.embed_tokens.weight
                        self.lm_head.share_memory()
                        
                        embed_loaded = True
                        break
                except Exception as e:
                    logger.warning(f"Error loading from {filename}: {e}")
                    continue
        
        # 如果从safetensors加载失败，尝试从bin文件加载
        if not embed_loaded:
            try:
                state_dict = torch.load(os.path.join(model_path, 'pytorch_model.bin'))
                if 'model.embed_tokens.weight' in state_dict:
                    # 创建并加载embedding层
                    self.embed_tokens = torch.nn.Embedding(
                        self.config.vocab_size, 
                        self.config.hidden_size
                    )
                    self.embed_tokens.weight.data.copy_(state_dict['model.embed_tokens.weight'])
                    self.embed_tokens.share_memory()
                    
                    # 创建并加载lm_head层
                    self.lm_head = torch.nn.Linear(
                        self.config.hidden_size, 
                        self.config.vocab_size, 
                        bias=False
                    )
                    if 'lm_head.weight' in state_dict:
                        self.lm_head.weight.data.copy_(state_dict['lm_head.weight'])
                    else:
                        # 如果没有单独的lm_head权重，使用tied weights
                        self.lm_head.weight = self.embed_tokens.weight
                    self.lm_head.share_memory()
                    
                    embed_loaded = True
            except Exception as e:
                logger.warning(f"Error loading from pytorch_model.bin: {e}")
        
        if not embed_loaded:
            raise RuntimeError("Failed to load embedding weights from model files")
        
        logger.info(f"Created shared embedding layer with dim={self.config.hidden_size}, vocab_size={self.config.vocab_size}")
        logger.info("Shared model components loaded and ready")
        
        # 创建共享字典来存储已加载的层
        self.manager = mp.Manager()
        self.loaded_layers = self.manager.dict()
    
    def get_components(self):
        """获取所有共享组件"""
        return {
            'tokenizer_path': self.tokenizer_path,
            'config': self.config,
            'embed_tokens': self.embed_tokens,
            'lm_head': self.lm_head,
            'loaded_layers': self.loaded_layers
        }

def start_master(model_path, gguf_path, num_workers, base_port):
    """启动主进程，加载共享组件并控制worker进程"""
    # 加载共享组件
    shared_components = SharedModelComponents(model_path)
    components = shared_components.get_components()
    
    # 启动worker进程
    processes = []
    for worker_id in range(num_workers):
        p = mp.Process(
            target=start_worker, 
            args=(worker_id, num_workers, base_port, model_path, gguf_path, components)
        )
        p.start()
        processes.append(p)
    
    logger.info(f"Started {num_workers} worker processes")
    
    # 等待所有进程结束
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, terminating workers...")
        for p in processes:
            p.terminate()
        
        # 清理共享资源
        logger.info("Cleaning up shared resources...")

def start_worker(worker_id, num_workers, base_port, model_path, gguf_path, components):
    """使用共享组件启动worker进程"""
    from ktransformers.worker import ModelWorker
    
    # 计算端口
    port = base_port + worker_id
    address = f"10.21.22.206:{port}"
    
    # 创建worker并传入共享组件
    worker = ModelWorker(address, worker_id, num_workers, shared_components=components)
    
    # 加载模型（使用共享组件）
    worker.load_model(model_path, gguf_path)
    
    # 运行worker
    worker.run()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--gguf_path', type=str, required=True)
    parser.add_argument('--base_port', type=int, default=29500)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()
    
    start_master(args.model_path, args.gguf_path, args.num_workers, args.base_port)

if __name__ == '__main__':
    main() 