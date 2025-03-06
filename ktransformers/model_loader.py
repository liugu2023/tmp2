import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelLoader:
    """负责加载模型并提供参数查询服务"""
    
    def __init__(self, model_path, gguf_path):
        self.model_path = model_path
        self.gguf_path = gguf_path
        self.config = None
        self.tokenizer = None
        self.loaded_layers = {}
        
    def load_model(self):
        """加载模型配置和tokenizer"""
        logger.info(f"加载模型配置和tokenizer: {self.model_path}")
        self.config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        
        # 加载嵌入层和规范化层
        logger.info("加载嵌入层和lm_head")
        self._load_shared_layers()
        
        logger.info("模型加载器初始化完成")
        return True
        
    def _load_shared_layers(self):
        """加载所有worker共享的层"""
        import safetensors.torch
        
        # 尝试从safetensors加载
        for filename in os.listdir(self.model_path):
            if filename.endswith('.safetensors'):
                try:
                    weights_path = os.path.join(self.model_path, filename)
                    logger.info(f"从 {weights_path} 加载权重")
                    weights = safetensors.torch.load_file(weights_path)
                    
                    # 加载嵌入层
                    if 'model.embed_tokens.weight' in weights:
                        logger.info("加载嵌入层")
                        self.loaded_layers['embeddings'] = weights['model.embed_tokens.weight']
                        
                    # 加载LM头部
                    if 'lm_head.weight' in weights:
                        logger.info("加载LM头部")
                        self.loaded_layers['lm_head'] = weights['lm_head.weight']
                    
                    # 加载规范化层
                    if 'model.norm.weight' in weights:
                        logger.info("加载规范化层")
                        self.loaded_layers['norm'] = weights['model.norm.weight']
                        
                    break
                except Exception as e:
                    logger.error(f"加载权重文件出错: {str(e)}")
                    continue
    
    def load_layer(self, layer_index):
        """按需加载特定层"""
        if f'layer_{layer_index}' in self.loaded_layers:
            return True
            
        logger.info(f"按需加载层 {layer_index}")
        try:
            # 实现按需加载特定层的逻辑
            # 这里可以使用更细粒度的方法加载单层
            # ...
            
            return True
        except Exception as e:
            logger.error(f"加载层 {layer_index} 时出错: {str(e)}")
            return False
    
    def get_config(self):
        """获取模型配置"""
        return self.config
        
    def get_tokenizer(self):
        """获取tokenizer"""
        return self.tokenizer
        
    def get_layer_params(self, layer_index):
        """获取特定层的参数"""
        key = f'layer_{layer_index}'
        if key not in self.loaded_layers:
            success = self.load_layer(layer_index)
            if not success:
                return None
                
        return self.loaded_layers[key]
        
    def get_shared_params(self):
        """获取共享参数(嵌入层、LM头等)"""
        return {
            'embeddings': self.loaded_layers.get('embeddings'),
            'lm_head': self.loaded_layers.get('lm_head'),
            'norm': self.loaded_layers.get('norm')
        } 