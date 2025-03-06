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
        
        # 加载配置并转换为字典
        config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        # 将配置转换为可序列化的字典
        self.config = {
            'architectures': config.architectures,
            'model_type': config.model_type,
            'num_hidden_layers': config.num_hidden_layers,
            'hidden_size': config.hidden_size,
            'intermediate_size': getattr(config, 'intermediate_size', 4 * config.hidden_size),
            'num_attention_heads': config.num_attention_heads,
            'max_position_embeddings': config.max_position_embeddings,
            'torch_dtype': str(config.torch_dtype),
            'pad_token_id': config.pad_token_id,
            'bos_token_id': config.bos_token_id,
            'eos_token_id': config.eos_token_id,
            'vocab_size': config.vocab_size
        }
        
        # 加载tokenizer
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
        key = f'layer_{layer_index}'
        if key in self.loaded_layers:
            return True
        
        logger.info(f"按需加载层 {layer_index}")
        try:
            # 实现按需加载特定层的逻辑
            import safetensors.torch
            
            layer_params = {}
            for filename in os.listdir(self.model_path):
                if filename.endswith('.safetensors'):
                    weights_path = os.path.join(self.model_path, filename)
                    weights = safetensors.torch.load_file(weights_path)
                    
                    # 提取该层的所有参数
                    layer_prefix = f'model.layers.{layer_index}.'
                    for key, value in weights.items():
                        if key.startswith(layer_prefix):
                            param_name = key[len(layer_prefix):]  # 去掉前缀
                            layer_params[param_name] = value
                    
                    if layer_params:  # 如果找到了这一层的参数
                        self.loaded_layers[key] = layer_params
                        logger.info(f"成功加载层 {layer_index} 的 {len(layer_params)} 个参数")
                        return True
            
            if not layer_params:
                logger.warning(f"在权重文件中未找到层 {layer_index} 的参数")
                return False
            
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