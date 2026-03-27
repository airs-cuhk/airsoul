import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from airsoul.utils.generator import GeneratorBase
from airsoul.utils.trainer import custom_load_model
from .tools import  Logger, log_progress, log_debug, log_warn, log_fatal


class APIRunner:
    """
    API Runner 将 generator_class 的接口暴露到 Runner 层
    
    使用方式：
        runner = APIRunner(config, use_gpu=True, world_size=1)
        runner.start(model_type, generator_class, extra_info=None)
        
        # 然后可以直接调用 generator 的方法
        runner.setup_topology(agent_num, sensor_num, related_sensor, related_agent, target_temperature)
        runner.reset()
        action = runner.get_action(sensor_obs, previous_actions)
    """
    
    def __init__(self, config, use_gpu=True, world_size=1):
        """
        初始化 API Runner
        
        参数:
            config: 配置对象
            use_gpu: 是否使用 GPU
            world_size: 分布式训练的进程数（目前只支持 world_size=1）
        """
        self.config = config
        self.use_gpu = use_gpu
        self.world_size = world_size
        
        # 模型和 generator
        self.model = None
        self.generator = None
        
        # 设备信息
        self.device = None
        self.device_type = None
        
        # 标记是否已启动
        self._started = False
    
    def start(self, model_type, generator_class, extra_info=None):
        """
        启动 API Runner
        
        参数:
            model_type: 模型类
            generator_class: generator 类
            extra_info: 额外信息
        
        返回:
            generator 实例
        """
        if self.world_size > 1:
            log_warn("APIRunner currently only supports world_size=1 for API mode")
        
        # 初始化设备
        self._init_device()
        
        # 创建模型
        self._create_model(model_type)
        
        # 加载模型权重
        self._load_model()
        
        # 创建 generator
        self._create_generator(generator_class, extra_info)
        
        # 执行 preprocess
        self.generator.preprocess()
        
        self._started = True
        log_debug("API Runner started successfully")
        
        # 返回 generator，方便直接访问
        return self.generator
    
    def _init_device(self):
        """初始化设备"""
        if self.use_gpu:
            self.device = torch.device('cuda:0')
            self.device_type = 'cuda'
            log_debug(f"Using GPU: {self.device}")
        else:
            self.device = torch.device('cpu')
            self.device_type = 'cpu'
            log_debug("Using CPU")
    
    def _create_model(self, model_type):
        """创建模型"""
        self.model = model_type(self.config.model_config, verbose=True)
        self.model = self.model.to(self.device)
        log_debug("Model created and moved to device")

    
    def _load_model(self):
        """加载模型权重"""
        if (self.config.has_attr("load_model_path") and 
            self.config.load_model_path is not None and 
            self.config.load_model_path.lower() != 'none'):
            
            # 黑名单参数
            if self.config.has_attr("load_model_parameter_blacklist"):
                black_list = self.config.load_model_parameter_blacklist
            else:
                black_list = []
            
            # 加载模型
            self.model, _ = custom_load_model(
                self.model, 
                self.config.load_model_path,
                black_list=black_list,
                verbose=True,
                strict_check=False
            )
            log_debug(f"Model loaded from {self.config.load_model_path}")
        else:
            log_warn("No model loaded as 'load_model_path' is not specified")
    
    def _create_generator(self, generator_class, extra_info):
        """创建 generator"""
        self.generator = generator_class(
            run_name=self.config.run_name,
            model=self.model,
            config=self.config.generator_config,
            log_config=self.config.log_config,
            action_dim=self.config.model_config.action_dim,
            rank=0,
            world_size=self.world_size,
            device_type=self.device_type,
            device=self.device,
            main=True,
            extra_info=extra_info
        )
        log_debug("Generator created")

    
    def call_method(self, method_name, *args, **kwargs):
        """
        调用 generator 的任意方法
        
        参数:
            method_name: 方法名
            *args, **kwargs: 方法参数
        
        返回:
            方法返回值
        """
        if not self._started:
            raise RuntimeError("API Runner has not been started. Call start() first.")
        
        if not hasattr(self.generator, method_name):
            raise AttributeError(f"Generator has no method '{method_name}'")
        
        method = getattr(self.generator, method_name)
        return method(*args, **kwargs)
    
    def reset_memory(self):
        self.model.module.reset()
        self.model.eval()
    
    def cleanup(self):
        """清理资源"""
        if self.model is not None:
            del self.model
        if self.generator is not None:
            del self.generator
        log_debug("API Runner cleaned up")

if __name__ == "__main__":
    # 示例：使用 API Runner
    
    from airsoul.utils.tools import Configure
    from airsoul.models import OmniRL_MultiAgent
    from projects.MultiAgentHVAC.hvac_api import HVACAPI
    
    # 1. 加载配置
    config = Configure()
    config.from_yaml(sys.argv[1])

    
    # 2. 创建 API Runner
    runner = APIRunner(config, use_gpu=True, world_size=1)
    
    # 3. 启动 Runner
    generator = runner.start(OmniRL_MultiAgent, HVACAPI)
    
    # 方式1: 直接通过 generator 调用方法
    generator.init(target_temperature, obs_graph, agent_graph)
    obs = env.reset()
    previous_actions = env.get_last_action()
    # TODO 写个 icl only的接口，当环境的动作不由 OmniRL 控制的时候，就调这个接口 icl

    while True:
        action, vocab_seq_batch = generator.get_action(obs, previous_actions)
        obs, reward, done = env(action)

        # env_action 是环境实际发生的动作，不一定非得是OmniRL的动作，也可以是 baseline 的，进行 offline learning
        current_action = env.get_last_action()
        generator.icl(vocab_seq_batch, current_action, previous_actions, done, reward)
        previous_actions = current_action

        if done:
            generator.epoch_end()
            break

    
    # 3. 清空模型 memory
    runner.reset_memory()

    # 4. 完全退出：清理
    runner.cleanup()