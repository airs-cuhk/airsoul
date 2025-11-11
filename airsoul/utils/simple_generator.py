import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from .tools import Logger, log_progress, log_debug, log_warn, log_fatal
from .tools import custom_load_model
from .trainer import Runner

class ModelLoader:
    def __init__(self, config=None, device="cuda", use_gpu=False, world_size=1, **kwargs):
        self.config = config
        self.use_gpu = use_gpu
        self.world_size = world_size
        self.device = torch.device(device if use_gpu and torch.cuda.is_available() else "cpu")
        self.rank = 0

        # Initialize distributed process group for DDP
        os.environ['MASTER_ADDR'] = 'localhost'
        # Arbitrary port
        os.environ['MASTER_PORT'] = '12355'
        if self.use_gpu:
            dist.init_process_group("nccl", rank=self.rank, world_size=self.world_size)
        else:
            dist.init_process_group("gloo", rank=self.rank, world_size=self.world_size)
        log_debug(f"ModelLoader initialized: device={self.device}, world_size={world_size}")

    def load_model(self, model_type):
        if not hasattr(self.config, 'model_config'):
            log_fatal("Config missing model_config")
            raise ValueError("Config missing model_config")
        if not hasattr(self.config, 'load_model_path') or not self.config.load_model_path:
            log_fatal("Config missing or invalid load_model_path")
            raise ValueError("Config missing or invalid load_model_path")

        try:
            model = model_type(self.config.model_config).to(self.device)
            device_ids = None
            if self.use_gpu and self.device.type == "cuda":
                gpu_index = int(self.device.index)  # Get the GPU index from device (e.g., 1 for cuda:1)
                device_ids = [gpu_index]
            model = DDP(model, device_ids=device_ids, find_unused_parameters=True)
            log_debug("Single-process DDP initialized")

            checkpoint_path = os.path.join(self.config.load_model_path)
            if not os.path.exists(checkpoint_path):
                log_fatal(f"Checkpoint not found: {checkpoint_path}")
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

            black_list = getattr(self.config, 'load_model_parameter_blacklist', [])
            model = custom_load_model(
                model,
                checkpoint_path,
                black_list=black_list,
                verbose=True,
                strict_check=False
            )
            log_debug(f"Model loaded from {checkpoint_path}")

            model.eval()
            return model
        except Exception as e:
            log_fatal(f"Failed to load model: {e}")
            raise
        finally:
            # Clean up process group
            dist.destroy_process_group()
