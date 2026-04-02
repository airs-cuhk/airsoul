import os
import sys
import argparse
import torch
import numpy
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from functools import wraps
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
from airsoul.dataloader.prefetch_dataloader import PrefetchDataLoader
from .tools import Configure, Logger, log_progress, log_debug, log_warn, log_fatal, log_sum_parameters_grad
from .tools import create_folder, check_model_validity, model_path, count_parameters, safety_check, apply_gradient_safely, custom_load_model, custom_save_model, memory_cpy
from .scheduler import noam_scheduler, cosine_function_scheduler
import time
import datetime

# def is_multi_node():
#     return int(os.environ.get("NNODES", "1")) > 1

def is_multi_node():
    """检查是否在多机分布式训练模式下运行"""
    # 最可靠的方法：比较总进程数和本地进程数
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', 1))
    return world_size > local_world_size

def EpochManager(cls):
    @wraps(cls, updated=())
    class WrapperEpochManager(object):
        def __init__(self, **kwargs):
            self.computer = cls(**kwargs)
            for key in kwargs:
                setattr(self, key, kwargs[key])
            
        def get(self, attr, config=None, default=None):
            if(hasattr(self.computer, attr)):
                return getattr(self.computer, attr)
            elif(config is not None):
                if(config.has_attr(attr)):
                    return getattr(self.config, attr)
                else:
                    return default
            else:
                return default

        def init_dataloader(self):
            self.dataloader = self.get('dataloader')
            if(self.dataloader is None):
                DataType = self.get('DataType')
                assert DataType is not None, f"either dataloader or DataType must be specified."
                dataset = DataType(self.config.data_path, 
                                    self.config.seq_len,
                                    self.world_size,
                                    self.config.max_data if self.config.has_attr("max_data") else None,
                                    verbose=self.main)
                if(self.is_training and self.config.has_attr("resume_data_by_step") and self.config.resume_data_by_step):
                    start_index = 0
                    if("steps" in self.training_metainfo):
                        start_index = self.training_metainfo["steps"] * self.config.batch_size * self.world_size
                    dataset.sequential = True
                    dataset.start_index = start_index
                    print(f"Resuming data from step {self.training_metainfo['steps']}, start index: {start_index}")
                print(f"Loading dataset from {self.config.data_path}, file count: {len(dataset)}")
                self.dataloader = PrefetchDataLoader(dataset, batch_size=self.config.batch_size, 
                                            rank=self.rank, world_size=self.world_size)
                self.computer.dataloader = self.dataloader

        def init_logger(self):
            self.logger = self.get('logger')
            if(self.logger is None):
                self.logger_keys = self.get('logger_keys')
                if(self.logger_keys is not None and len(self.computer.logger_keys)!=0):
                    assert type(self.computer.logger_keys) == list, \
                        f"The logger_keys must be a list of string."
                    if(self.is_training):
                        process_name = f"Training-{self.computer.__class__.__name__}"
                        max_iter = len(self.dataloader)
                    else:
                        process_name = f"Evaluation-{self.computer.__class__.__name__}"
                        max_iter = -1
                    log_file = self.get('log_file')
                    if(log_file is None):
                        if(self.is_training):
                            log_file = self.log_config.training_log
                        else:
                            log_file = self.log_config.evaluation_log

                    # Make sure file exist.
                    log_dir = os.path.dirname(log_file)
                    if log_dir and not os.path.exists(log_dir):
                        os.makedirs(log_dir, exist_ok=True)

                    self.logger = Logger(
                            *self.logger_keys,
                            on=self.main, 
                            max_iter=max_iter,
                            use_tensorboard=self.log_config.use_tensorboard,
                            log_file=log_file,
                            prefix=f"{self.run_name}-{process_name}",
                            field=f"{self.log_config.tensorboard_log}/{self.run_name}-{process_name}")
            self.computer.logger = self.logger

        def init_optimizer(self):
            if(self.is_training):
                self.optimizer = self.get('optimizer')
                if(self.optimizer is None):
                    lr = self.get('lr', config=self.config)
                    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
                    self.computer.optimizer = self.optimizer

                # Initialize the learning rate schedulers
                self.lr_scheduler = self.get('lr_scheduler')
                # if(self.lr_scheduler is None):
                #     lr_decay_interval = self.get('lr_decay_interval', config=self.config)
                #     self.lr_scheduler = LambdaLR(self.optimizer, 
                #         lr_lambda=lambda x:noam_scheduler(x, lr_decay_interval))
                #     self.computer.lr_scheduler = self.lr_scheduler
                
                if(self.lr_scheduler is None):
                    # lr_decay_interval = self.get('lr_decay_interval', config=self.config)
                    self.lr_scheduler = LambdaLR(self.optimizer, 
                        lr_lambda=lambda x:cosine_function_scheduler(x, self.config.lr_T_max, self.config.lr_warmup_step, self.config.lr_max, self.config.lr_min))
                    
                    self.computer.lr_scheduler = self.lr_scheduler

                self.lr_scheduler.step(self.get_global_batch_id)

                self.scaler=None
                if(self.config.use_scaler):
                    self.scaler = GradScaler()
                self.computer.scaler = self.scaler
                extra_states = self.get('extra_states')

                if(extra_states):
                    if("optimizer_state" in extra_states):
                        self.optimizer.load_state_dict(extra_states["optimizer_state"])
                    if("lr_scheduler_state" in extra_states):
                        self.lr_scheduler.load_state_dict(extra_states["lr_scheduler_state"])
                    if(self.scaler is not None and "scaler_state" in extra_states):
                        self.scaler.load_state_dict(extra_states["scaler_state"])
        @property
        def get_global_epoch_id(self):
            if("epochs" in self.training_metainfo):
                return self.training_metainfo["epochs"]
            else:
                return 0
        @property
        def get_global_batch_id(self): 
            if("steps" in self.training_metainfo):
                return self.training_metainfo["steps"]
            else:
                return 0

        def _valid_epoch(self):
            if(hasattr(self.computer, 'valid_epoch')):
                return self.computer.valid_epoch(self.get_global_epoch_id)
            return True

        def _epoch_start(self):
            if(not self._valid_epoch()):
                return
            if(hasattr(self.computer, 'epoch_start')):
                self.computer.epoch_start(self.get_global_epoch_id)
        
        def _epoch_end(self):
            if(not self._valid_epoch()):
                return
            if(hasattr(self.computer, 'epoch_end')):
                self.computer.epoch_end(self.get_global_epoch_id, self.get_global_batch_id)

        def _preprocess(self):
            if(hasattr(self.computer, 'preprocess')):
                self.computer.preprocess()
            if("training_metainfo" in self.__dict__):
                if("steps" not in self.training_metainfo):
                    self.training_metainfo["steps"] = 0
                if("epochs" not in self.training_metainfo):
                    self.training_metainfo["epochs"] = 0
            self.init_dataloader()
            self.init_logger()
            self.init_optimizer()

        def _postprocess(self):
            if(hasattr(self.computer, 'postprocess')):
                self.computer.postprocess()

        def _cast_batch_to_dtype(self, batch, dtype):
            if isinstance(batch, torch.Tensor):
                if torch.is_floating_point(batch):
                    return batch.to(dtype)
                return batch
            if isinstance(batch, (list, tuple)):
                return type(batch)(self._cast_batch_to_dtype(x, dtype) for x in batch)
            if isinstance(batch, dict):
                return {k: self._cast_batch_to_dtype(v, dtype) for k, v in batch.items()}
            return batch

        def emergency_save_check(self):
            if("watch_dir" not in self.__dict__ or self.watch_dir is None):
                return False
            if(self.main and os.path.exists(f"{self.watch_dir}/emergency_save")):
                os.remove(f"{self.watch_dir}/emergency_save")
                return True
            return False

        def run(self, device, device_type):
            if(not self._valid_epoch()):
                return

            acc_iter_log = 0

            if(not hasattr(self.computer, 'compute')):
                log_fatal("The computer object must have compute method.")
            if(self.config.has_attr("manual_sync")):
                print(f"manual_sync: {self.config.manual_sync}")
                manual_sync = self.config.manual_sync
            else:
                manual_sync = False
                
            if(hasattr(self, "use_bf16")):
                use_bf16 = self.use_bf16
            elif(self.config.has_attr("use_bf16")):
                use_bf16 = self.config.use_bf16
            else:
                use_bf16 = False
            if(use_bf16 and self.config.has_attr("use_amp") and self.config.use_amp):
                log_warn("use_bf16=True overrides use_amp; autocast will be disabled.")
            data_length = len(self.dataloader)
            print(f"data_length: {data_length}, use_bf16: {use_bf16}")

            if("training_metainfo" in self.__dict__ and self.is_training):
                done = self.training_metainfo["epochs"] > self.config.max_epochs
            else:
                done = False

            # Backup memory for stateful training before validation.
            if not self.is_training:
                if self.stateful_training:
                    backup_memory_dict_list = self.model.module.get_mem()
                    if self.dual_track:
                        self.model.module.reset(stateful_reset=False)
                    else:
                        self.model.module.reset()
                else:
                    self.model.module.reset()

            for batch_id, batch_data in enumerate(self.dataloader):
                acc_iter_log += 1

                # Important: Must not reset the model before segment iteration, when Stateful training
                if not self.stateful_training:
                    self.model.module.reset()
                elif self.dual_track:
                    print(f"Resetting model: {self.model.module.__class__.__name__} for dual track training")
                    self.model.module.reset(stateful_reset=True)

                if(self.is_training):
                    # print("Training mode")
                    self.model.train()
                    self.optimizer.zero_grad()
                    if(use_bf16):
                        batch_data = self._cast_batch_to_dtype(batch_data, torch.bfloat16)
                    with autocast(dtype=torch.bfloat16, enabled=(self.config.use_amp and not use_bf16), device_type=device_type):
                        self.computer.compute(
                                  *batch_data, 
                                  local_batch_id=batch_id,
                                  global_batch_id=self.get_global_batch_id,
                                  global_epoch_id=self.get_global_epoch_id)
                    if(manual_sync):
                        for param in self.model.parameters():
                            if(param.grad is not None):
                                param.grad = param.grad.contiguous()
                                dist.all_reduce(param.grad)
                                param.grad.div_(self.world_size)
                    # log_sum_parameters_grad(self.model, self.rank)
                    apply_gradient_safely(self.model, self.optimizer, scaler=self.scaler)
                    self.lr_scheduler.step()
                    self.training_metainfo["steps"] += 1
                else:
                    self.model.eval()
                    with torch.no_grad():
                        if(use_bf16):
                            batch_data = self._cast_batch_to_dtype(batch_data, torch.bfloat16)
                        
                        # print("-------------------------------------")
                        # print("batch_data: ", batch_data[0].dtype)
                        # last_type = None
                        # for name, param in self.model.named_parameters():
                        #     if param.dtype != last_type:
                        #         print(f"{name}: {param.dtype}")
                        #         last_type = param.dtype
                        # print("-------------------------------------")
                        
                        self.computer.compute(
                                  *batch_data, 
                                  local_batch_id=batch_id,
                                  global_batch_id=self.get_global_batch_id,
                                  global_epoch_id=self.get_global_epoch_id)

                # Emergency Save
                if(self.emergency_save_check()):
                    log_debug("Emergency save triggered, saving model...")
                    custom_save_model(self.model, self.config.save_model_path,
                                    self.__class__.__name__, self.training_metainfo,
                                    appendix="emergency",
                                    optimizer=self.optimizer, lr_scheduler=self.lr_scheduler,
                                    scaler=self.scaler)

                # Safety Check and Save
                need_break = False
                if(self.is_training and self.config.has_attr("max_save_iterations") 
                                and (self.get_global_batch_id + 1) % self.config.max_save_iterations == 0
                                and self.config.max_save_iterations > 0):
                    log_debug("\nSAVE MODEL FOR FAIL-SAFETY...\n", on=self.main)
                    if(self.main):
                        check_model_validity(self.model.module)
                        global_epoch_id=self.get_global_epoch_id
                        save_model_path = model_path(self.config.save_model_path, global_epoch_id)
                        # torch.save(self.model.state_dict(), save_model_path)
                        current_iter = self.training_metainfo["steps"]
                        custom_save_model(self.model, self.config.save_model_path,
                                self.__class__.__name__, self.training_metainfo,
                                appendix=f"iter_{current_iter}",
                                optimizer=self.optimizer, lr_scheduler=self.lr_scheduler,
                                scaler=self.scaler)
                        # Save additional iter-based model file.
                        # iter_model_path = os.path.join(self.config.save_model_path, f"model-epoch{global_epoch_id}-{acc_iter_log}.pth")
                        # torch.save(self.model.state_dict(), iter_model_path)
                    need_break = True

                
                if(not self.is_training):
                    log_progress((batch_id + 1) / data_length, on=self.main)

                yield need_break, done

            if self.is_training:
                self.training_metainfo["epochs"] += 1
            else:
                # Restore memory after validation
                if self.stateful_training:
                    self.model.module.set_mem(backup_memory_dict_list)
            
            # Save At Training Epoch End
            if(self.main and self.is_training):
                custom_save_model(self.model, self.config.save_model_path,
                                self.__class__.__name__, self.training_metainfo,
                                optimizer=self.optimizer, lr_scheduler=self.lr_scheduler,
                                scaler=self.scaler)

            if("training_metainfo" in self.__dict__ and self.is_training):
                done = self.training_metainfo["epochs"] > self.config.max_epochs
            else:
                done = False

            yield True, done

    return WrapperEpochManager

def dist_process(rank, use_gpu, world_size, config, main_rank,
                model_type, train_objects, evaluate_objects, extra_info):
    if use_gpu:
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        device_type = 'cuda'
        print(f"[Rank {rank}] Using GPU: {local_rank}, Total GPUs: {torch.cuda.device_count()}")
        dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=3600))  # 延长至30分钟)
    else:
        device = torch.device('cpu')
        device_type = 'cpu'
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

    if(main_rank is None):
        main = False
    elif(main_rank == "all" or main_rank == rank):
        main = True
    else:
        main = False

    if(main):
        log_debug("Main gpu", use_gpu, "rank:", rank, device)
        



    start_time = time.time()
    # Create model and move it to GPU with id `gpu`
    model = model_type(config.model_config, verbose=main)
    model = model.to(device)
    time_cost = time.time() - start_time
    print(f"Model init time cost: ", time_cost)
    
    start_time = time.time()
    if(config.has_attr("use_bf16") and config.use_bf16):
        model = model.to(torch.bfloat16)    
    if use_gpu:
        if is_multi_node():
            # If using multiple nodes, we need to specify the device_ids
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        else:
            model = DDP(model, device_ids=[rank])
    else:
        model = DDP(model)

    time_cost = time.time() - start_time
    print(f"Model DDP time cost: ", time_cost)


    start_time = time.time()
    extra_states = dict()
    # Load the model if specified in the configuration
    if(config.has_attr("load_model_path") and 
            config.load_model_path is not None and 
            config.load_model_path.lower() != 'none'):
        if(config.has_attr("load_model_parameter_blacklist")):
            black_list = config.load_model_parameter_blacklist
        else:
            black_list = []
        model, metainfo, extra_states = custom_load_model(model, config.load_model_path, 
                                  black_list=black_list,
                                  verbose=main, 
                                  strict_check=False)
    else:
        metainfo = []
        log_warn("No model is loaded as `load_model_path` is not found in config or is None", on=main)

    if(not isinstance(train_objects, list) and not isinstance(train_objects, tuple)):
        train_objects = [train_objects]
    if(not isinstance(evaluate_objects, list) and not isinstance(evaluate_objects, tuple)):
        evaluate_objects = [evaluate_objects]        

    if(config.has_attr('monitor_dir')):
        watch_dir = config.monitor_dir
    else:
        watch_dir = None

    train_list = []
    for train_object in train_objects:
        if(train_object.__name__ not in metainfo):
            object_info = dict()
        else:
            object_info = metainfo[train_object.__name__]
        if(config.has_attr("reset_metainfo")):
            for key, value in config.get_dict("reset_metainfo").items():
                object_info[key] = value
        train_list.append(train_object(run_name=config.run_name, 
                                        model=model, 
                                        training_metainfo=object_info,
                                        config=config.train_config,
                                        log_config=config.log_config,
                                        rank=rank,
                                        world_size=world_size,
                                        device_type=device_type,
                                        device=device,
                                        main=main,
                                        is_training=True,
                                        stateful_training=config.stateful_training,
                                        dual_track=config.dual_track,
                                        watch_dir=watch_dir,
                                        extra_info=extra_info,
                                        use_bf16=config.use_bf16 if config.has_attr("use_bf16") else False,
                                        extra_states=extra_states,
                                        ))



    evaluate_list = []
    # Build log_config.
    for dataset in config.test_config.datasets:
        # Create test_config，load dataset dict.
        test_config = Configure()
        test_config.from_dict(dataset)
        # Build log_config.
        log_config = Configure()
        log_config_dict = {
            # "tensorboard_log": dataset["log_dir"],
            "tensorboard_log": config.log_config.tensorboard_log,
            "evaluation_log": dataset["output"],
            "use_tensorboard": config.log_config.use_tensorboard
        }
        log_config.from_dict(log_config_dict)

        # Create evalutaion objects.
        evaluate_list.append(evaluate_objects[0](
            run_name=f"{config.run_name}_{dataset['name']}",
            model=model,
            training_metainfo=dict(),
            config=test_config,
            log_config=log_config,
            rank=rank,
            world_size=world_size,
            device_type=device_type,
            device=device,
            main=main,
            is_training=False,
            stateful_training=config.stateful_training,
            dual_track=config.dual_track,
            extra_info=extra_info,
            use_bf16=config.use_bf16 if config.has_attr("use_bf16") else False,
            extra_states=extra_states,
        ))

    time_cost = time.time() - start_time
    print(f"Config time cost: ", time_cost)

    start_time = time.time()
    for train_object in train_list:
        train_object._preprocess()
    time_cost = time.time() - start_time
    print(f"Preprocess time cost: ", time_cost)
    
    for evaluate_object in evaluate_list:
        evaluate_object._preprocess()

    def evaluate_epoch():
        for evaluate_object in evaluate_list:
            print(f"Evaluating dataset: {evaluate_object.run_name}")
            evaluate_object._epoch_start()
            for _ in evaluate_object.run(device, device_type):
                pass
            evaluate_object._epoch_end()

    if(len(train_list) < 1):
        evaluate_epoch() # Doing single epoch evaluation
    else:
        all_train_done = False
        while not all_train_done:
            all_train_done = True
            for train_object in train_list:
                train_object._epoch_start()
                for need_evaluate, done in train_object.run(device, device_type):                        
                    if(need_evaluate):
                        evaluate_epoch()
                    if(not done):
                        all_train_done = False
                train_object._epoch_end()

    for train_object in train_list:
        train_object._postprocess()
    for evaluate_object in evaluate_list:
        evaluate_object._postprocess()

class Runner(object):
    """
    Trainer class manage the training process and framework
    """
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('configuration', type=str, help="YAML configuration file")
        parser.add_argument('--configs', nargs='*', help="List of all configurations, overwrite configuration file: eg. train_config.batch_size=16 test_config.xxx=...")
        args = parser.parse_args()

        self.use_gpu = torch.cuda.is_available()
        self.world_size = torch.cuda.device_count() if self.use_gpu else os.cpu_count()
        if "WORLD_SIZE" in os.environ:
            self.world_size = int(os.environ["WORLD_SIZE"])

        if(self.use_gpu):
            log_debug("Use Parallel GPUs: %s" % self.world_size)
        else:
            log_debug("Use Parallel CPUs: %s" % self.world_size)

        self.config = Configure()
        self.config.from_yaml(args.configuration)

        # Get the dictionary of attributes
        if args.configs:
            for pair in args.configs:
                key, value = pair.split('=')
                self.config.set_value(key, value)
                print(f"Rewriting configurations from args: {key} to {value}")
        
        print("Final configuration:\n", self.config)

        if not is_multi_node():
            if(self.config.has_attr('monitor_dir')):
                create_folder(self.config.monitor_dir)
                self.config.from_yaml(f"{self.config.monitor_dir}/config_monitor.yaml")

            if('MASTER_ADDR' in os.environ):
                log_debug(f"Environment variable MASTER_ADDR is already set, using {os.environ['MASTER_ADDR']}.")
            else:
                if(self.config.has_attr('master_addr')):
                    os.environ['MASTER_ADDR'] = self.config.master_addr
                else:
                    os.environ['MASTER_ADDR'] = 'localhost' 
                log_debug(f"MASTER_ADDR set to {os.environ['MASTER_ADDR']}.")
            if('MASTER_PORT' in os.environ):
                log_debug(f"Environment variable MASTER_PORT is already set, using {os.environ['MASTER_PORT']}.")
            else:
                os.environ['MASTER_PORT'] = self.config.master_port
                log_debug(f"MASTER_PORT set to {os.environ['MASTER_PORT']}.")

    def start(self, model_type, train_objects, evaluate_objects, extra_info=None):

        if is_multi_node():
            print("Training with Multi-node")
            rank = int(os.environ["RANK"])
            local_rank = int(os.environ["LOCAL_RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            dist_process(
                rank=rank,
                use_gpu=self.use_gpu,
                world_size=world_size,
                config=self.config,
                main_rank=0,
                model_type=model_type,
                train_objects=train_objects,
                evaluate_objects=evaluate_objects,
                extra_info=extra_info
            )
        else:
            mp.spawn(dist_process,
                    args=(self.use_gpu, 
                        self.world_size, 
                        self.config, 
                        0, # always use #0 as the main GPU
                        model_type,
                        train_objects, 
                        evaluate_objects,
                        extra_info),
                    nprocs=self.world_size if self.use_gpu else min(self.world_size, 4),  # Limit CPU processes if desired
                    join=True)
