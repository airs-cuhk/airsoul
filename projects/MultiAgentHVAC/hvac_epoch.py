import os
import torch
import numpy
import pickle
from copy import deepcopy
from pathlib import Path

from airsoul.dataloader import segment_iterator
from airsoul.utils import Logger, log_progress, log_debug, log_warn, log_fatal
from airsoul.utils import DistStatistics, downsample
from airsoul.utils import EpochManager, GeneratorBase, Logger
from airsoul.dataloader import MultiAgentLoadDateSet, MultiAgentDataSetVetorized, MultiAgentDistributionLoadDateSet
from data.anyhvac import create_cooler_cooler_graph, create_cooler_sensor_graph, HVACRLTester
import data.anyhvac.rl_trainer_hvac
import sys
sys.modules["rl_trainer_hvac"] = data.anyhvac.rl_trainer_hvac

from xenoverse.anyhvac.anyhvac_sampler import HVACTaskSampler
from xenoverse.anyhvac.anyhvac_solver import HVACSolverGTPID
from xenoverse.anyhvac.anyhvac_env import HVACEnvDiscreteAction, HVACEnvDiffAction
from env_wapper import HVACEnvWrapper, plot_cooler_values

def string_mean_var(downsample_length, res):
    string=""
    if(numpy.size(res["mean"]) > 1):
        for i, (xm,xb) in enumerate(zip(res["mean"], res["bound"])):
            string += f'{downsample_length * i}\t{xm}\t{xb}\n'
    else:
        string =  f'{0}\t{res["mean"]}\t{res["bound"]}\n'
    return string

@EpochManager
class HVACEpoch:
    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])
        if self.config.use_kl:
            self.DataType=MultiAgentDistributionLoadDateSet
        else:
            self.DataType=MultiAgentLoadDateSet
        if(self.is_training):
            self.logger_keys = ["learning_rate", 
                        "loss_worldmodel_state", 
                        "loss_worldmodel_other_agent",
                        "loss_worldmodel_reward", 
                        "loss_policymodel",
                        "entropy"]
            self.stat = DistStatistics()
            self.reduce = 1
        else:
            self.logger_keys = ["validation_state_pred", 
                        "validation_other_agent_pred",
                        # "validation_reward_pred", 
                        "validation_policy",
                        "validation_entropy"]
            self.stat_ws = DistStatistics()
            self.stat_wa = DistStatistics()
            self.stat_p = DistStatistics()
            self.reduce = None
            if(self.config.has_attr("downsample_length")):
                self.downsample_length = self.config.downsample_length
            else:
                self.downsample_length = 100
        if(self.config.has_attr('state_dropout')):
            self.state_dropout = self.config.state_dropout
        else:
            self.state_dropout = 0.20
        if(self.config.has_attr('reward_dropout')):
            self.reward_dropout = self.config.reward_dropout
        else:
            self.reward_dropout = 0.20

        self.dataset = MultiAgentDataSetVetorized(
            directory="./",
            time_step=5000,
            max_obs_num=self.config.vocab.max_obs_num,
            max_agent_num=self.config.vocab.max_agent_num,
            prompt_num=self.config.vocab.prompt_num,
            temperature_value_num=self.config.vocab.temperature_value_num,
            temperature_resolution=self.config.vocab.temperature_resolution,
            policy_value_num=self.config.vocab.policy_value_num,
            policy_resolution=self.config.vocab.policy_resolution,
            vocab_size=self.config.vocab.vocab_size,
            verbose=False
        )

    def prepare_label_for_kl_loss(self, action_start_idx, action_end_idx, vocab_size, label, device, 
                             check_prob_sum=True, tolerance=1e-6):
        """
        准备标签数据用于计算词表化模型的 KL 散度损失
        
        参数:
            action_start_idx: int, 动作起始索引
            action_end_idx: int, 动作结束索引
            vocab_size: int, 词表大小
            label: torch.Tensor, 原始标签，shape [batch, len, action_discrete_num]
            device: torch.device, 目标设备
            check_prob_sum: bool, 是否检查概率和是否为1
            tolerance: float, 概率和的容错范围
            
        返回:
            torch.Tensor: 处理后的标签，shape [batch, len, vocab_size]
        """
        # 获取维度信息
        batch_size, seq_len, action_discrete_num = label.shape
        
        # 检查 action_discrete_num 是否匹配
        expected_action_num = action_end_idx - action_start_idx
        if action_discrete_num != expected_action_num:
            raise ValueError(
                f"action_discrete_num ({action_discrete_num}) 不等于 "
                f"(action_end_idx - action_start_idx) ({expected_action_num})"
            )
        
        # 创建全零张量
        new_label = torch.zeros(batch_size, seq_len, vocab_size, device=device, dtype=label.dtype)
        
        # 复制原始标签值
        new_label[:, :, action_start_idx:action_end_idx] = label
        
        return new_label

    def compute(self, seq_arr, label_arr,
                        global_batch_id=-1,
                        local_batch_id=-1,
                        global_epoch_id=-1):
        """
        Defining the computation function for each batch
        """
        state_dropout = 0.0
        if(self.is_training):
            assert self.optimizer is not None, "optimizer is required for training"
            state_dropout = self.state_dropout
        else:
            state_dropout = 0.0

        losses = []
        for sub_idx, seq, label in segment_iterator(self.config.seq_len, self.config.seg_len, self.device, seq_arr, label_arr):
            if self.config.use_kl:
                action_start_idx = self.dataset.POLICY_VALUE_BASE
                action_end_idx = self.dataset.ACTION_OFF_BASE
                vocab_size = self.dataset.vocab_size
                label = self.prepare_label_for_kl_loss(action_start_idx, action_end_idx, vocab_size, label, self.device)
            loss = self.model.module.sequential_loss(
                    seq, 
                    label, 
                    use_loss_weight=self.is_training,
                    update_memory=True,
                    use_kl=self.config.use_kl,
                    reduce_dim=self.reduce)
            losses.append(loss)

            if(self.is_training):
                syn_loss = (self.config.lossweight_worldmodel_states * loss["wm_obs"]
                        + self.config.lossweight_worldmodel_actions * loss["wm_agent"]
                        + self.config.lossweight_policymodel * loss["policy"]
                        + self.config.lossweight_worldmodel_rewards * loss["reward"]
                        + self.config.lossweight_entropy * loss["ent"]
                        + self.config.lossweight_l2 * loss["causal-l2"])
                if(self.scaler is not None):
                    self.scaler.scale(syn_loss).backward()
                else:
                    syn_loss.backward()
                self.stat.gather(self.device,
                    loss_worldmodel_state = loss["wm_obs"] / loss["count_s"],
                    loss_worldmodel_other_agent = loss["wm_agent"] / loss["count_a"],
                    loss_worldmodel_reward = loss["reward"] / loss["count_p"],
                    loss_policymodel = loss["policy"] / loss["count_p"],
                    entropy = -loss["ent"] / loss["count_p"],
                    count = loss["count_p"])
                
        if(self.is_training):
            stat_res = self.stat()
            if(self.logger is not None):
                self.logger(self.optimizer.param_groups[0]['lr'],
                        stat_res["loss_worldmodel_state"]["mean"], 
                        stat_res["loss_worldmodel_other_agent"]["mean"],
                        stat_res["loss_worldmodel_reward"]["mean"], 
                        stat_res["loss_policymodel"]["mean"], 
                        stat_res["entropy"]["mean"],
                        epoch=global_epoch_id,
                        iteration=local_batch_id)
        else:
            loss_wm_s = torch.cat([loss["wm_obs"] / torch.clamp_min(loss["count_s"], 1.0e-3) 
                    for loss in losses], dim=1)
            loss_wm_a = torch.cat([loss["wm_agent"] / torch.clamp_min(loss["count_a"], 1.0e-3) 
                    for loss in losses], dim=1)
            loss_wm_r = torch.cat([loss["reward"] / torch.clamp_min(loss["count_p"], 1.0e-3) 
                    for loss in losses], dim=1)
            loss_pm = torch.cat([loss["policy"] / torch.clamp_min(loss["count_p"], 1.0e-3) 
                    for loss in losses], dim=1)
            loss_ent = torch.cat([-loss["ent"] / torch.clamp_min(loss["count_p"], 1.0e-3) 
                    for loss in losses], dim=1)
            counts_ws = torch.cat([loss["count_s"] for loss in losses], dim=1)
            counts_wa = torch.cat([loss["count_a"] for loss in losses], dim=1)
            counts_p = torch.cat([loss["count_p"] for loss in losses], dim=1)

            bsz = loss_wm_s.shape[0]
            self.obs_sensor_pre_step = int(torch.sum(counts_ws) / torch.sum(counts_p))
            self.obs_agent_pre_step = int(torch.sum(counts_wa) / torch.sum(counts_p))
            
            def extract_nonzero_per_batch(tensor):
                results = []
                max_len = 0
                for i in range(tensor.size(0)):
                    batch_data = tensor[i]
                    mask = batch_data != 0
                    non_zero = batch_data[mask]
                    results.append(non_zero)
                    if non_zero.numel() > max_len:
                        max_len = non_zero.numel()
                padded = torch.zeros(len(results), max_len, 
                                    dtype=tensor.dtype, 
                                    device=tensor.device)
                for i, t in enumerate(results):
                    padded[i, :t.numel()] = t
                return padded

            loss_wm_s = extract_nonzero_per_batch(loss_wm_s)
            loss_wm_a = extract_nonzero_per_batch(loss_wm_a)
            loss_wm_r = extract_nonzero_per_batch(loss_wm_r)
            loss_pm = extract_nonzero_per_batch(loss_pm)
            loss_ent = extract_nonzero_per_batch(loss_ent)
            counts_ws = extract_nonzero_per_batch(counts_ws)
            counts_wa = extract_nonzero_per_batch(counts_wa)
            counts_p = extract_nonzero_per_batch(counts_p)

            loss_wm_s = downsample(loss_wm_s, self.downsample_length * self.obs_sensor_pre_step)
            loss_wm_a = downsample(loss_wm_a, self.downsample_length * self.obs_agent_pre_step)
            loss_wm_r = downsample(loss_wm_r, self.downsample_length)
            loss_pm = downsample(loss_pm, self.downsample_length)
            loss_ent = downsample(loss_ent, self.downsample_length)
            counts_ws = downsample(counts_ws, self.downsample_length * self.obs_sensor_pre_step)
            counts_wa = downsample(counts_wa, self.downsample_length * self.obs_agent_pre_step)
            counts_p = downsample(counts_p, self.downsample_length)

            def check_shape(loss,count):
                if loss.shape[0] != count.shape[0]:
                    min_len = min(loss.shape[0], count.shape[0])
                    return  loss[:min_len], count[:min_len]
                else:
                    return loss, count
            
            for i in range(bsz):
                wm_loss, wm_count = check_shape(loss_wm_s[i], counts_ws[i])
                wa_loss, wa_count = check_shape(loss_wm_a[i], counts_wa[i])

                self.stat_ws.gather(self.device,
                        validation_state_pred=wm_loss, 
                        count=wm_count)
                self.stat_wa.gather(self.device,
                        validation_other_agent_pred=wa_loss,
                        count=wa_count)
                self.stat_p.gather(self.device,
                        # validation_reward_pred=loss_wm_r[i], 
                        validation_policy=loss_pm[i],
                        validation_entropy=loss_ent[i],
                        count=counts_p[i])
    
    def epoch_end(self, epoch_id):
        if(not self.is_training):
            stat_res_ws = self.stat_ws()
            stat_res_wa = self.stat_wa()
            stat_res_p = self.stat_p()
            if(self.logger is not None):
                self.logger(stat_res_ws["validation_state_pred"]["mean"], 
                        stat_res_wa["validation_other_agent_pred"]["mean"],
                        # stat_res_p["validation_reward_pred"]["mean"], 
                        stat_res_p["validation_policy"]["mean"],
                        stat_res_p["validation_entropy"]["mean"],
                        epoch=epoch_id)
            if(self.extra_info is not None):
                if(self.extra_info.lower() == 'validate' and self.main):
                    if not os.path.exists(self.config.output):
                        os.makedirs(self.config.output)
                    for key_name in stat_res_ws:
                        res_text = string_mean_var(self.downsample_length * self.obs_sensor_pre_step, stat_res_ws[key_name])
                        file_path = f'{self.config.output}/result_{key_name}.txt'
                        if os.path.exists(file_path):
                            os.remove(file_path)
                        with open(file_path, 'w') as f_model:
                            f_model.write(res_text)
                    for key_name in stat_res_wa:
                        res_text = string_mean_var(self.downsample_length * self.obs_agent_pre_step, stat_res_wa[key_name])
                        file_path = f'{self.config.output}/result_{key_name}.txt'
                        if os.path.exists(file_path):
                            os.remove(file_path)
                        with open(file_path, 'w') as f_model:
                            f_model.write(res_text)
                    for key_name in stat_res_p:
                        res_text = string_mean_var(self.downsample_length, stat_res_p[key_name])
                        file_path = f'{self.config.output}/result_{key_name}.txt'
                        if os.path.exists(file_path):
                            os.remove(file_path)
                        with open(file_path, 'w') as f_model:
                            f_model.write(res_text)

class HVACGenerator(GeneratorBase):
    def preprocess(self):
        if(self.config.env.lower().find("hvac") >= 0):
            self.env = HVACEnvWrapper(reward_mode = self.config.default_prompt, verbose=True)
            self.pid_env = HVACEnvDiscreteAction(reward_mode = self.config.default_prompt, verbose=False)
            self.rl_coach_env = HVACEnvDiffAction(reward_mode = self.config.default_prompt, verbose=False)
            self.constant_env = HVACEnvDiscreteAction(reward_mode = self.config.default_prompt, verbose=False)
            self.task_sampler = self.task_sampler_anyhvacv2
        else:
            log_fatal("Unsupported environment:", self.config.env)

        if(self.config.has_attr("task_file")):
            if self.config.multi_task:
                self.tasks = []
                self.tasks_folder = []
                base_dir = self.config.task_file
                if os.path.exists(base_dir) and os.path.isdir(base_dir):
                    for subdir in os.listdir(base_dir):
                        subdir_path = os.path.join(base_dir, subdir)
                        if os.path.isdir(subdir_path):
                            for file_name in os.listdir(subdir_path):
                                if file_name.endswith('.pkl'):
                                    pkl_file_path = os.path.join(subdir_path, file_name)
                                    try:
                                        with open(pkl_file_path, 'rb') as fr:
                                            task_data = pickle.load(fr)
                                            self.tasks.append(task_data)
                                            self.tasks_folder.append(subdir_path)
                                        log_debug(f"Loaded task from {pkl_file_path}")
                                    except Exception as e:
                                        log_warn(f"Failed to load {pkl_file_path}: {e}")
            else:
                with open(self.config.task_file, 'rb') as fr:
                    self.tasks = pickle.load(fr)
                    self.task_name = os.path.splitext(os.path.basename(pkl_file_path))[0]
            log_debug(f"Read tasks from {self.config.task_file} success")
        else:
            self.tasks = None

        logger_keys = ["step", "reward", "state_prediction", "action_prediction", "reward_prediction"]
        self.stat = DistStatistics(*logger_keys)
        self.logger = Logger("trail_idx",
                            "total_steps",
                            *logger_keys, 
                            on=self.main, 
                            use_tensorboard=False)
        
        self.dataset = MultiAgentDataSetVetorized(
            directory="./",
            time_step=5000,
            max_obs_num=self.config.vocab.max_obs_num,
            max_agent_num=self.config.vocab.max_agent_num,
            prompt_num=self.config.vocab.prompt_num,
            temperature_value_num=self.config.vocab.temperature_value_num,
            temperature_resolution=self.config.vocab.temperature_resolution,
            policy_value_num=self.config.vocab.policy_value_num,
            policy_resolution=self.config.vocab.policy_resolution,
            vocab_size=self.config.vocab.vocab_size,
            verbose=False
        )
        self.vocabularize = self.dataset.vocabularize
        self.interactive_prompt = self.config.default_prompt

        self.max_total_steps = self.config.max_total_steps
        self.max_trails = self.config.max_trails
        self.max_steps = self.config.max_steps

        self.mask_self_action = True
        self.mask_self_action_step = 500
    
    def epoch_end(self, epoch_id):
        pass

    def task_sampler_anyhvacv2(self, epoch_id=0):
        task_id = None
        if(self.tasks is None):
            task = HVACTaskSampler(control_type='Temperature')
        else:
            if self.config.multi_task:
                task_id = (epoch_id * self.world_size + self.rank) % len(self.tasks)
                task = self.tasks[task_id]
                self.task_folder = self.tasks_folder[task_id]
                self.task_name = os.path.basename(self.task_folder)
                self.task_try_count = (epoch_id * self.world_size + self.rank) // len(self.tasks)
            else:
                task = self.tasks
        self.env.set_task(task, 
                                    discretize_rl_action_space=False,
                                    add_action_cost=False,
                                    too_cold_limit=True)
        self.pid_env.set_task(task)
        self.rl_coach_env.set_task(task, 
                                    discretize_rl_action_space=True,
                                    add_action_cost=False,
                                    too_cold_limit=False)
        self.constant_env.set_task(task)

        knn = 4
        obs_graph = create_cooler_sensor_graph(self.env, knn)
        agent_graph = create_cooler_cooler_graph(self.env, knn)
        self.agent_num, self.sensor_num = obs_graph.shape
        self.related_sensor = numpy.zeros((self.agent_num, knn), dtype=numpy.int32)
        self.related_agent = numpy.zeros((self.agent_num, knn), dtype=numpy.int32)
        for i in range(self.agent_num):
            sensor_indices = numpy.where(obs_graph[i] > 0)[0]
            sensor_weights = obs_graph[i][sensor_indices]
            sensor_sorted_indices = numpy.argsort(sensor_weights)
            sensor_indices = sensor_indices[sensor_sorted_indices]
            agent_indices = numpy.where(agent_graph[i] > 0)[0]
            agent_weights = agent_graph[i][agent_indices]
            agent_sorted_indices = numpy.argsort(agent_weights)
            agent_indices = agent_indices[agent_sorted_indices]
            self.related_sensor[i] = sensor_indices
            self.related_agent[i] = agent_indices

        self.env._create_agent_target_temperture(obs_graph)

        return True
    
    def in_context_learn_from_teacher(self, epoch_id):
        pass # TODO

    def build_up_vocab_seq_in_batch(self, obs_sensor, obs_agent, current_batch_seq=None, 
                                    action=None, reward=None, reset=False, use_relative_idx=True):
        # [num_agent, time, value]
        # obs_agent and action should contain [:,t-1:t,:] two timestep if t>0.
        if current_batch_seq is None:
            current_batch_seq = []
            # [num, value] -> [num, 1, value]
            obs_sensor = obs_sensor[:,numpy.newaxis,numpy.newaxis]
            obs_sensor_vocabularize = self.vocabularize('value', obs_sensor).squeeze()
            obs_agent_vocabularize = self.vocabularize('value', obs_agent, use_diff_action=False)[:,-1:,:].squeeze()
            for agent_id in range(self.agent_num):
                current_agent_seq = []
                # 1, Related agent idx and value
                if use_relative_idx:
                    current_agent_seq.append(self.vocabularize('agent_id', 0))
                else:
                    current_agent_seq.append(self.vocabularize('agent_id', agent_id))
                if self.mask_self_action:
                    current_agent_seq.append(self.dataset.PADDING_IDX)
                else:
                    current_agent_seq.append(obs_agent_vocabularize[agent_id])
                for i, related_agent in enumerate(self.related_agent[agent_id]):
                    if use_relative_idx:
                        current_agent_seq.append(self.vocabularize('agent_id', i+1))
                    else:
                        current_agent_seq.append(self.vocabularize('agent_id', related_agent))
                    current_agent_seq.append(obs_agent_vocabularize[related_agent])
                    # current_agent_seq.append(self.vocabularize('value', obs_agent[related_agent]))
                # 2, Related sensor idx and value
                for i, related_obs in enumerate(self.related_sensor[agent_id]):
                    if use_relative_idx:
                        current_agent_seq.append(self.vocabularize('obs_id', i))
                    else:
                        current_agent_seq.append(self.vocabularize('obs_id', related_obs))
                    current_agent_seq.append(obs_sensor_vocabularize[related_obs])
                    # current_agent_seq.append(self.vocabularize('value', obs_sensor[related_obs]))
                # 3, Prompt
                current_agent_seq.append(self.vocabularize('special_token', 'idx_prompt'))
                current_agent_seq.append(self.vocabularize('prompt_value', self.interactive_prompt))
                # 4, Self action flag
                current_agent_seq.append(self.vocabularize('special_token', 'idx_a_self'))
                current_batch_seq.append(current_agent_seq)
            current_batch_seq = [[int(x) for x in lst] for lst in current_batch_seq]
            return current_batch_seq
        else:
            agent_action_vocabularize = self.vocabularize('policy_action', action).squeeze()
            if self.mask_self_action:
                agent_action_vocabularize = numpy.full_like(agent_action_vocabularize, self.dataset.PADDING_IDX) 
                
            if self.model.module.include_reward:
                reward_idx_vocabularize = self.vocabularize('special_token', 'idx_reward')
                reward_vocabularize = self.vocabularize('value', reward)
            timestep_end_vocabularize = self.vocabularize('special_token', 
                                             'idx_reset_env' if reset else 'idx_end_timestep')
            if self.model.module.include_reward:
                to_add = numpy.array([
                    agent_action_vocabularize,                              # 5, Self action value
                    numpy.full(self.agent_num, reward_idx_vocabularize),   # 6, Reward idx and value
                    numpy.full(self.agent_num, reward_vocabularize),  
                    numpy.full(self.agent_num, timestep_end_vocabularize)  # 7, End
                ], dtype=object).T  # (num_agents, 4)
            else:
                to_add = numpy.array([
                    agent_action_vocabularize,                             # 5, Self action value
                    numpy.full(self.agent_num, timestep_end_vocabularize)  # 6, End
                ], dtype=object).T  # (num_agents, 2)
            for i, seq in enumerate(current_batch_seq):
                seq.extend(to_add[i])
            current_batch_seq = [[int(x) for x in lst] for lst in current_batch_seq]
            return current_batch_seq
    
    def build_up_env_action(self, action_in_vocab):
        # Shape of action_in_vocab: [num, 1, 1]
        # Shape of action_value_previous: [num, 1, 2], action: (on/off, temp)

        action_in_value = self.vocabularize('action_vocab', 
                                            action_in_vocab) # [num, 2]
        # Convert actual temperature settings to normalized values 
        action_temp_diff = action_in_value.squeeze()[:,1]
        action_temp_diff = numpy.clip(action_temp_diff, -3.0, 3.0)
        action_in_value = action_temp_diff
        action_temp_diff_normalized = (action_temp_diff + 3) / 6
        
        return action_temp_diff_normalized
    
    def _convert_diff_action(self, previous_action, current_action):
        prev_temp = previous_action[:, 0, 0]  # shape: (n_agents,)
        prev_switch = previous_action[:, 0, 1]  # shape: (n_agents,)
        curr_temp = current_action[:, 0, 0]  # shape: (n_agents,)
        curr_switch = current_action[:, 0, 1]  # shape: (n_agents,)
        diff_temp = curr_temp - prev_temp
        diff_switch = curr_switch
        data = numpy.column_stack((diff_temp, diff_switch))
        result = data.reshape((len(diff_temp), 1, 2))
        return result
    
    def _append_step_data(self, step_data, stat_array_dict):
        
        for key, value in step_data.items():
            if key not in stat_array_dict:
                log_warn(f"Key '{key}' not found in stat_array_dict, skipping...")
                continue
            
            if key in ["reward", "cooler_power"]:
                stat_array_dict[key] = numpy.append(stat_array_dict[key], float(value))
            else:
                stat_array_dict[key] = numpy.append(stat_array_dict[key], int(value))
        
        return stat_array_dict
    
    def _init_pid_rl_solver(self):

        self.pid_env_last_obs = self.pid_env.reset()[0]
        self.rl_env_last_obs = self.rl_coach_env.reset()[0]
        self.constant_env.reset()

        self.pid_solver = HVACSolverGTPID(self.pid_env)
        self.n_coolers = len(self.pid_env.coolers)

        rl_path = os.path.join(
            self.task_folder, 
            f"{self.config.rl_mode_type}_reward_mode_{self.config.default_prompt}.zip"
        )
    
        if not os.path.exists(rl_path):
            log_warn(f"RL solver file not found: {rl_path}")
            return False
        
        try:
            self.rl_solver = HVACRLTester(rl_path, self.config.rl_mode_type, "cpu")
            self.rl_solver.reset()
            log_debug(f"Successfully initialized RL solver from {rl_path}")

        except Exception as e:
            log_warn(f"Failed to initialize RL solver from {rl_path}: {e}")
            return False

        def create_stat_dict():
            return {
                "reward": numpy.array([], dtype=numpy.float32),
                "cooler_power": numpy.array([], dtype=numpy.float32),
                "cool_0": numpy.array([], dtype=numpy.int32),
                "cool_-2": numpy.array([], dtype=numpy.int32),
                "cool_-4": numpy.array([], dtype=numpy.int32),
                "cool_-6": numpy.array([], dtype=numpy.int32),
                "hot_0": numpy.array([], dtype=numpy.int32),
                "hot_2": numpy.array([], dtype=numpy.int32),
                "hot_4": numpy.array([], dtype=numpy.int32),
                "hot_6": numpy.array([], dtype=numpy.int32),
                "fail": numpy.array([], dtype=numpy.int32)
            }

        self.stat_array_dict_pid = create_stat_dict()
        self.stat_array_dict_rl_coach = create_stat_dict()
        self.stat_array_dict_constant = create_stat_dict()

        return True

        
        
    def _step_pid_rl_env(self):
        pid_action = 1 - self.pid_solver.policy(self.pid_env_last_obs["sensor_readings"])[self.n_coolers:]
        rl_action = self.rl_solver.predict(self.rl_env_last_obs, deterministic=False)
        constant_action = self.constant_env.sample_action(mode="constant")


        self.pid_env_last_obs, reward_pid, terminated_pid, truncated_pid, info_pid = self.pid_env.step(pid_action)
        self.rl_env_last_obs, reward_rl, terminated_rl, truncated_rl, info_rl = self.rl_coach_env.step(rl_action)
        _, _ , terminated_constant, truncated_constant, info_constant = self.constant_env.step(constant_action)

        self.stat_array_dict_pid = self._append_step_data(info_pid["step_stat"], self.stat_array_dict_pid)
        self.stat_array_dict_rl_coach = self._append_step_data(info_rl["step_stat"], self.stat_array_dict_rl_coach)
        self.stat_array_dict_constant = self._append_step_data(info_constant["step_stat"], self.stat_array_dict_constant) 
        
        if terminated_pid or truncated_pid:
            pass
            # self.pid_env_last_obs = self.pid_env.reset()[0]
            # self.pid_solver = HVACSolverGTPID(self.pid_env)
        if terminated_rl or truncated_rl:
            pass
            # self.rl_env_last_obs = self.rl_coach_env.reset()[0]
            # self.rl_solver.reset()

    def _save_stat_array_dict(self, stat_array_dict, target_path, compressed=True):

        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        
        if compressed:
            numpy.savez_compressed(target_path, **stat_array_dict)
        else:
            numpy.savez(target_path, **stat_array_dict)
        
        print(f"Statistics saved to {target_path}")

    def __call__(self, epoch_id):

        task_id = self.task_sampler(epoch_id=epoch_id)

        if not self._init_pid_rl_solver():
            return
        
        env_action_array = []
        stat_array_dict = {
            "reward": numpy.array([], dtype=numpy.float32),
            "cooler_power": numpy.array([], dtype=numpy.float32),
            "cool_0": numpy.array([], dtype=numpy.int32),
            "cool_-2": numpy.array([], dtype=numpy.int32),
            "cool_-4": numpy.array([], dtype=numpy.int32),
            "cool_-6": numpy.array([], dtype=numpy.int32),
            "hot_0": numpy.array([], dtype=numpy.int32),
            "hot_2": numpy.array([], dtype=numpy.int32),
            "hot_4": numpy.array([], dtype=numpy.int32),
            "hot_6": numpy.array([], dtype=numpy.int32),
            "fail": numpy.array([], dtype=numpy.int32)
        }

        total_step = 0
        done = False

        if self.config.learn_from_data:
            self.in_context_learn_from_teacher(epoch_id)

        # while total_step < self.max_total_steps:
        print("Max total step: ", self.max_total_steps)
        step = 0

        obs = self.env.reset()[0]
        previous_state = self.env._compute_temperature_deviation(obs)
        
        while total_step < self.max_total_steps:
            temp = self._scheduler(total_step)
            print("obs: ", obs)
            previous_action = self.env._convert_env_action(self.env.current_action)
            vocab_seq_batch = self.build_up_vocab_seq_in_batch(previous_state,
                                                                previous_action)
            
            world_model_obs, world_model_action, action = self.model.module.generate(
                vocab_seq_batch,
                single_batch=False,
                reward_prediction=False,
                Temp=temp)
            env_action = self.build_up_env_action(action)   

            obs, reward, terminated, truncated, info = self.env.step(env_action)
            stat_array_dict = self._append_step_data(info["step_stat"], stat_array_dict)
            env_action = deepcopy(self.env.last_action)
            switch = env_action["switch"]
            value = env_action["value"]
            for i in range(len(switch)):
                if switch[i]<0.5:
                    value[i] = -1.0
            env_action_array.append(value)
            

            obs_deviation = self.env._compute_temperature_deviation(obs)
            current_action = self.env._convert_env_action(self.env.current_action)
            diff_action = self._convert_diff_action(previous_action, current_action)
            done = terminated or truncated
            vocab_seq_batch = self.build_up_vocab_seq_in_batch(previous_state,
                                                                previous_action,
                                                                current_batch_seq=vocab_seq_batch,
                                                                action=diff_action,
                                                                reward=None, #reward=numpy.array(reward).reshape(1, 1, 1),
                                                                reset=done
                                                                )
            cache = self.model.module.incontext_learn(vocab_seq_batch, need_cache=False)
            previous_state = obs_deviation
            step += 1
            total_step += 1

            if step > self.mask_self_action_step:
                self.mask_self_action = False

            print("env_action: ", env_action)
            print("reward: ", reward)
            print("done: ", done)

            if done:
                pass
                # obs = self.env.reset()[0]
                # previous_state = self.env._compute_temperature_deviation(obs)

            # Record pid and rl performance     
            self._step_pid_rl_env()
        
        os.makedirs(self.config.output, exist_ok=True)
        self._save_stat_array_dict(stat_array_dict, os.path.join(self.config.output, f"{self.task_name}_omnirl_stat_try{self.task_try_count}.npz"))
        self._save_stat_array_dict(self.stat_array_dict_pid, os.path.join(self.config.output, f"{self.task_name}_pid_stat_try{self.task_try_count}.npz"))
        self._save_stat_array_dict(self.stat_array_dict_rl_coach, os.path.join(self.config.output, f"{self.task_name}_rl_coach_stat_try{self.task_try_count}.npz"))
        self._save_stat_array_dict(self.stat_array_dict_constant, os.path.join(self.config.output, f"{self.task_name}_constant_stat_try{self.task_try_count}.npz"))
 
        if not self.config.multi_task:
            plot_cooler_values(env_action_array, self.config.output, "behavior", len(self.env.coolers), show_plot=False)