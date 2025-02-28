import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from airsoul.dataloader import segment_iterator
from airsoul.utils import Logger, log_progress, log_debug, log_warn, log_fatal
from airsoul.utils import custom_load_model, noam_scheduler, LinearScheduler
from airsoul.utils import Configure, DistStatistics, rewards2go, downsample
from airsoul.utils import EpochManager, GeneratorBase, Logger
from airsoul.utils import tag_vocabulary, tag_mapping_id, tag_mapping_gamma
from airsoul.dataloader import AnyMDPDataSet, AnyMDPDataSetContinuousState, AnyMDPDataSetContinuousStateAction

import gymnasium 
import gym
import imageio
import numpy
import pickle
from pathlib import Path
import random
import re
from online_rl_utils import DiscreteEnvWrapper, OnlineRL, AgentVisualizer, Switch2
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from l3c.anymdp import AnyMDPTaskSampler
from l3c.anymdp import AnyMDPSolverOpt, AnyMDPSolverOTS, AnyMDPSolverQ
from l3c.anymdp.solver import get_final_transition, get_final_reward
from stable_baselines3 import DQN, A2C, TD3, PPO


def string_mean_var(downsample_length, res):
    string=""
    if(numpy.size(res["mean"]) > 1):
        for i, (xm,xb) in enumerate(zip(res["mean"], res["bound"])):
            string += f'{downsample_length * i}\t{xm}\t{xb}\n'
    else:
        string =  f'{0}\t{res["mean"]}\t{res["bound"]}\n'
    return string

@EpochManager
class OmniRLEpoch:
    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])
        self.DataType=AnyMDPDataSet
        if(self.is_training):
            self.logger_keys = ["learning_rate", 
                        "loss_worldmodel_state", 
                        "loss_worldmodel_reward", 
                        "loss_policymodel",
                        "entropy"]
            self.stat = DistStatistics()
            self.reduce = 1
        else:
            self.logger_keys = ["validation_state_pred", 
                        "validation_reward_pred", 
                        "validation_policy",
                        "validation_entropy"]
            self.stat = DistStatistics()
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

    def compute(self, sarr, parr, tarr, baarr, rarr, laarr,
                        epoch_id=-1, 
                        batch_id=-1):
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
        for sub_idx, states, prompts, tags, bactions, rewards, lactions in segment_iterator(
                    self.config.seq_len, self.config.seg_len, self.device, 
                    (sarr, 1), parr, tarr, baarr, rarr, laarr):
            loss = self.model.module.sequential_loss(
                    states,  # Observations
                    prompts,  # Prompts
                    tags,  # Tags
                    bactions, # Behavior Actions
                    rewards, # Rewards
                    lactions, # Reference Actions
                    state_dropout=state_dropout, 
                    use_loss_weight=self.is_training,
                    reduce_dim=self.reduce) # Do not use loss weight for evaluation
            losses.append(loss)
            if(self.is_training):
                syn_loss = (self.config.lossweight_worldmodel_states * loss["wm-s"]
                        + self.config.lossweight_worldmodel_rewards * loss["wm-r"]
                        + self.config.lossweight_entropy * loss["ent"]
                        + self.config.lossweight_policymodel * loss["pm"]
                        + self.config.lossweight_l2 * loss["causal-l2"])
                if(self.scaler is not None):
                    self.scaler.scale(syn_loss).backward()
                else:
                    syn_loss.backward()
                self.stat.gather(self.device,
                    loss_worldmodel_state = loss["wm-s"] / loss["count_s"],
                    loss_worldmodel_reward = loss["wm-r"] / loss["count_s"],
                    loss_policymodel = loss["pm"] / loss["count_a"],
                    entropy = -loss["ent"] / loss["count_a"],
                    count = loss["count_a"])
        if(self.is_training):
            stat_res = self.stat()
            if(self.logger is not None):
                self.logger(self.optimizer.param_groups[0]['lr'],
                        stat_res["loss_worldmodel_state"]["mean"], 
                        stat_res["loss_worldmodel_reward"]["mean"], 
                        stat_res["loss_policymodel"]["mean"], 
                        stat_res["entropy"]["mean"],
                        epoch=epoch_id,
                        iteration=batch_id)
        else:
            loss_wm_s = torch.cat([loss["wm-s"] / torch.clamp_min(loss["count_s"], 1.0e-3) 
                    for loss in losses], dim=1)
            loss_wm_r = torch.cat([loss["wm-r"] / torch.clamp_min(loss["count_s"], 1.0e-3) 
                    for loss in losses], dim=1)
            loss_pm = torch.cat([loss["pm"] / torch.clamp_min(loss["count_a"], 1.0e-3) 
                    for loss in losses], dim=1)
            loss_ent = torch.cat([-loss["ent"] / torch.clamp_min(loss["count_a"], 1.0e-3) 
                    for loss in losses], dim=1)
            counts = torch.cat([loss["count_a"] for loss in losses], dim=1)

            bsz = loss_wm_s.shape[0]

            loss_wm_s = downsample(loss_wm_s, self.downsample_length)
            loss_wm_r = downsample(loss_wm_r, self.downsample_length)
            loss_pm = downsample(loss_pm, self.downsample_length)
            loss_ent = downsample(loss_ent, self.downsample_length)
            counts = downsample(counts, self.downsample_length)

            for i in range(bsz):
                self.stat.gather(self.device,
                        validation_state_pred=loss_wm_s[i], 
                        validation_reward_pred=loss_wm_r[i], 
                        validation_policy=loss_pm[i],
                        validation_entropy=loss_ent[i],
                        count=counts[i])
            
    def epoch_end(self, epoch_id):
        if(not self.is_training):
            stat_res = self.stat()
            if(self.logger is not None):
                self.logger(stat_res["validation_state_pred"]["mean"], 
                        stat_res["validation_reward_pred"]["mean"], 
                        stat_res["validation_policy"]["mean"],
                        stat_res["validation_entropy"]["mean"],
                        epoch=epoch_id)
            if(self.extra_info is not None):
                if(self.extra_info.lower() == 'validate' and self.main):
                    if not os.path.exists(self.config.output):
                        os.makedirs(self.config.output)
                    for key_name in stat_res:
                        res_text = string_mean_var(self.downsample_length, stat_res[key_name])
                        file_path = f'{self.config.output}/result_{key_name}.txt'
                        if os.path.exists(file_path):
                            os.remove(file_path)
                        with open(file_path, 'w') as f_model:
                            f_model.write(res_text)

# use gamma_vocabulary and tag_vocabulary
class OmniRLGenerator(GeneratorBase):
    def preprocess(self):
        self.mult_anymdp_task = False
        if(self.config.env.lower().find("lake") >= 0):
            self.task_sampler = self.task_sampler_lake
        elif(self.config.env.lower().find("cliff") >= 0):
            self.task_sampler = self.task_sampler_cliff
        elif(self.config.env.lower().find("anymdp") >= 0):
            self.env = gym.make("anymdp-v0", max_steps=self.max_steps)
            self.task_sampler = self.task_sampler_anymdp
            if self.config.mult_anymdp_task:
                self.mult_anymdp_task = True
            if self.config.save_gif:
                self.drawer = AgentVisualizer(self.config.output, visualize_online=False, skip_episode=self.config.save_gif_gap)
        elif(self.config.env.lower().find("mountaincar") >= 0):
            self.task_sampler = self.task_sampler_mountain_car
        elif(self.config.env.lower().find("pendulum") >= 0):
            self.task_sampler = self.task_sampler_pendulum
        elif(self.config.env.lower().find("switch") >= 0):
            self.task_sampler = self.task_sampler_switch
        else:
            log_fatal("Unsupported environment:", self.config.env)

        if(self.config.has_attr("task_file")):
            with open(self.config.task_file, 'rb') as fr:
                self.tasks = pickle.load(fr)
            log_debug(f"Read tasks from {self.config.task_file} success")
        else:
            self.tasks = None

        logger_keys = ["step", "reward", "state_prediction", "reward_prediction", "success_rate"]
        benchmark_logger_keys = ["step", "reward", "success_rate"]

        self.stat = DistStatistics(*logger_keys)
        self.stat_opt = DistStatistics(*benchmark_logger_keys)
        self.stat_online = DistStatistics(*logger_keys)
        self.stat_random = DistStatistics(*benchmark_logger_keys)
        self.logger = Logger("trail_idx",
                            "total_steps",
                            *logger_keys, 
                            on=self.main, 
                            use_tensorboard=False)
        self.logger_benchmark = Logger("trail_idx",
                            "total_steps",
                            *benchmark_logger_keys, 
                            on=self.main, 
                            use_tensorboard=False)
        self.logger_random = Logger("trail_idx",
                            "total_steps",
                            *benchmark_logger_keys, 
                            on=self.main, 
                            use_tensorboard=False)

    def task_sampler_anymdp(self, epoch_id=0):
        task_id = None
        if(self.tasks is None):
            dims = self.config.env.lower().replace("anymdp", "").split("x")
            task = AnyMDPTaskSampler(int(dims[0]), int(dims[1]))
        else:
            task_num = len(self.tasks)
            task_id = (epoch_id * self.world_size + self.rank) % task_num
            task = self.tasks[task_id]
        self.env.set_task(task)
        return task_id

    def task_sampler_lake(self, epoch_id=0):
        self.env = gymnasium.make(
            'FrozenLake-v1', 
            map_name=self.config.env.replace("lake", ""), 
            is_slippery=True, 
            max_episode_steps=1000,
            render_mode='rgb_array_list')
        return None
    
    def task_sampler_cliff(self, epoch_id=0):
        env = gymnasium.make(
            'CliffWalking-v0',
            render_mode='rgb_array_list')
        # If fall into cliff, truncated is set to True and return to starting position.
        # If reach goal, teminated is set to True.
        self.env = DiscreteEnvWrapper(env=env,
                                    env_name=self.config.env.lower(),
                                    action_space=self.config.action_clip,
                                    state_space_dim1=48,
                                    state_space_dim2=1)
        return None
    
    def extract_state_space_dimensions(self, env_name, name="pendulum"):
        pattern = rf'^{name}(\d+)x(\d+)$'
        match = re.match(pattern, env_name)
        if not match:
            raise ValueError(f"Invalid environment name format: {env_name}. Expected format: '{name}<digit>x<digit>'.")
        state_space_dim1 = int(match.group(1))
        state_space_dim2 = int(match.group(2))

        return state_space_dim1, state_space_dim2
    
    def task_sampler_mountain_car(self, epoch_id=0):
        env = gymnasium.make("MountainCar-v0", render_mode="rgb_array_list", max_episode_steps=self.config.max_steps)  
        if self.config.map_env_discrete:
            state_space_dim1, state_space_dim2 = self.extract_state_space_dimensions(self.config.env.lower(), "mountaincar")
            self.env = DiscreteEnvWrapper(env=env,
                                            env_name=self.config.env.lower(),
                                            action_space=self.config.action_clip,
                                            state_space_dim1=state_space_dim1,
                                            state_space_dim2=state_space_dim2,
                                            reward_shaping=True,
                                            skip_frame=self.config.skip_frame)
        else:
            self.env = env
        return None
    
    def task_sampler_pendulum(self, epoch_id=0):
        env = gymnasium.make("Pendulum-v1", render_mode="rgb_array_list", g=9.81, max_episode_steps=self.config.max_steps)
        if self.config.map_env_discrete:
            state_space_dim1, state_space_dim2 = self.extract_state_space_dimensions(self.config.env.lower(), "pendulum")
            self.env = DiscreteEnvWrapper(env=env,
                                            env_name=self.config.env.lower(),
                                            action_space=self.config.action_clip,
                                            state_space_dim1=state_space_dim1,
                                            state_space_dim2=state_space_dim2,
                                            skip_frame=self.config.skip_frame)
        else:
            self.env = env
        return None

    def task_sampler_switch(self, epoch_id=0):
        self.env = Switch2(n_agents=2, full_observable=True, max_steps=self.config.max_steps)
        return None

    def reward_shaping(self, done, terminated, reward):
        if(self.config.env.lower().find("lake") >= 0):
            if done and reward < 0.5:
                reward = -1.0
        elif(self.config.env.lower().find("mountaincar") >= 0):
            if done:
                if terminated:
                    reward = 1.0
        elif(self.config.env.lower().find("pendulum") >= 0):
            reward = max(reward/30 + 0.1, -0.1)
        elif(self.config.env.lower().find("cliff") >=0):
            if done:
                if terminated:
                    reward = 1.0
                else:
                    reward = -1.0    
            else:
                reward = -0.03
            
        return reward
            

    def is_success_fail(self, reward, trail_reward, terminated):
        if(self.config.env.lower().find("lake") >= 0):
            if reward > 1.0e-3:
                return 1
            else:
                return 0
        elif(self.config.env.lower().find("lander") >= 0):
            if trail_reward >= 200:
                return 1
            else:
                return 0
        elif(self.config.env.lower().find("mountaincar") >= 0):
            if terminated:
                return 1
            else:
                return 0
        elif(self.config.env.lower().find("cliff") >= 0):
            if terminated:
                return 1
            else:
                return 0
        else:
            return 0
    
    def reset_env(self):
        if self.config.env.lower().find("pendulum") >= 0:
            state, *_ = self.env.reset(seed=123, options={"low": -0.7, "high": 0.5})
        elif self.config.env.lower().find("mountaincar") >= 0:
            state, *_ = self.env.reset(seed=123, options={"x_init": numpy.pi/2, "y_init": 0.5})
        else:
            state, *_ = self.env.reset()
        return state

    def check_task(self, oracle_reward_file, oracle_prompt_file, random_reward_file, random_prompt_file, threshold = 1.0):
        def get_reward(reward_file_path, prompt_file_path):
            rewards = numpy.load(reward_file_path)
            prompts = numpy.load(prompt_file_path)
            episode_ranges = []
            current_episode_start = 0

            for i, prompt in enumerate(prompts):
                if prompt == 7:
                    episode_ranges.append((current_episode_start, i))
                    current_episode_start = i + 1

            if current_episode_start < len(prompts):
                episode_ranges.append((current_episode_start, len(prompts)))

            reward_sums = []
            for start, end in episode_ranges:
                episode_rewards = rewards[start:end]
                reward_sum = numpy.sum(episode_rewards)
                reward_sums.append(reward_sum)
            return reward_sums
        
        oracle_episode_reward = get_reward(oracle_reward_file, oracle_prompt_file)
        random_episode_reward = get_reward(random_reward_file, random_prompt_file)
        oracle_mean = numpy.mean(oracle_episode_reward)
        random_mean = numpy.mean(random_episode_reward)

        if abs(oracle_mean - random_mean) < 0.0001:
            print(f"Task transition has problem, reward all equal to 0.0, oracle mean = {oracle_mean}, random mean = {random_mean}")
            return False, oracle_mean, random_mean

        self.reward_nomalize_factor = 1 / (oracle_mean - random_mean)
        self.reward_nomalize_constant = - random_mean * self.reward_nomalize_factor
        oracle_episode_reward_normalized = self.reward_nomalize_factor * numpy.array(oracle_episode_reward) + self.reward_nomalize_constant 
        bar = numpy.var(oracle_episode_reward_normalized)
        if self.reward_nomalize_factor < 0:
            print("Random episode reward larger than oracle reward, ", oracle_mean - random_mean)
            return False, oracle_mean, random_mean
        elif bar < threshold:
            print("Reward variance < threshold, ", bar)
            return True, oracle_mean, random_mean
        else:
            print("Reward variance is too high, ", bar)
            return False, oracle_mean, random_mean
        
    def calculate_average_total_reward(self, reward_file_path, prompt_file_path, average = True):

        rewards = numpy.load(reward_file_path)
        prompts = numpy.load(prompt_file_path)

        episode_ranges = []
        current_episode_start = 0

        for i, prompt in enumerate(prompts):
            if prompt == 7:
                episode_ranges.append((current_episode_start, i))
                current_episode_start = i + 1

        if current_episode_start < len(prompts):
            episode_ranges.append((current_episode_start, len(prompts)))

        reward_sums = []
        reward = []
        for start, end in episode_ranges:
            episode_rewards = rewards[start:end]
            reward_sum = numpy.sum(episode_rewards)
            reward_sums.append(reward_sum)
            for value in rewards[start:end]:
                reward.append(value)

        total_reward_sum = sum(reward_sums)
        episode_num = len(episode_ranges)
        average_total_reward = total_reward_sum / episode_num if episode_num > 0 else 0
        if average:
            return average_total_reward
        else:
            return numpy.mean(reward)
    
    def nomalize_anymdp_reward(self, epoch_id):
        task_num = len(self.tasks)
        task_id = (epoch_id * self.world_size + self.rank) % task_num
        dirname = Path(self.config.data_root).parent
        oracle_path = os.path.join(dirname, "oracle", f"record-{task_id:06d}")
        random_path = os.path.join(dirname, "random", f"record-{task_id:06d}")
        oracle_rewards_path = os.path.join(oracle_path, 'rewards.npy')
        oracle_prompts_path = os.path.join(oracle_path, 'prompts.npy')
        random_rewards_path = os.path.join(random_path, 'rewards.npy')
        random_prompts_path = os.path.join(random_path, 'prompts.npy')

        pass_test, oracle_mean, random_mean = self.check_task(
            oracle_rewards_path, oracle_prompts_path, random_rewards_path, random_prompts_path, threshold=0.3)
        if not pass_test:
            return False

        
        # step
        oracle_step_mean = self.calculate_average_total_reward(oracle_rewards_path, oracle_prompts_path, average=False)
        random_step_mean = self.calculate_average_total_reward(random_rewards_path, random_prompts_path, average=False)
        self.step_reward_nomalize_factor = 1 / (oracle_step_mean - random_step_mean)
        self.step_reward_nomalize_constant = -random_step_mean * self.step_reward_nomalize_factor
        print("data avg step_reward_nomalize_factor = ", self.step_reward_nomalize_factor)
        print("data avg step_reward_nomalize_constant = ", self.step_reward_nomalize_constant)
        print("task id:", task_id, "oracle mean:", oracle_mean, "random mean:", random_mean, "factor:", self.reward_nomalize_factor, "constant:", self.reward_nomalize_constant)
        return True
    
    def get_exp_q(self):
        t_mat = get_final_transition(
            transition=self.env.transition_matrix,
            reset_states=self.env.reset_states,
            reset_triggers=self.env.reset_triggers)
        r_mat = get_final_reward(
            reward=self.env.reward_matrix,
            reset_triggers=self.env.reset_triggers,
        )   

        def get_q(t_mat, r_mat, is_greedy):
            max_iteration=5
            gamma = 0.99
            diff = 1.0
            ns, na, _ = r_mat.shape
            cur_vm = numpy.copy(numpy.zeros((ns, na)))
            iteration = 0
            while diff > 1.0e-4 and (
                    (max_iteration < 0) or 
                    (max_iteration > iteration and max_iteration > 1) or
                    (iteration < 1 and random.random() < max_iteration)):
                iteration += 1
                old_vm = numpy.copy(cur_vm)
                for s in range(ns):
                    for a in range(na):
                        exp_q = 0.0
                        for sn in range(ns):
                            if is_greedy: 
                                exp_q += t_mat[s,a,sn] * numpy.max(cur_vm[sn])
                            else:
                                exp_q += t_mat[s,a,sn] * numpy.mean(cur_vm[sn])
                        cur_vm[s,a] = numpy.dot(r_mat[s,a], t_mat[s,a]) + gamma * exp_q
                diff = numpy.sqrt(numpy.mean((old_vm - cur_vm)**2))
            return numpy.mean(cur_vm) 
        
        exp_q_opt = get_q(t_mat, r_mat,True)
        exp_q_random = get_q(t_mat, r_mat,False)
        self.step_reward_nomalize_factor = 1 / (exp_q_opt - exp_q_random)
        self.step_reward_nomalize_constant = - self.step_reward_nomalize_factor * exp_q_random
        print("exp q step_reward_nomalize_factor = ", self.step_reward_nomalize_factor)
        print("exp q step_reward_nomalize_constant = ", self.step_reward_nomalize_constant)



    def in_context_learn_from_teacher(self, epoch_id):
        # Task ID: retrieve the correpsonding teacher trajectory with task ID
        for folder in os.listdir(self.config.data_root):
            folder_path = os.path.join(self.config.data_root, folder)
            
            if self.mult_anymdp_task:
                task_num = len(self.tasks)
                task_id = (epoch_id * self.world_size + self.rank) % task_num
                folder = f"record-{task_id:06d}"
                folder_path = os.path.join(self.config.data_root, folder)
                print("task id:", task_id, "folder_path:", folder_path)

            if os.path.isdir(folder_path):
                states = numpy.load(os.path.join(folder_path, 'observations.npy'))
                prompts = numpy.load(os.path.join(folder_path, 'prompts.npy'))
                tags = numpy.load(os.path.join(folder_path, 'tags.npy'))
                actions = numpy.load(os.path.join(folder_path, 'actions_behavior.npy'))
                rewards = numpy.load(os.path.join(folder_path, 'rewards.npy'))
                states = states.astype(numpy.int32)
                prompts = prompts.astype(numpy.int32)
                tags = tags.astype(numpy.int32)
                actions = actions.astype(numpy.int32)
                rewards = rewards.astype(numpy.float32)
                segment_len = 1000
                for start in range(0, len(states), segment_len):
                    end = min(start + segment_len, len(states))
                    self.model.module.in_context_learn(
                        states[start:end],
                        prompts[start:end],
                        tags[start:end],
                        actions[start:end],
                        rewards[start:end],
                        single_batch=True,
                        single_step=False)
                if self.mult_anymdp_task:
                    print("Finish anymdp single task Learning.")
                    return
            else:
                log_warn(f"Folder {folder_path} does not exist.")
        print("Finish Learning.")

    def benchmark(self, epoch_id):
        supported_gym_env = ["lake", "lander", "mountaincar", "pendulum", "cliff"]
        # Load opt model
        if self.config.env.lower().find("anymdp") >= 0:
            model = AnyMDPSolverOpt(self.env)
            def benchmark_model(state):
                return model.policy(state)
            self.benchmark_opt_model = benchmark_model
        elif any(self.config.env.lower().find(name) == 0 for name in supported_gym_env):
            if self.config.run_benchmark.run_opt:
                model_classes = {'dqn': DQN, 'a24': A2C, 'td3': TD3, 'ppo': PPO}
                model_name = self.config.benchmark_model_name.lower()
                if model_name not in model_classes:
                    raise ValueError("Unknown policy type: {}".format())
                model = model_classes[model_name].load(f'{self.config.benchmark_model_save_path}/model/{model_name}.zip', env=self.env)
                def benchmark_model(state):
                    action, _ = model.predict(state)
                    return int(action)
                self.benchmark_opt_model = benchmark_model
        else:
            raise ValueError("Unsupported environment:", self.config.env)
        
        def run_online_rl():
            online_rl = OnlineRL(env=self.env, 
                                 env_name=self.config.env.lower(),
                                 model_name=self.config.benchmark_model_name.lower(),
                                 max_trails=self.config.max_trails,
                                 max_steps=self.config.max_steps,
                                 downsample_trail=self.config.downsample_trail)
            rew_stat, step_trail, success_rate = online_rl()
            ds_step_trail = downsample(step_trail, self.config.downsample_trail)
            ds_rewards = downsample(rew_stat, self.config.downsample_trail)
            ds_success = downsample(success_rate, self.config.downsample_trail)
            self.stat_online.gather(self.device,
                                step=ds_step_trail,
                                reward=ds_rewards,
                                success_rate=ds_success)


        # Function to run opt model or random model
        def run_benchmark(benchmark_model, logger_benchmark, stat_benchmark, epoch_id):
            rew_stat = []
            success_rate = []
            step_trail = []
            trail = 0
            total_steps = 0
            success_rate_f = 0.0

            if self.mult_anymdp_task:
                if not self.nomalize_anymdp_reward(epoch_id):
                    return
            
            while trail < self.max_trails:
                step = 0
                trail_reward = 0.0
                done = False
                new_state = self.reset_env()
                
                while not done:
                    action= benchmark_model(new_state)
                    new_state, new_reward, terminated, truncated, *_ = self.env.step(action)
                    if self.config.env.lower().find("anymdp") >= 0:
                        done = terminated
                    else:
                        if terminated or truncated:
                            done = True
                    shaped_reward = self.reward_shaping(done, terminated, new_reward)
                    trail_reward += new_reward

                    step += 1
                    if step > self.max_steps:
                        print("Reach max_steps, break trail.")
                        done = True
                    if done:
                        # success rate
                        succ_fail = self.is_success_fail(new_reward, trail_reward, terminated)
                        if trail + 1 < self.config.downsample_trail:
                            success_rate_f = (1-1/(trail+1)) * success_rate_f + succ_fail / (trail+1)
                        else:
                            success_rate_f = (1-1/self.config.downsample_trail) * success_rate_f + succ_fail / self.config.downsample_trail
                        
                        if self.mult_anymdp_task:
                            trail_reward = self.reward_nomalize_factor * trail_reward + self.reward_nomalize_constant
                        
                        rew_stat.append(trail_reward)
                        success_rate.append(success_rate_f)
                        step_trail.append(step)
                
                trail += 1
                total_steps += step
                logger_benchmark(trail,
                                total_steps,
                                step_trail[-1],
                                rew_stat[-1],
                                success_rate[-1])
            
            ds_step_trail = downsample(step_trail, self.config.downsample_trail)
            ds_rewards = downsample(rew_stat, self.config.downsample_trail)
            ds_success = downsample(success_rate, self.config.downsample_trail)

            stat_benchmark.gather(self.device,
                                step=ds_step_trail,
                                reward=ds_rewards,
                                success_rate=ds_success)
        
        if self.config.run_benchmark.run_opt:
            run_benchmark(self.benchmark_opt_model, self.logger_benchmark, self.stat_opt, epoch_id)
        
        if self.config.run_benchmark.run_online:
            run_online_rl()

        if self.config.run_benchmark.run_random:
            def random_model(state):
                return random.randint(0,self.config.action_clip - 1)
            run_benchmark(random_model, self.logger_random, self.stat_random, epoch_id)
        
    def in_context_learn_with_tag(self, trail_reward, step, trail_state_arr, trail_action_arr, trail_reward_arr):
        if self.config.env.lower().find("pendulum") >= 0:
            if trail_reward < -800:
                trail_tag_arr = numpy.full(len(trail_state_arr), 6, dtype=numpy.int32)
                trail_prompt_arr = numpy.full(len(trail_state_arr), 3, dtype=numpy.int32)
            elif trail_reward < -600:
                trail_tag_arr = numpy.full(len(trail_state_arr), 5, dtype=numpy.int32)
                trail_prompt_arr = numpy.full(len(trail_state_arr), 3, dtype=numpy.int32)
            elif trail_reward < -400:
                trail_tag_arr = numpy.full(len(trail_state_arr), 2, dtype=numpy.int32)
                trail_prompt_arr = numpy.full(len(trail_state_arr), 3, dtype=numpy.int32)
            else:
                trail_tag_arr = numpy.full(len(trail_state_arr), 3, dtype=numpy.int32)
                trail_prompt_arr = numpy.full(len(trail_state_arr), 3, dtype=numpy.int32)
        elif self.config.env.lower().find("mountaincar") >= 0:
            if step >=self.max_steps:
                trail_tag_arr = numpy.full(len(trail_state_arr), 7, dtype=numpy.int32)
                trail_prompt_arr = numpy.full(len(trail_state_arr), 3, dtype=numpy.int32)
            elif step >= 200:
                trail_tag_arr = numpy.full(len(trail_state_arr), 5, dtype=numpy.int32)
                trail_prompt_arr = numpy.full(len(trail_state_arr), 3, dtype=numpy.int32)
            else:
                trail_tag_arr = numpy.full(len(trail_state_arr), 3, dtype=numpy.int32)
                trail_prompt_arr = numpy.full(len(trail_state_arr), 3, dtype=numpy.int32)
        elif self.config.env.lower().find("cliff") >= 0:
            if trail_reward < -100:
                return
            elif trail_reward < -50:
                trail_tag_arr = numpy.full(len(trail_state_arr), 4, dtype=numpy.int32)
                trail_prompt_arr = numpy.full(len(trail_state_arr), 3, dtype=numpy.int32)
            elif trail_reward < -35:
                trail_tag_arr = numpy.full(len(trail_state_arr), 5, dtype=numpy.int32)
                trail_prompt_arr = numpy.full(len(trail_state_arr), 3, dtype=numpy.int32)
            else:
                trail_tag_arr = numpy.full(len(trail_state_arr), 3, dtype=numpy.int32)
                trail_prompt_arr = numpy.full(len(trail_state_arr), 3, dtype=numpy.int32)
        else:
            return

        self.model.module.in_context_learn(
                trail_state_arr,
                trail_prompt_arr,
                trail_tag_arr,
                trail_action_arr,
                trail_reward_arr,
                single_batch=True,
                single_step=False)

        
    def __call__(self, epoch_id):

        task_id = self.task_sampler(epoch_id=epoch_id)

        if self.mult_anymdp_task:
            skip_task = not self.nomalize_anymdp_reward(task_id)
            self.get_exp_q()
            if skip_task:
                print("Skip task: ", task_id)
                return

        if not self.config.run_icl:
            if self.config.run_benchmark.run_opt or self.config.run_benchmark.run_online or self.config.run_benchmark.run_random:
                print("Run Benchmark Only.")
                self.benchmark(epoch_id)
                return
            else:
                print("run_icl & run_benchmark both False, please check config.")
                return
        else:
            if self.config.run_benchmark.run_opt or self.config.run_benchmark.run_online or self.config.run_benchmark.run_random:
                print("Run Benchmark & ICL")
                self.benchmark(epoch_id)
            else:
                print("Run ICL Only.")
        # Start ICL
        obs_arr = []
        act_arr = []
        rew_arr = []
        rew_wo_done_arr = []
        frames = []
        
        reward_error = []
        state_error = []
        rew_stat = []
        success_rate = []
        step_trail = []
        success_rate_f = 0.0

        trail = 0
        total_step = 0
        pred_state_dist = None

        interactive_prompt = numpy.array([3]) # opt3 with gamma 0.994
        self.interactive_tag = numpy.array([7]) # Unknown, let model deside current policy quality 

        if self.config.learn_from_data:
            self.in_context_learn_from_teacher(epoch_id)

        while trail < self.max_trails or total_step < self.max_total_steps:
            step = 0
            done = False
            trail_reward = 0.0
            trail_reward_shaped = 0.0
            trail_obs_loss = 0.0
            trail_reward_loss = 0.0
            trail_state_arr = []
            trail_action_arr = []
            trail_reward_arr = []

            previous_state = self.reset_env()
            trail_state_arr.append(previous_state)
            obs_arr.append(previous_state)
            if(pred_state_dist is not None):
                trail_obs_loss += -numpy.log(pred_state_dist[int(previous_state)].item())
            temp = self._scheduler(total_step)
            while not done:
                # Generate action, world model prediction
                pred_state_dist, action, pred_reward = self.model.module.generate(
                    previous_state,
                    interactive_prompt,
                    self.interactive_tag,
                    temp=temp,
                    need_numpy=True,
                    single_batch=True,
                    future_prediction=True)
                env_action = action % self.config.action_clip 
                # Interact with environment         
                new_state, new_reward, terminated, truncated, *_ = self.env.step(env_action)
                if self.config.env.lower().find("anymdp") >= 0:
                        done = terminated
                else:
                    if terminated or truncated:
                        done = True
                # Reward shaping
                shaped_reward = self.reward_shaping(done, terminated, new_reward)

                # collect data
                trail_action_arr.append(action)
                trail_reward_arr.append(shaped_reward)
                act_arr.append(action)
                rew_arr.append(new_reward)
                if total_step < self.max_total_steps:
                    rew_wo_done_arr.append(new_reward)
                # collect gif frame
                if self.config.save_gif and trail % self.config.save_gif_gap == 0: 
                    if self.config.env.lower().find("anymdp") < 0:
                        frames.extend(self.env.render())
                    else:
                        frames.append((previous_state, action, new_reward, new_state, done>0.1))


                # start learning     
                self.model.module.in_context_learn(
                    previous_state,
                    interactive_prompt,
                    self.interactive_tag,
                    action,
                    shaped_reward)

                trail_state_arr.append(new_state)
                obs_arr.append(new_state) 
                previous_state = new_state

                trail_obs_loss += -numpy.log(pred_state_dist[int(new_state)].item())
                trail_reward += new_reward
                trail_reward_shaped += shaped_reward
                trail_reward_loss += (shaped_reward - pred_reward) ** 2

                step += 1
                if(step > self.max_steps):
                    print("Reach max_steps, break trail.")
                    done = True
                if(done):
                    trail_action_arr.append(self.action_dim)
                    trail_reward_arr.append(0.0)
                    act_arr.append(self.action_dim)
                    rew_arr.append(0.0)
                    self.model.module.in_context_learn(
                        new_state,
                        interactive_prompt,
                        self.interactive_tag,
                        self.action_dim,
                        0.0)
                    if self.config.use_dym_tag:
                        self.in_context_learn_with_tag(trail_reward, step, trail_state_arr, trail_action_arr, trail_reward_arr)
                    # success rate
                    succ_fail = self.is_success_fail(new_reward, trail_reward, terminated)
                    if trail + 1 < self.config.downsample_trail:
                        success_rate_f = (1-1/(trail+1)) * success_rate_f + succ_fail / (trail+1)
                    else:
                        success_rate_f = (1-1/self.config.downsample_trail) * success_rate_f + succ_fail / self.config.downsample_trail
                    
                    if self.mult_anymdp_task:
                        trail_reward = self.reward_nomalize_factor * trail_reward + self.reward_nomalize_constant
                    
                    rew_stat.append(trail_reward)
                    state_error.append(trail_obs_loss / step)
                    reward_error.append(trail_reward_loss / step)
                    success_rate.append(success_rate_f)
                    step_trail.append(step)

            trail += 1
            total_step += step
            self.logger(trail,
                        total_step,
                        step_trail[-1],
                        rew_stat[-1], 
                        state_error[-1], 
                        reward_error[-1],
                        success_rate[-1])

        # Save gif
        if self.config.save_gif and self.config.env.lower().find("anymdp") < 0:
            gif_path = f'{self.config.output}/gym.gif'
            imageio.mimsave(gif_path, [numpy.array(frame) for frame in frames], fps=30)
        elif self.config.save_gif and self.config.env.lower().find("anymdp") >= 0:
            print("start drawing")
            self.drawer.draw(frames)
            print("finish drawing")

        # Save step reward
        if self.max_total_steps > 1.0:
            array_to_save = numpy.array(rew_wo_done_arr)
            array_to_save = array_to_save * self.step_reward_nomalize_factor + self.step_reward_nomalize_constant
            file_path = f'{self.config.output}/step_reward/'
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            save_path = f'{file_path}/step_reward_{task_id}.npy'
            numpy.save(save_path, array_to_save)
            
        
        ds_state_err = downsample(state_error, self.config.downsample_trail)
        ds_reward_err = downsample(reward_error, self.config.downsample_trail)
        ds_rewards = downsample(rew_stat, self.config.downsample_trail)
        ds_success = downsample(success_rate, self.config.downsample_trail)
        ds_step_trail = downsample(step_trail, self.config.downsample_trail)

        self.stat.gather(self.device,
                         step=ds_step_trail,
                         reward=ds_rewards,
                         state_prediction=ds_state_err,
                         reward_prediction=ds_reward_err,
                         success_rate = ds_success)
    
    def save_results(self, results, prefix):
        if self.config.has_attr("output"):
            if not os.path.exists(self.config.output):
                os.makedirs(self.config.output)
            
            for key_name in results:
                res_text = string_mean_var(self.config.downsample_trail, results[key_name])
                file_path = f'{self.config.output}/{prefix}_{key_name}.txt'
                
                if os.path.exists(file_path):
                    os.remove(file_path)
                
                with open(file_path, 'w') as f_model:
                    f_model.write(res_text)

    def postprocess(self):
        if self.config.run_icl:
            # Final Result
            final_results = self.stat()
            self.logger("Final_Result",
                        self.config.max_trails,
                        final_results['step']['mean'],
                        final_results['reward']['mean'],
                        final_results['state_prediction']['mean'],
                        final_results['reward_prediction']['mean'],
                        final_results['success_rate']['mean'])
            self.save_results(final_results, "result")

        # Benchmark Result
        if self.config.run_benchmark.run_opt:
            # Benchmark Result
            benchmark_results = self.stat_opt()
            self.logger_benchmark("Benchmark_Result",
                                self.config.max_trails,
                                benchmark_results['step']['mean'],
                                benchmark_results['reward']['mean'],
                                benchmark_results['success_rate']['mean'])
            self.save_results(benchmark_results, "benchmark_result")
        if self.config.run_benchmark.run_online:
            # Online Result
            online_results = self.stat_online()
            self.logger_benchmark("Online_Result",
                                self.config.max_trails,
                                online_results['step']['mean'],
                                online_results['reward']['mean'],
                                online_results['success_rate']['mean'])
            self.save_results(online_results, "online_result")
        if self.config.run_benchmark.run_random:
            # Random Result
            random_results = self.stat_random()
            self.logger_random("Random_Result",
                                self.config.max_trails,
                                random_results['step']['mean'],
                                random_results['reward']['mean'],
                                random_results['success_rate']['mean'])
            self.save_results(random_results, "random_result")

class MultiAgentGenerator(OmniRLGenerator):

    # Mult-Agent Generator
    def preprocess(self):
        super().preprocess()

        logger_keys = ["step", "reward", "state_prediction", "reward_prediction", "success_rate"]

        self.stats = []
        self.loggers = []
        for i in range(self.agent_num):
            stat = DistStatistics(*logger_keys)
            logger = Logger(f"trail_idx_{i}",
                            "total_steps",
                            *logger_keys,
                            on=self.main,
                            use_tensorboard=False)
            self.stats.append(stat)
            self.loggers.append(logger)

    def reward_shaping(self, done, reward, last_obs, obs):
        if(self.config.env.lower().find("switch") >=0):
            for i in range(len(done)):
                goal_pos = self.env.final_agent_pos[i]
                last_pos = last_obs[self.agent_num + i]
                current_pos = obs[self.agent_num + i]

                def distance_to_goal(pos, goal_pos):
                    return abs(goal_pos[0]-pos[0]) + abs(goal_pos[1]-pos[1])
                
                if distance_to_goal(current_pos,goal_pos) < distance_to_goal(last_pos,goal_pos):
                    rew = 0
                else:
                    rew = -0.005

                if not done[i]:
                    reward[i] = rew
                else:
                    reward[i] = 1.0 if reward[i] > 0.0 else rew
                    
        return reward
        
    def __call__(self, epoch_id):

        task_id = self.task_sampler(epoch_id=epoch_id)

        if not self.config.run_icl:
            if self.config.run_benchmark.run_opt or self.config.run_benchmark.run_online or self.config.run_benchmark.run_random:
                print("Run Benchmark Only.")
                self.benchmark(epoch_id)
                return
            else:
                print("run_icl & run_benchmark both False, please check config.")
                return
        else:
            if self.config.run_benchmark.run_opt or self.config.run_benchmark.run_online or self.config.run_benchmark.run_random:
                print("Run Benchmark & ICL")
                self.benchmark(epoch_id)
            else:
                print("Run ICL Only.")
        # Start ICL
        obs_arrs = [[] for _ in range(self.agent_num)]
        act_arrs = [[] for _ in range(self.agent_num)]
        rew_arrs = [[] for _ in range(self.agent_num)]

        reward_errors = [[] for _ in range(self.agent_num)]
        state_errors = [[] for _ in range(self.agent_num)]
        rew_stats = [[] for _ in range(self.agent_num)]
        success_rates = [[] for _ in range(self.agent_num)]
        step_trails = [[] for _ in range(self.agent_num)]
        success_rate_fs = [0.0] * self.agent_num
        pred_state_dists = [None] * self.agent_num
        pred_reward_dists = [None] * self.agent_num
        total_steps = [0] * self.agent_num

        frames = []
        trail = 0

        interactive_prompt = numpy.array([3]) # opt3 with gamma 0.994
        self.interactive_tag = numpy.array([7]) # Unknown, let model deside current policy quality 

        #if self.config.learn_from_data:
        #    self.in_context_learn_from_teacher(epoch_id)

        while trail < self.max_trails:
            agents_info = {i: {
                'step': 0,
                'done': False,
                'stop_learning': False,
                'trail_reward': 0.0,
                'trail_reward_shaped': 0.0,
                'trail_obs_loss': 0.0,
                'trail_reward_loss': 0.0,
                'trail_state_arr': [],
                'trail_action_arr': [],
                'trail_reward_arr': []
            } for i in range(self.agent_num)}

            previous_state = self.env.reset()

            for agent_index in range(self.agent_num):
                agents_info[agent_index]['trail_state_arr'].append(previous_state[agent_index])
                obs_arrs[agent_index].append(previous_state[agent_index])

            temp = self._scheduler(max(total_steps))
            while not all(agent_info['done'] for agent_info in agents_info.values()):
                # Generate action, world model prediction
                env_action=[]
                agent_action=[]
                for agent_index in range(self.agent_num):
                    if not agents_info[agent_index]['done']:
                        pred_state_dists[agent_index], action, pred_reward_dists[agent_index] = self.model[agent_index].module.generate(
                            previous_state[agent_index],
                            interactive_prompt,
                            self.interactive_tag,
                            temp=temp,
                            need_numpy=True,
                            single_batch=True,
                            future_prediction=True
                        )
                        env_action.append(action % self.config.action_clip)
                        agent_action.append(action)
                    else:
                        env_action.append(4) #ToDo, Not for every multi-agent env
                        agent_action.append(4)
                # Interact with environment         
                new_state, new_reward, done, *_ = self.env.step(env_action)
                # Reward shaping
                shaped_reward = self.reward_shaping(done, new_reward, previous_state, new_state)
                # Collect gif frame
                if self.config.save_gif and trail % self.config.save_gif_gap == 0: 
                    if self.config.env.lower().find("anymdp") < 0:
                        frames.extend(self.env.render())

                for agent_index in range(self.agent_num):
                    agents_info[agent_index]['done'] = done[agent_index]
                    if not agents_info[agent_index]['stop_learning']:
                        agents_info[agent_index]['trail_action_arr'].append(agent_action[agent_index])
                        agents_info[agent_index]['trail_reward_arr'].append(shaped_reward[agent_index])
                        agents_info[agent_index]['trail_state_arr'].append(new_state[agent_index])
                        obs_arrs[agent_index].append(new_state[agent_index])
                        act_arrs[agent_index].append(agent_action[agent_index])
                        rew_arrs[agent_index].append(new_reward[agent_index])
                        agents_info[agent_index]['trail_obs_loss'] += -numpy.log(pred_state_dists[agent_index][int(new_state[agent_index])].item())
                        agents_info[agent_index]['trail_reward_loss'] += (shaped_reward[agent_index] - pred_reward_dists[agent_index]) ** 2
                        agents_info[agent_index]['trail_reward'] += new_reward[agent_index]
                        agents_info[agent_index]['trail_reward_shaped'] += shaped_reward[agent_index]
                        
                        self.model[agent_index].module.in_context_learn(
                            previous_state[0],
                            interactive_prompt,
                            self.interactive_tag,
                            agent_action[agent_index],
                            shaped_reward[agent_index])

                        agents_info[agent_index]['step'] += 1
                        if agents_info[agent_index]['step'] > self.max_steps:
                            print(f"Agent {agent_index + 1} Reach max_steps, break trail.")
                            agents_info[agent_index]['done'] = True
                        if agents_info[agent_index]['done']:
                            agents_info[agent_index]['trail_action_arr'].append(self.action_dim)
                            agents_info[agent_index]['trail_reward_arr'].append(0.0)
                            act_arrs[agent_index].append(self.action_dim)
                            rew_arrs[agent_index].append(0.0)
                            self.model[agent_index].module.in_context_learn(
                                new_state[agent_index],
                                interactive_prompt,
                                self.interactive_tag,
                                self.action_dim,
                                0.0)
                            # Todo: Calucute success rate if needed
                            #if self.config.use_dym_tag:
                            #    self.in_context_learn_with_tag(trail_reward, step, trail_state_arr, trail_action_arr, trail_reward_arr)
                            rew_stats[agent_index].append(agents_info[agent_index]['trail_reward_shaped'])
                            state_errors[agent_index].append(agents_info[agent_index]['trail_obs_loss'] / agents_info[agent_index]['step'])
                            reward_errors[agent_index].append(agents_info[agent_index]['trail_reward_loss'] / agents_info[agent_index]['step'])
                            success_rates[agent_index].append(success_rate_fs[agent_index])
                            step_trails[agent_index].append(agents_info[agent_index]['step'])
                            agents_info[agent_index]['stop_learning'] = True
                 
                previous_state = new_state    

            trail += 1
            for agent_index in range(self.agent_num):
                total_steps[agent_index] += agents_info[agent_index]['step']
                self.loggers[agent_index](trail,
                                        total_steps[agent_index],
                                        step_trails[agent_index][-1],
                                        rew_stats[agent_index][-1], 
                                        state_errors[agent_index][-1], 
                                        reward_errors[agent_index][-1],
                                        success_rates[agent_index][-1])

        # Save gif
        if self.config.save_gif and self.config.env.lower().find("anymdp") < 0:
            gif_path = f'{self.config.output}/gym.gif'
            imageio.mimsave(gif_path, [numpy.array(frame) for frame in frames], fps=30)
        
        for agent_index in range(self.agent_num):
            ds_state_err = downsample(state_errors[agent_index], self.config.downsample_trail)
            ds_reward_err = downsample(reward_errors[agent_index], self.config.downsample_trail)
            ds_rewards = downsample(rew_stats[agent_index], self.config.downsample_trail)
            ds_success = downsample(success_rates[agent_index], self.config.downsample_trail)
            ds_step_trail = downsample(step_trails[agent_index], self.config.downsample_trail)
            self.stats[agent_index].gather(self.device,
                            step=ds_step_trail,
                            reward=ds_rewards,
                            state_prediction=ds_state_err,
                            reward_prediction=ds_reward_err,
                            success_rate = ds_success)

    def postprocess(self):
        if self.config.run_icl:
            for i in range(self.agent_num):
                final_results = self.stats[i]()
                self.loggers[i](f"Agent {i + 1} Final_Result",
                            self.config.max_trails,
                            final_results['step']['mean'],
                            final_results['reward']['mean'],
                            final_results['state_prediction']['mean'],
                            final_results['reward_prediction']['mean'],
                            final_results['success_rate']['mean'])
                self.save_results(final_results, f"Agent{i + 1}_result")