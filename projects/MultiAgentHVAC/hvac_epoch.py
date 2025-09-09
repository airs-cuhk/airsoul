import os
import torch
import numpy
import pickle

from airsoul.dataloader import segment_iterator
from airsoul.utils import Logger, log_progress, log_debug, log_warn, log_fatal
from airsoul.utils import DistStatistics, downsample
from airsoul.utils import EpochManager, GeneratorBase, Logger
from airsoul.dataloader import MultiAgentLoadDateSet, MultiAgentDataSetVetorized

from xenoverse.anyhvacv2.anyhvac_sampler import HVACTaskSampler
from xenoverse.anyhvacv2.anyhvac_env_vis import HVACEnvVisible, HVACEnv
from xenoverse.anyhvacv2.anyhvac_solver import HVACSolverGTPID
from env_wapper import HVACEnvWrapper

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
                        "validation_reward_pred", 
                        "validation_policy",
                        "validation_entropy"]
            self.stat_w = DistStatistics()
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
            loss = self.model.module.sequential_loss(
                    seq, 
                    label, 
                    use_loss_weight=self.is_training,
                    update_memory=True,
                    reduce_dim=self.reduce)
            losses.append(loss)
            # obs_pre_step = loss["count_s"]/loss["count_p"]
            # agent_pre_step = loss["count_a"]/loss["count_p"]
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
            counts_w = torch.cat([loss["count_s"] for loss in losses], dim=1)
            counts_p = torch.cat([loss["count_p"] for loss in losses], dim=1)

            bsz = loss_wm_s.shape[0]
            self.obs_pre_step = int(torch.sum(counts_w) / torch.sum(counts_p))
            
            def extract_nonzero_per_batch(tensor):
                results = []
                for i in range(tensor.size(0)):
                    batch_data = tensor[i]
                    mask = batch_data != 0
                    non_zero = batch_data[mask]
                    results.append(non_zero)
                max_len = max(t.numel() for t in results)
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
            counts_w = extract_nonzero_per_batch(counts_w)
            counts_p = extract_nonzero_per_batch(counts_p)

            loss_wm_s = downsample(loss_wm_s, self.downsample_length * self.obs_pre_step)
            loss_wm_a = downsample(loss_wm_a, self.downsample_length * self.obs_pre_step)
            loss_wm_r = downsample(loss_wm_r, self.downsample_length)
            loss_pm = downsample(loss_pm, self.downsample_length)
            loss_ent = downsample(loss_ent, self.downsample_length)
            counts_w = downsample(counts_w, self.downsample_length * self.obs_pre_step)
            counts_p = downsample(counts_p, self.downsample_length)

            for i in range(bsz):
                self.stat_w.gather(self.device,
                        validation_state_pred=loss_wm_s[i], 
                        validation_other_agent_pred=loss_wm_a[i],
                        count=counts_w[i])
                self.stat_p.gather(self.device,
                        validation_reward_pred=loss_wm_r[i], 
                        validation_policy=loss_pm[i],
                        validation_entropy=loss_ent[i],
                        count=counts_p[i])
    
    def epoch_end(self, epoch_id):
        if(not self.is_training):
            stat_res_w = self.stat_w()
            stat_res_p = self.stat_p()
            if(self.logger is not None):
                self.logger(stat_res_w["validation_state_pred"]["mean"], 
                        stat_res_w["validation_other_agent_pred"]["mean"],
                        stat_res_p["validation_reward_pred"]["mean"], 
                        stat_res_p["validation_policy"]["mean"],
                        stat_res_p["validation_entropy"]["mean"],
                        epoch=epoch_id)
            if(self.extra_info is not None):
                if(self.extra_info.lower() == 'validate' and self.main):
                    if not os.path.exists(self.config.output):
                        os.makedirs(self.config.output)
                    for key_name in stat_res_w:
                        res_text = string_mean_var(self.downsample_length * self.obs_pre_step, stat_res_w[key_name])
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
            self.env = HVACEnvWrapper()
            self.task_sampler = self.task_sampler_anyhvacv2
        else:
            log_fatal("Unsupported environment:", self.config.env)

        if(self.config.has_attr("task_file")):
            with open(self.config.task_file, 'rb') as fr:
                self.tasks = pickle.load(fr)
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
            tag_num=self.config.vocab.tag_num,
            value_num=self.config.vocab.value_num,
            resolution=self.config.vocab.resolution,
            vocab_size=self.config.vocab.vocab_size,
            verbose=False
        )
        self.vocabularize = self.dataset.vocabularize
        self.use_diff_action = True
    
    def epoch_end(self, epoch_id):
        pass

    def task_sampler_anyhvacv2(self, epoch_id=0):
        task_id = None
        if(self.tasks is None):
            task = HVACTaskSampler(control_type='Temperature')
        else:
            #task_num = len(self.tasks)
            #task_id = (epoch_id * self.world_size + self.rank) % task_num
            #task = self.tasks[task_id]
            task = self.tasks
        self.env.set_task(task)

        knn = 3
        obs_graph = self.env._create_cooler_sensor_graph(knn)
        agent_graph = self.env._create_cooler_cooler_graph(knn)
        self.agent_num, self.sensor_num = obs_graph.shape
        self.related_sensor = numpy.zeros((self.agent_num, knn), dtype=numpy.int32)
        self.related_agent = numpy.zeros((self.agent_num, knn), dtype=numpy.int32)
        for i in range(self.agent_num):
            sensor_indices = numpy.where(obs_graph[i] == 1)[0]
            agent_indices = numpy.where(agent_graph[i] == 1)[0]
            self.related_sensor[i] = sensor_indices
            self.related_agent[i] = agent_indices

        self.env._create_agent_target_temperture()

        return task_id
    
    def in_context_learn_from_teacher(self, epoch_id):
        pass # TODO

    def build_up_vocab_seq_in_batch(self, obs_sensor, obs_agent, current_batch_seq=None, 
                                    action=None, reward=None, reset=False, use_diff_action=True, use_relative_idx=True):
        # [num_agent, time, value]
        # obs_agent and action should contain [:,t-1:t,:] two timestep if t>0.
        if current_batch_seq is None:
            current_batch_seq = []
            # [num, value] -> [num, 1, value]
            obs_sensor = obs_sensor[:,numpy.newaxis,numpy.newaxis]
            obs_sensor_vocabularize = self.vocabularize('value', obs_sensor).squeeze()
            obs_agent_vocabularize = self.vocabularize('value', obs_agent)[:,-1:,:].squeeze()
            for agent_id in range(self.agent_num):
                current_agent_seq = []
                # 1, Related sensor idx and value
                for i, related_obs in enumerate(self.related_sensor[agent_id]):
                    if use_relative_idx:
                        current_agent_seq.append(self.vocabularize('obs_id', i))
                    else:
                        current_agent_seq.append(self.vocabularize('obs_id', related_obs))
                    current_agent_seq.append(obs_sensor_vocabularize[related_obs])
                    # current_agent_seq.append(self.vocabularize('value', obs_sensor[related_obs]))
                # 2, Related agent idx and value
                for i, related_agent in enumerate(self.related_agent[agent_id]):
                    if use_relative_idx:
                        current_agent_seq.append(self.vocabularize('agent_id', i))
                    else:
                        current_agent_seq.append(self.vocabularize('agent_id', related_agent))
                    current_agent_seq.append(obs_agent_vocabularize[related_agent])
                    # current_agent_seq.append(self.vocabularize('value', obs_agent[related_agent]))
                # 3, Tag
                current_agent_seq.append(self.vocabularize('special_token', 'idx_tag'))
                current_agent_seq.append(self.vocabularize('tag_value', self.interactive_tag))
                # 4, Self action flag
                current_agent_seq.append(self.vocabularize('special_token', 'idx_a_self'))
                current_batch_seq.append(current_agent_seq)
            current_batch_seq = [[int(x) for x in lst] for lst in current_batch_seq]
            return current_batch_seq
        else:
            agent_action_vocabularize = self.vocabularize('value', action)[:,-1,:].squeeze()
            reward_idx_vocabularize = self.vocabularize('special_token', 'idx_reward')
            reward_vocabularize = self.vocabularize('value', reward)
            timestep_end_vocabularize = self.vocabularize('special_token', 
                                             'idx_reset_env' if reset else 'idx_end_timestep')
            to_add = numpy.array([
                agent_action_vocabularize,                              # 5, Self action value
                numpy.full(self.agent_num, reward_idx_vocabularize),   # 6, Reward idx and value
                numpy.full(self.agent_num, reward_vocabularize),  
                numpy.full(self.agent_num, timestep_end_vocabularize)  # 7, End
            ], dtype=object).T  # (num_agents, 4)
            for i, seq in enumerate(current_batch_seq):
                seq.extend(to_add[i])
            current_batch_seq = [[int(x) for x in lst] for lst in current_batch_seq]
            return current_batch_seq
    
    def build_up_env_action(self, action_in_vocab, action_value_previous=None, use_diff_action=True):
        # Shape of action_in_vocab: [num, 1, 1]
        # Shape of action_value_previous: [num, 1, 2], action: (on/off, temp)

        action_in_value = self.vocabularize('action_vocab', 
                                            action_in_vocab, 
                                            value_previous=action_value_previous,
                                            use_diff_action=use_diff_action) # [num, 2]
        # Convert actual temperature settings to normalized values 
        action_temp = action_in_value.squeeze()[:,1] + self.env.agent_target_temp
        action_temp = (action_temp - self.env.lower_bound) / (self.env.upper_bound - self.env.lower_bound)
        action_temp = numpy.clip(action_temp, 0.0, 1.0)
        action_switch = action_in_value.squeeze()[:,0] 
        action_switch = numpy.where(action_switch > 0.5, 1.0, 0.0)
        n_coolers = action_temp.shape[0]
        env_action = numpy.zeros(2*n_coolers, dtype=numpy.float32)
        env_action[:n_coolers] = action_switch
        env_action[n_coolers:n_coolers*2] = action_temp
        
        return env_action, action_in_value

    def __call__(self, epoch_id):

        task_id = self.task_sampler(epoch_id=epoch_id)

        obs_sensor_array = []
        obs_action_array = []
        rew_array = []

        obs_sensor_error = []
        obs_action_error = []
        rew_stat = []

        trail = 0
        total_step = 0

        self.interactive_tag = 5 # tag_num = 6

        if self.config.learn_from_data:
            self.in_context_learn_from_teacher(epoch_id)

        while trail < self.max_trails or total_step < self.max_total_steps:
            step = 0
            done = False
            trail_reward = 0.0
            trail_obs_sensor_loss = 0.0
            trail_obs_action_loss = 0.0
            trail_reward_loss = 0.0

            obs = self.env.reset()[0]
            previous_state = self.env._compute_temperature_deviation(obs)
            previous_action = numpy.zeros((self.agent_num, 1, 2), dtype=numpy.float32)
            while not done:
                
                print("obs: ", obs[:len(self.env.sensors)])

                vocab_seq_batch = self.build_up_vocab_seq_in_batch(previous_state,
                                                                   previous_action,
                                                                   use_diff_action=self.use_diff_action)
                world_model_obs, world_model_action, action = self.model.module.generate(
                    vocab_seq_batch,
                    single_batch=False,
                    reward_prediction=False)
                env_action, current_action = self.build_up_env_action(action, 
                                                              action_value_previous=previous_action[:,-1:,:],
                                                              use_diff_action=self.use_diff_action)   
                
                obs, reward, terminated, truncated, info = self.env.step(env_action)
                obs_sensor_array.append(obs)
                rew_array.append(reward)

                obs_deviation = self.env._compute_temperature_deviation(obs)
                combined_action = numpy.concatenate((previous_action, current_action), axis=1)[:,-2:,:]
                done = terminated or truncated
                vocab_seq_batch = self.build_up_vocab_seq_in_batch(previous_state,
                                                                   previous_action,
                                                                   current_batch_seq=vocab_seq_batch,
                                                                   action=combined_action,
                                                                   reward=numpy.array(reward).reshape(1, 1, 1),
                                                                   reset=done,
                                                                   use_diff_action=self.use_diff_action
                                                                   )
                cache = self.model.module.incontext_learn(vocab_seq_batch, need_cache=False)
                previous_state = obs_deviation

                
                print("current_action: ", current_action)
                print("reward: ", reward)
                print("done: ", done)