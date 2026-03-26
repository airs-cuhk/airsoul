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

    def compute(self, seq_arr, label_arr,
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
        for sub_idx, seq, label in segment_iterator(self.config.seq_len, self.config.seg_len, self.device, seq_arr, label_arr):
            loss = self.model.module.sequential_loss(
                    seq, 
                    label, 
                    use_loss_weight=self.is_training,
                    update_memory=True,
                    reduce_dim=self.reduce)
            losses.append(loss)
            obs_pre_step = loss["count_s"]/loss["count_p"]
            agent_pre_step = loss["count_a"]/loss["count_p"]
            if(self.is_training):
                syn_loss = (self.config.lossweight_worldmodel_states * loss["wm_obs"] / obs_pre_step
                        + self.config.lossweight_worldmodel_actions * loss["wm_agent"] / agent_pre_step
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
                        epoch=epoch_id,
                        iteration=batch_id)
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
            counts = torch.cat([loss["count_p"] for loss in losses], dim=1)

            bsz = loss_wm_s.shape[0]

            loss_wm_s = downsample(loss_wm_s, self.downsample_length * obs_pre_step)
            loss_wm_a = downsample(loss_wm_a, self.downsample_length * obs_pre_step)
            loss_wm_r = downsample(loss_wm_r, self.downsample_length)
            loss_pm = downsample(loss_pm, self.downsample_length)
            loss_ent = downsample(loss_ent, self.downsample_length)
            counts = downsample(counts, self.downsample_length)

            for i in range(bsz):
                self.stat.gather(self.device,
                        validation_state_pred=loss_wm_s[i], 
                        validation_other_agent_pred=loss_wm_a[i],
                        validation_reward_pred=loss_wm_r[i], 
                        validation_policy=loss_pm[i],
                        validation_entropy=loss_ent[i],
                        count=counts[i])
    
    def epoch_end(self, epoch_id):
        if(not self.is_training):
            stat_res = self.stat()
            if(self.logger is not None):
                self.logger(stat_res["validation_state_pred"]["mean"], 
                        stat_res["validation_other_agent_pred"]["mean"],
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

class HVACGenerator(GeneratorBase):
    def preprocess(self):
        if(self.config.env.lower().find("hvac") >= 0):
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
            directory="",
            time_step=5000,
            max_obs_num=self.config.max_obs_num,
            max_agent_num=self.config.max_agent_num,
            tag_num=self.config.tag_num,
            value_num=self.config.value_num,
            resolution=self.config.resolution,
            vocab_size=self.config.vocab_size,
            verbose=False
        )
        self.vocabularize = self.dataset.vocabularize
    
    def epoch_end(self, epoch_id):
        pass

    def task_sampler_anyhvacv2(self, epoch_id=0):
        task_id = None
        if(self.tasks is None):
            task = HVACTaskSampler(control_type='Temperature')
        else:
            task_num = len(self.tasks)
            task_id = (epoch_id * self.world_size + self.rank) % task_num
            task = self.tasks[task_id]
        self.env.set_task(task)
        return task_id
    
    def in_context_learn_from_teacher(self, epoch_id):
        pass # TODO

    def build_up_vocab_seq_in_batch(self, obs_sensor,  obs_agent, current_batch_seq=None, 
                                    action=None, reward=None, reset=False):
        if current_batch_seq is None:
            current_batch_seq = []
            # [num, value] -> [num, 1, value]
            obs_sensor = obs_sensor.unsqueeze(1)
            obs_agent = obs_agent.unsqueeze(1)
            obs_sensor_vocabularize = self.vocabularize('value', obs_sensor).squeeze()
            obs_agent_vocabularize = self.vocabularize('value', obs_agent).squeeze()
            for agent_id in range(self.num_agents):
                current_agent_seq = []
                # 1, Related sensor idx and value
                for related_obs in self.related_sensor[agent_id]:
                    current_agent_seq.append(self.vocabularize('obs_id', related_obs))
                    current_agent_seq.append(obs_sensor_vocabularize[related_obs])
                    # current_agent_seq.append(self.vocabularize('value', obs_sensor[related_obs]))
                # 2, Related agent idx and value
                for related_agent in self.related_agent[agent_id]:
                    current_agent_seq.append(self.vocabularize('agent_id', related_agent))
                    current_agent_seq.append(obs_agent_vocabularize[related_agent])
                    # current_agent_seq.append(self.vocabularize('value', obs_agent[related_agent]))
                # 3, Tag
                current_agent_seq.append(self.vocabularize('special_token', 'idx_tag'))
                current_agent_seq.append(self.vocabularize('tag_value', self.interactive_tag))
                # 4, Self action flag
                current_agent_seq.append(self.vocabularize('special_token', 'idx_a_self'))
                current_batch_seq.append(current_agent_seq)
            return current_batch_seq
        else:
            for agent_id in range(self.num_agents):
                # 5, Self action value
                current_batch_seq[agent_id].append(self.vocabularize('value', action[agent_id]))
                # 6, Reward idx and value
                current_batch_seq[agent_id].append(self.vocabularize('special_token', 'idx_reward'))
                current_batch_seq[agent_id].append(self.vocabularize('value', reward))
                # 7, End
                if reset:
                    current_batch_seq[agent_id].append(self.vocabularize('special_token', 'idx_reset_env'))
                else:
                    current_batch_seq[agent_id].append(self.vocabularize('special_token', 'idx_end_timestep'))
            return current_batch_seq
    
    def build_up_env_action(self, action_in_vocab):
        if self.previous_action is None:
            self.previous_action = numpy.full((self.num_agents, 1, 2), [0.0, 0.0])
        # [num, value] -> [num, 1, value]
        action_in_vocab.unsqueeze(1)
        action_in_value = self.vocabularize('action_vocab',...)

    def __call__(self, epoch_id):

        task_id = self.task_sampler(epoch_id=epoch_id)

        obs_sensor_array = []
        obs_action_array = []
        rew_wo_done_array = []

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

            previous_state = self.env.reset()
