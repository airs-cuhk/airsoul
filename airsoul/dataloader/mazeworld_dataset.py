import os
import sys
import torch
import numpy as np
from numpy import random
from torch.utils.data import DataLoader, Dataset
# cut the observation, action, position, reward, BEV, agent, target
import math


def expend_data(
    observations,
    actions_behavior_id,
    actions_behavior_val,
    actions_label_id,
    actions_label_val,
    rewards,
    command,
    actions_behavior_prior,
    percentage=2,
):

    def expend_delta(n_start, n_end, percentage):
        fractional_part, integer_part = math.modf(percentage)
        _split = []
        for i in range(int(integer_part)):
            _split.append([n_start, n_end])
        delta = int((n_end - n_start) * fractional_part)
        _split.append([n_end - delta, n_end])
        return _split

    split_id = []  # select the start and end of the data
    n_start = 0
    n_end = 0

    flag = False  # flag为True: 下一个end没用
    for i in range(len(observations)):
        if actions_behavior_id[i] == 16:
            if flag:
                flag = False
                continue
            flag = True

            n_end = i+1
            split_id.extend(expend_delta(n_start, n_end, percentage))
            n_start = n_end

    # print("split_id",split_id)
    (
        _obs_arr,
        _bact_id_arr,
        _lact_id_arr,
        _bact_val_arr,
        _lact_val_arr,
        _reward_arr,
        _command_arr,
        _bact_prior_arr
    ) = ([], [], [], [], [], [], [], [])

    for split in split_id:
        n_b = split[0]
        n_e = split[1]

        _obs_arr.extend(observations[n_b : (n_e + 1)])
        _bact_id_arr.extend(
            actions_behavior_id[n_b : n_e + 1]
        ) 
        _bact_val_arr.extend(
            actions_behavior_val[n_b : n_e + 1]
        ) 
        _lact_id_arr.extend(actions_label_id[n_b : n_e + 1]) 
        _lact_val_arr.extend(
            actions_label_val[n_b : n_e + 1]
        ) 
        _reward_arr.extend(rewards[n_b : n_e + 1]) 
        _command_arr.extend(command[n_b : n_e + 1])
        if actions_behavior_prior is not None:
            _bact_prior_arr.extend(actions_behavior_prior[n_b : n_e + 1])
        

    return (
        np.array(_obs_arr),
        np.array(_bact_id_arr),
        np.array(_bact_val_arr),
        np.array(_lact_id_arr),
        np.array(_lact_val_arr),
        np.array(_reward_arr),
        np.array(_command_arr),
        np.array(_bact_prior_arr)
    )

def cut_data(
    observations,
    actions_behavior_id,
    actions_behavior_val,
    actions_label_id,
    actions_label_val,
    rewards,
    command,
    actions_behavior_prior,
    percentage=1,
    time_step=2_000,
):

    split_id = []  # select the start and end of the data
    n_start = 0
    n_end = 0
    n_episode = 0

    flag = False  # flag为True: 下一个end没用
    for i in range(len(observations)):
        if actions_behavior_id[i] == 16:  # end

            if flag:
                flag = False
                continue

            flag = True

            n_episode += 1
            n_end = i + 1 # 跳过全黑

            delta = n_end - n_start
            # delta = int((n_end - n_start) * percentage)
            if delta == 0:
                delta = 1
            n_start = n_end - delta
            split_id.append((n_start, n_end))
            n_start = n_end+1
        if n_episode*2 >= time_step:
            break
    # print("sum_data(split_id)", sum_data(split_id))
    split_id = cut_data_(split_id, aim_len=time_step+1)
    # print("sum_data(split_id)", sum_data(split_id))
    # print("split_id",split_id)
    (
        _obs_arr,
        _bact_id_arr,
        _lact_id_arr,
        _bact_val_arr,
        _lact_val_arr,
        _reward_arr,
        _command_arr,
        _bact_prior_arr,
    ) = ([], [], [], [], [], [], [], [])

    for split in split_id:
        n_b = split[0]
        n_e = split[1]

        _obs_arr.extend(observations[n_b : (n_e + 1)])
        _bact_id_arr.extend(
            actions_behavior_id[n_b : n_e + 1]
        ) 
        _bact_val_arr.extend(
            actions_behavior_val[n_b : n_e + 1]
        ) 
        _lact_id_arr.extend(actions_label_id[n_b : n_e + 1]) 
        _lact_val_arr.extend(
            actions_label_val[n_b : n_e + 1]
        ) 
        _reward_arr.extend(rewards[n_b : n_e + 1]) 

        _command_arr.extend(command[n_b : n_e + 1])
        if actions_behavior_prior is not None:
            _bact_prior_arr.extend(actions_behavior_prior[n_b : n_e + 1])

        

    return (
        np.array(_obs_arr),
        np.array(_bact_id_arr),
        np.array(_bact_val_arr),
        np.array(_lact_id_arr),
        np.array(_lact_val_arr),
        np.array(_reward_arr),
        np.array(_command_arr),
        np.array(_bact_prior_arr),
    )






def sum_data(split_id):
    _sum = 0
    for split in split_id:
        n_b = split[0]
        n_e = split[1]
        _sum += n_e - n_b+1
    return _sum

def cut_data_(split_id,aim_len=2_000):
    while(sum_data(split_id) != aim_len):
        for split in split_id:
            n_b = split[0]
            n_e = split[1]
            if n_e - n_b >= 2:
                split_id.remove(split)
                split_id.append([n_b+1,n_e])
            if sum_data(split_id) == aim_len:
                break
    return split_id






class MazeDataSet(Dataset):
    def __init__(self, directory, time_step, verbose=False, max_maze=None, folder_verbose=False):
        self.folder_verbose = folder_verbose
        if(verbose):
            print("\nInitializing data set from file: %s..." % directory)
        if folder_verbose:
            print("Folder verbose is on")
        self.file_list = []
        directories = []
        if(isinstance(directory, list)):
            directories.extend(directory)
        else:
            directories.append(directory)
        self.directories = directories
        count = 0
        for d in directories:
            print(f"Loading data from {d}")
            for folder in os.listdir(d):
                folder_path = os.path.join(d, folder)
                if os.path.isdir(folder_path):
                    single_layer_flag = False
                    for file in os.listdir(folder_path):
                        if file == "observations.npy": # while...there must be a observation file right?
                            single_layer_flag = True
                            break
                        if os.path.isdir(os.path.join(folder_path, file)): # if there is a subfolder, then it is not a single layer folder
                            single_layer_flag = False
                            break
                    if max_maze != None and count >= max_maze:
                        break
                    if single_layer_flag:
                        self.file_list.append(folder_path)
                        count += 1
                    else:
                        for subfolder in os.listdir(folder_path):
                            subfolder_path = os.path.join(folder_path, subfolder)
                            if os.path.isdir(subfolder_path):
                                self.file_list.append(subfolder_path)
                                count += 1
                                if max_maze != None and count >= max_maze:
                                    break
            print(f"Before filtered Loading data from {d} finished, the number of data is {len(self.file_list)}")

        filter_file_list = []
        self.time_interval = None
        if isinstance(time_step, list):
            assert len(time_step) == 2, "time_step must be a list with length 2"
            self.time_interval = time_step.copy()
            time_step = time_step[1] - time_step[0]
        
        for file_path in self.file_list:
            if "maze" in file_path:
                filter_file_list.append(file_path)
                continue
            actions_behavior_id = np.load(file_path + '/actions_behavior_id.npy')
            if actions_behavior_id.shape[0] >= time_step:
                filter_file_list.append(file_path)
        self.file_list = filter_file_list
        
        assert len(self.file_list) > 0, "No data in the data set"
        if len(self.file_list) % 8 != 0:
            print(f"[Warning] The number of data is not divisible by 8, the number of data is {len(self.file_list)}")
            self.file_list = self.file_list[:len(self.file_list) - len(self.file_list) % 8]

        self.time_step = time_step

        if(verbose):
            print("...finished initializing data set, number of samples: %s\n" % len(self.file_list))

    def __getitem__(self, index):
        path = self.file_list[index]
        if "traj" in path.split("/")[-1] or "path" in path.split("/")[-1]:
            folder_name = os.path.join(path.split("/")[-2], path.split("/")[-1])
            # print(folder_name)
        else:
            folder_name = path.split("/")[-1]
        if "maze" in path:
            # print("dataset loading maze data")
            if self.folder_verbose:
                return self.__get_maze__(index), folder_name
            return self.__get_maze__(index)
        else:
            # print("dataset loading procthor data")
            # if self.folder_verbose:
            #     return self.__get_procthor__(index), folder_name
            # return self.__get_procthor__(index)

            if self.folder_verbose:
                return self.__get_procthor_short__(index), path
            return self.__get_procthor_short__(index)
    
    def __len__(self):
        return len(self.file_list)
    
    def id2val(self, id_arr):
        DEFAULT_ACTION_SPACE_16 = [(0.0, 0.5), 
                        (0.05, 0.0), (-0.05, 0.0),
                        (0.1, 0.0), (-0.1, 0.0),
                        (0.2, 0.0), (-0.2, 0.0),
                        (0.3, 0.0), (-0.3, 0.0),
                        (0.5, 0.0), (-0.5, 0.0), 
                        (0.0, 1.0),
                        (0.05, 1.0), 
                        (-0.05, 1.0), 
                        (0.10, 1.0), 
                        (-0.10, 1.0), 
                        ]

        val_arr = []
        for id in id_arr:
            if id > 15:
                val_arr.append((0.0, 0.0))
            else:
                val_arr.append(DEFAULT_ACTION_SPACE_16[id])
        return np.array(val_arr)
    
    def __get_maze__(self, index):
        path = self.file_list[index]
        try:
            cmds = np.load(path + '/commands.npy')
            observations = np.load(path + '/observations.npy')
            actions_behavior_id = np.load(path + '/actions_behavior_id.npy')
            actions_label_id = np.load(path + '/actions_label_id.npy')
            if os.path.exists(path + '/actions_behavior_val.npy') and os.path.exists(path + '/actions_label_val.npy'):
                actions_behavior_val = np.load(path + '/actions_behavior_val.npy')
                actions_label_val = np.load(path + '/actions_label_val.npy')
            else:
                actions_behavior_val = self.id2val(actions_behavior_id)
                actions_label_val = self.id2val(actions_label_id)
            
            rewards = np.load(path + '/rewards.npy')
            # bevs = np.load(path + '/BEVs.npy')
            # if os.path.exists(path + '/actions_behavior_prior.npy'):
            #     actions_behavior_prior = np.load(path + '/actions_behavior_prior.npy')
            max_t = actions_behavior_id.shape[0]

            # Shape Check
            assert max_t == rewards.shape[0], f"Shape mismatch: rewards, expected {max_t}, got {rewards.shape[0]}"
            assert max_t == actions_behavior_val.shape[0], f"Shape mismatch: actions_behavior_val, expected {max_t}, got {actions_behavior_val.shape[0]}"
            assert max_t == actions_label_id.shape[0], f"Shape mismatch: actions_label_id, expected {max_t}, got {actions_label_id.shape[0]}"
            assert max_t == actions_label_val.shape[0], f"Shape mismatch: actions_label_val, expected {max_t}, got {actions_label_val.shape[0]}"
            # assert max_t == bevs.shape[0]
            assert max_t + 1 == observations.shape[0], f"Shape mismatch: observations, expected {max_t + 1}, got {observations.shape[0]}"

            if(self.time_step > max_t):
                print(f'[Warning] Load samples from {path} that is shorter ({max_t}) than specified time step ({self.time_step})')
                n_b = 0
                n_e = max_t
            else:
                n_b = 0
                n_e = self.time_step
            cmd_arr = torch.from_numpy(cmds).float()
            
            # Normalize command to [B, 16*16*3]
            if(cmd_arr.dim() == 2): # Normalize to [B，16，16，3]
                cmd_arr = np.repeat(cmd_arr, 256, axis=1)
            elif(cmd_arr.dim() == 4):
                cmd_arr = cmd_arr.reshape(cmd_arr.shape[0], -1)
            
            cmd_arr = cmd_arr[n_b:(n_e)]
            obs_arr = torch.from_numpy(observations[n_b:(n_e + 1)]).float() 
            bact_id_arr = torch.from_numpy(actions_behavior_id[n_b:n_e]).long() 
            lact_id_arr = torch.from_numpy(actions_label_id[n_b:n_e]).long() 
            bact_val_arr = torch.from_numpy(actions_behavior_val[n_b:n_e]).float() 
            lact_val_arr = torch.from_numpy(actions_label_val[n_b:n_e]).float() 
            reward_arr = torch.from_numpy(rewards[n_b:n_e]).float()
            # bev_arr = torch.from_numpy(bevs[n_b:n_e]).float()
            
            return cmd_arr, obs_arr, bact_id_arr, lact_id_arr, bact_val_arr, lact_val_arr, reward_arr#, bev_arr
        except Exception as e:
            print(f"Unexpected reading error founded when loading {path}: {e}")
            return None
    
    def __get_procthor_short__(self, index):

        path = self.file_list[index]
        try:
            observations = np.load(path + "/observations.npy").astype(np.uint8)
            actions_behavior_id = np.load(path + "/actions_behavior_id.npy").astype(np.int32)
            actions_behavior_val = np.load(path + "/actions_behavior_val.npy").astype(np.float32)
            actions_label_id = np.load(path + "/actions_label_id.npy").astype(np.int32)
            actions_label_val = np.load(path + "/actions_label_val.npy").astype(np.float32)
            if os.path.exists(path + "/actions_behavior_prior.npy"):
                actions_behavior_prior = np.load(path + "/actions_behavior_prior.npy").astype(np.int32)

            # for i in range(len(actions_label_id)):
            #     if actions_label_id[i][0] == 16:
            #         actions_label_id[i][0] = 17
                    
            rewards = np.load(path + "/rewards.npy").astype(np.float32)
            if os.path.exists(path + "/commands.npy"):
                command = np.load(path + "/commands.npy").astype(np.uint8)
            else:
                assert False, "WE MUST HAVE COMMAND!, No command found in %s" % path
                command = np.zeros((len(observations), 16, 16, 3)).astype(np.uint8)
            
            # get the last time steps' index
            n_b = observations.shape[0] - (self.time_step + 1)
            n_e = self.time_step + n_b
            
            if self.time_interval is not None:
                n_b = self.time_interval[0]
                n_e = self.time_interval[1]
            # Ensure that all arrays are of correct dtype
            obs_arr = torch.from_numpy(observations).float()
            bact_id_arr = torch.from_numpy(
                actions_behavior_id
            ).long() 
            bact_val_arr = torch.from_numpy(
                actions_behavior_val
            ).float() 
            lact_id_arr = torch.from_numpy(
                actions_label_id
            ).long() 
            lact_val_arr = torch.from_numpy(
                actions_label_val
            ).float()
             
            reward_arr = torch.from_numpy(rewards).float() 
            
            command_arr = torch.from_numpy(command).float() 
            if actions_behavior_prior is not None and len(actions_behavior_prior) > 0:
                bact_prior_arr = torch.from_numpy(actions_behavior_prior).float() 

            obs_arr = obs_arr.permute(0, 2, 1, 3)
            # rotate the image by 90 degrees 
            obs_arr = torch.rot90(obs_arr, 2, [1, 2])

            # obs_arr = torch.rot90(obs_arr, 1, [1, 2])

            assert obs_arr[n_b:n_e + 1].shape[0] == self.time_step + 1, f"shape mismatch: obs_arr, expected {self.time_step + 1}, got {obs_arr[n_b:].shape[0]}"
            assert bact_id_arr[n_b:n_e].shape[0] == self.time_step, f"shape mismatch: bact_id_arr, expected {self.time_step}, got {bact_id_arr[n_b:].shape[0]}"
            assert lact_id_arr[n_b:n_e].shape[0] == self.time_step, f"shape mismatch: lact_id_arr, expected {self.time_step}, got {lact_id_arr[n_b:].shape[0]}"
            assert bact_val_arr[n_b:n_e].shape[0] == self.time_step, f"shape mismatch: bact_val_arr, expected {self.time_step}, got {bact_val_arr[n_b:].shape[0]}"
            assert lact_val_arr[n_b:n_e].shape[0] == self.time_step, f"shape mismatch: lact_val_arr, expected {self.time_step}, got {lact_val_arr[n_b:].shape[0]}"
            assert reward_arr[n_b:n_e].shape[0] == self.time_step, f"shape mismatch: reward_arr, expected {self.time_step}, got {reward_arr[n_b:].shape[0]}"
            assert command_arr[n_b:n_e].shape[0] == self.time_step, f"shape mismatch: command_arr, expected {self.time_step}, got {command_arr[n_b:].shape[0]}"
            # assert bact_prior_arr.shape[0] == self.time_step

            # return (
            #     # cmd_arr, obs_arr, bact_id_arr, lact_id_arr, bact_val_arr, lact_val_arr, reward_arr
            #     command_arr[:n_e].view(command_arr[:n_e].shape[0], -1),
            #     obs_arr[:n_e+1],
            #     bact_id_arr[:n_e], # cut the last 'end'
            #     lact_id_arr[:n_e, 0], # lact_id_arr[0:self.time_step],
            #     bact_val_arr[:n_e],
            #     lact_val_arr[:n_e],
            #     reward_arr[:n_e],
            #     # bact_prior_arr[0:self.time_step]
            # )

            return (
                # cmd_arr, obs_arr, bact_id_arr, lact_id_arr, bact_val_arr, lact_val_arr, reward_arr
                command_arr[n_b:n_e].view(command_arr[n_b:n_e].shape[0], -1),
                obs_arr[n_b:n_e+1],
                bact_id_arr[n_b:n_e], # cut the last 'end'
                lact_id_arr[n_b:n_e, 0], # lact_id_arr[0:self.time_step],
                bact_val_arr[n_b:n_e],
                lact_val_arr[n_b:n_e],
                reward_arr[n_b:n_e],
                # bact_prior_arr[0:self.time_step]
            )

            # return (
            #     # cmd_arr, obs_arr, bact_id_arr, lact_id_arr, bact_val_arr, lact_val_arr, reward_arr
            #     command_arr[0:self.time_step].view(command_arr[0:self.time_step].shape[0], -1),
            #     obs_arr[0:self.time_step+1],
            #     bact_id_arr[0:self.time_step], # cut the last 'end'
            #     lact_id_arr[0:self.time_step, 0], # lact_id_arr[0:self.time_step],
            #     bact_val_arr[0:self.time_step],
            #     lact_val_arr[0:self.time_step],
            #     reward_arr[0:self.time_step],
            #     # bact_prior_arr[0:self.time_step]
            # )
        except Exception as e:
            print(f"Unexpected reading error founded when loading {path}: {e}")
            return None

    def __get_procthor__(self, index):

        path = self.file_list[index]
        try:
            observations = np.load(path + "/observations.npy").astype(np.uint8)
            actions_behavior_id = np.load(path + "/actions_behavior_id.npy").astype(np.int32)
            actions_behavior_val = np.load(path + "/actions_behavior_val.npy").astype(np.float32)
            actions_label_id = np.load(path + "/actions_label_id.npy").astype(np.int32)
            actions_label_val = np.load(path + "/actions_label_val.npy").astype(np.float32)
            if os.path.exists(path + "/actions_behavior_prior.npy"):
                actions_behavior_prior = np.load(path + "/actions_behavior_prior.npy").astype(np.int32)

            rewards = np.load(path + "/rewards.npy").astype(np.float32)
            if os.path.exists(path + "/commands.npy"):
                command = np.load(path + "/commands.npy").astype(np.uint8)
            # elif os.path.exists(path + "/target.npy"):
            #     command = np.load(path + "/target.npy").astype(np.uint8)
            else:
                assert False, "WE MUST HAVE COMMAND!, No command found in %s" % path
                command = np.zeros((len(observations), 16, 16, 3)).astype(np.uint8)
            
            # FIXED
            for i in range(len(actions_label_id)):
                if actions_label_id[i][1] == 16:
                    actions_label_id[i][1] = 17

            # print(len(observations))
            # 1800 length by now
            percent = self.time_step / len(observations)
            percent = 10000 / len(observations)
            if percent < 1:
                (
                    observations,
                    actions_behavior_id,
                    actions_behavior_val,
                    actions_label_id,
                    actions_label_val,
                    rewards,
                    command,
                    actions_behavior_prior
                ) = cut_data(
                    observations,
                    actions_behavior_id,
                    actions_behavior_val,
                    actions_label_id,
                    actions_label_val,
                    rewards,
                    command,
                    actions_behavior_prior,
                    percentage=percent,
                    time_step=self.time_step,
                )
            else:
                (
                    observations,
                    actions_behavior_id,
                    actions_behavior_val,
                    actions_label_id,
                    actions_label_val,
                    rewards,
                    command,
                    actions_behavior_prior
                ) = expend_data(
                    observations,
                    actions_behavior_id,
                    actions_behavior_val,
                    actions_label_id,
                    actions_label_val,
                    rewards,
                    command,
                    actions_behavior_prior, # TODO
                    percentage=percent,
                )


            # Ensure that all arrays are of correct dtype
            obs_arr = torch.from_numpy(observations).float()
            bact_id_arr = torch.from_numpy(
                actions_behavior_id
            ).long() 
            bact_val_arr = torch.from_numpy(
                actions_behavior_val
            ).float() 
            lact_id_arr = torch.from_numpy(
                actions_label_id
            ).long() 
            lact_val_arr = torch.from_numpy(
                actions_label_val
            ).float() 
            reward_arr = torch.from_numpy(rewards).float() 
            
            command_arr = torch.from_numpy(command).float() 
            if actions_behavior_prior is not None and len(actions_behavior_prior) > 0:
                bact_prior_arr = torch.from_numpy(actions_behavior_prior).float() 

            # print(obs_arr.shape)
            # print(self.time_step)
            obs_arr = obs_arr.permute(0, 2, 1, 3)
            # rotate the image by 90 degrees 
            obs_arr = torch.rot90(obs_arr, 2, [1, 2])
            # return (
            #     # cmd_arr, obs_arr, bact_id_arr, lact_id_arr, bact_val_arr, lact_val_arr, reward_arr
            #     command_arr[0:1800].view(command_arr[0:1800].shape[0], -1),
            #     obs_arr[0:1801],
            #     bact_id_arr[0:1800], # cut the last 'end'
            #     lact_id_arr[0:1800, 0], # lact_id_arr[0:self.time_step],
            #     bact_val_arr[0:1800],
            #     lact_val_arr[0:1800],
            #     reward_arr[0:1800],
            #     # bact_prior_arr[0:self.time_step]
            # )
            return (
                # cmd_arr, obs_arr, bact_id_arr, lact_id_arr, bact_val_arr, lact_val_arr, reward_arr
                command_arr[0:self.time_step].view(command_arr[0:self.time_step].shape[0], -1),
                obs_arr[0:self.time_step+1],
                bact_id_arr[0:self.time_step], # cut the last 'end'
                lact_id_arr[0:self.time_step, 1], # lact_id_arr[0:self.time_step],
                bact_val_arr[0:self.time_step],
                lact_val_arr[0:self.time_step],
                reward_arr[0:self.time_step],
                # bact_prior_arr[0:self.time_step]
            )
        except Exception as e:
            print(f"Unexpected reading error founded when loading {path}: {e}")
            return None


class MazeSplitShortDataSet(Dataset):
    def __init__(self, directory, time_step, verbose=False, max_maze=None, folder_verbose=False):
        self.folder_verbose = folder_verbose
        if(verbose):
            print("\nInitializing data set from file: %s..." % directory)
        if folder_verbose:
            print("Folder verbose is on")
        self.file_list = []
        directories = []
        if(isinstance(directory, list)):
            directories.extend(directory)
        else:
            directories.append(directory)
        self.directories = directories
        count = 0
        self.time_interval = None
        if isinstance(time_step, list):
            raise NotImplementedError("time_step must not be a list")
            # assert len(time_step) == 2, "time_step must be a list with length 2"
            # self.time_interval = time_step.copy()
            # time_step = time_step[1] - time_step[0]
            
        # OBS_LENGTH = 10001
        # self.n_split = OBS_LENGTH // time_step
        # self.split_interval_start = [i for i in range(0, OBS_LENGTH, time_step)][:self.n_split]
        directory_files = []
        for d in directories:
            directory_file_list = []
            print(f"Loading data from {d}")
            for folder in os.listdir(d):
                folder_path = os.path.join(d, folder)
                if os.path.isdir(folder_path):
                    single_layer_flag = False
                    for file in os.listdir(folder_path):
                        if file == "observations.npy": # while...there must be a observation file right?
                            single_layer_flag = True
                            break
                        if os.path.isdir(os.path.join(folder_path, file)): # if there is a subfolder, then it is not a single layer folder
                            single_layer_flag = False
                            break
                    if max_maze != None and count >= max_maze:
                        break
                    if single_layer_flag:
                        self.file_list.append(folder_path)
                        directory_file_list.append(folder_path)
                        count += 1
                    else:
                        for subfolder in os.listdir(folder_path):
                            subfolder_path = os.path.join(folder_path, subfolder)
                            if os.path.isdir(subfolder_path):
                                self.file_list.append(subfolder_path)
                                directory_file_list.append(folder_path)
                                count += 1
                                if max_maze != None and count >= max_maze:
                                    break
            directory_files.append(directory_file_list)
        length_of_data = [len(df) for df in directory_files]
        print(f"Before filter, totally there are {sum(length_of_data)}")
        for i, d in enumerate(directories):
            print(f"{d} has {length_of_data[i]} data")

        filter_file_list = []

        # for file_path in self.file_list:
        #     # if "maze" in file_path:
        #     #     for st in self.split_interval_start:
        #     #         filter_file_list.append((file_path, st))
        #     #     continue
        #     actions_behavior_id = np.load(file_path + '/actions_behavior_id.npy')
        #     n_split = actions_behavior_id.shape[0] // time_step
        #     split_interval_start = [i for i in range(0, actions_behavior_id.shape[0], time_step)][:n_split]
        #     if actions_behavior_id.shape[0] >= time_step:
        #         for st in split_interval_start:
        #             filter_file_list.append((file_path, st))
        for df in directory_files:
            max_file_per_dir = None # 8000 # TODO
            count = 0
            for file_path in df:
                actions_behavior_id = np.load(file_path + '/actions_behavior_id.npy')
                n_split = actions_behavior_id.shape[0] // time_step
                split_interval_start = [i for i in range(0, actions_behavior_id.shape[0], time_step)][:n_split]
                if max_file_per_dir is not None:
                    if count >= max_file_per_dir:
                        break
                if actions_behavior_id.shape[0] >= time_step:
                    for st in split_interval_start:
                        filter_file_list.append((file_path, st))
                    count += 1
        self.file_list = filter_file_list
        
        assert len(self.file_list) > 0, "No data in the data set"
        if len(self.file_list) % 8 != 0:
            print(f"[Warning] The number of data is not divisible by 8, the number of data is {len(self.file_list)}")
            self.file_list = self.file_list[:len(self.file_list) - len(self.file_list) % 8]
        self.time_step = time_step

        if(verbose):
            print("...finished initializing data set, number of samples: %s\n" % len(self.file_list))

    def __getitem__(self, index):
        path, start = self.file_list[index]
        if "traj" in path.split("/")[-1] or "path" in path.split("/")[-1]:
            folder_name = os.path.join(path.split("/")[-2], path.split("/")[-1])
            # print(folder_name)
        else:
            folder_name = path.split("/")[-1]
        if "maze" in path:
            # print("dataset loading maze data")
            if self.folder_verbose:
                return self.__get_maze__(index), folder_name
            return self.__get_maze__(index)
        else:
            # print("dataset loading procthor data")
            # if self.folder_verbose:
            #     return self.__get_procthor__(index), folder_name
            # return self.__get_procthor__(index)

            if self.folder_verbose:
                return self.__get_procthor_short__(index), path
            return self.__get_procthor_short__(index)

    def id2val(self, id_arr):
        DEFAULT_ACTION_SPACE_16 = [(0.0, 0.5), 
                        (0.05, 0.0), (-0.05, 0.0),
                        (0.1, 0.0), (-0.1, 0.0),
                        (0.2, 0.0), (-0.2, 0.0),
                        (0.3, 0.0), (-0.3, 0.0),
                        (0.5, 0.0), (-0.5, 0.0), 
                        (0.0, 1.0),
                        (0.05, 1.0), 
                        (-0.05, 1.0), 
                        (0.10, 1.0), 
                        (-0.10, 1.0), 
                        ]

        val_arr = []
        for id in id_arr:
            if id > 15:
                val_arr.append((0.0, 0.0))
            else:
                val_arr.append(DEFAULT_ACTION_SPACE_16[id])
        return np.array(val_arr)
  
    def __len__(self):
        return len(self.file_list)
    
    def __get_maze__(self, index):
        path, start = self.file_list[index]
        try:
            cmds = np.load(path + '/commands.npy')
            observations = np.load(path + '/observations.npy')
            actions_behavior_id = np.load(path + '/actions_behavior_id.npy')
            actions_label_id = np.load(path + '/actions_label_id.npy')
            # actions_label_id = np.load(path + '/actions_behavior_id.npy')
            
            if os.path.exists(path + '/actions_behavior_val.npy') and os.path.exists(path + '/actions_label_val.npy'):
                actions_behavior_val = np.load(path + '/actions_behavior_val.npy')
                actions_label_val = np.load(path + '/actions_label_val.npy')
            else:
                actions_behavior_val = self.id2val(actions_behavior_id)
                actions_label_val = self.id2val(actions_label_id)
            rewards = np.load(path + '/rewards.npy')
            # bevs = np.load(path + '/BEVs.npy')
            # if os.path.exists(path + '/actions_behavior_prior.npy'):
            #     actions_behavior_prior = np.load(path + '/actions_behavior_prior.npy')
            max_t = actions_behavior_id.shape[0]

            # Shape Check
            assert max_t == rewards.shape[0], f"Shape mismatch: rewards, expected {max_t}, got {rewards.shape[0]}"
            assert max_t == actions_behavior_val.shape[0], f"Shape mismatch: actions_behavior_val, expected {max_t}, got {actions_behavior_val.shape[0]}"
            assert max_t == actions_label_id.shape[0], f"Shape mismatch: actions_label_id, expected {max_t}, got {actions_label_id.shape[0]}"
            assert max_t == actions_label_val.shape[0], f"Shape mismatch: actions_label_val, expected {max_t}, got {actions_label_val.shape[0]}"
            # assert max_t == bevs.shape[0]
            assert max_t + 1 == observations.shape[0], f"Shape mismatch: observations, expected {max_t + 1}, got {observations.shape[0]}"

            if(self.time_step > max_t):
                print(f'[Warning] Load samples from {path} that is shorter ({max_t}) than specified time step ({self.time_step})')
                n_b = start
                n_e = max_t
            else:
                n_b = start
                n_e = start + self.time_step
            cmd_arr = torch.from_numpy(cmds).float()
            
            # Normalize command to [B, 16*16*3]
            if(cmd_arr.dim() == 2): # Normalize to [B，16，16，3]
                cmd_arr = np.repeat(cmd_arr, 256, axis=1)
            elif(cmd_arr.dim() == 4):
                cmd_arr = cmd_arr.reshape(cmd_arr.shape[0], -1)
            
            cmd_arr = cmd_arr[n_b:(n_e)]
            obs_arr = torch.from_numpy(observations[n_b:(n_e + 1)]).float() 
            bact_id_arr = torch.from_numpy(actions_behavior_id[n_b:n_e]).long() 
            lact_id_arr = torch.from_numpy(actions_label_id[n_b:n_e]).long() 
            bact_val_arr = torch.from_numpy(actions_behavior_val[n_b:n_e]).float() 
            lact_val_arr = torch.from_numpy(actions_label_val[n_b:n_e]).float() 
            reward_arr = torch.from_numpy(rewards[n_b:n_e]).float()
            # bev_arr = torch.from_numpy(bevs[n_b:n_e]).float()
            
            return cmd_arr, obs_arr, bact_id_arr, lact_id_arr, bact_val_arr, lact_val_arr, reward_arr#, bev_arr
        except Exception as e:
            print(f"Unexpected reading error founded when loading {path}: {e}")
            return None
    
    def __get_procthor_short__(self, index):

        path, start = self.file_list[index]
        try:
            observations = np.load(path + "/observations.npy").astype(np.uint8)
            actions_behavior_id = np.load(path + "/actions_behavior_id.npy").astype(np.int32)
            actions_behavior_val = np.load(path + "/actions_behavior_val.npy").astype(np.float32)
            actions_label_id = np.load(path + "/actions_label_id.npy").astype(np.int32)
            actions_label_val = np.load(path + "/actions_label_val.npy").astype(np.float32)
            if os.path.exists(path + "/actions_behavior_prior.npy"):
                actions_behavior_prior = np.load(path + "/actions_behavior_prior.npy").astype(np.int32)

            # for i in range(len(actions_label_id)):
            #     if actions_label_id[i][0] == 16:
            #         actions_label_id[i][0] = 17
                    
            rewards = np.load(path + "/rewards.npy").astype(np.float32)
            if os.path.exists(path + "/commands.npy"):
                command = np.load(path + "/commands.npy").astype(np.uint8)
            else:
                assert False, "WE MUST HAVE COMMAND!, No command found in %s" % path
                command = np.zeros((len(observations), 16, 16, 3)).astype(np.uint8)
            
            # get the last time steps' index
            n_b = start + observations.shape[0] - (self.time_step + 1)
            n_e = self.time_step + n_b
            
            if self.time_interval is not None:
                n_b = self.time_interval[0]
                n_e = self.time_interval[1]
            # Ensure that all arrays are of correct dtype
            obs_arr = torch.from_numpy(observations).float()
            bact_id_arr = torch.from_numpy(
                actions_behavior_id
            ).long() 
            bact_val_arr = torch.from_numpy(
                actions_behavior_val
            ).float() 
            lact_id_arr = torch.from_numpy(
                actions_label_id
            ).long() 
            lact_val_arr = torch.from_numpy(
                actions_label_val
            ).float()
             
            reward_arr = torch.from_numpy(rewards).float() 
            
            command_arr = torch.from_numpy(command).float() 
            if actions_behavior_prior is not None and len(actions_behavior_prior) > 0:
                bact_prior_arr = torch.from_numpy(actions_behavior_prior).float() 

            obs_arr = obs_arr.permute(0, 2, 1, 3)
            # rotate the image by 90 degrees 
            obs_arr = torch.rot90(obs_arr, 2, [1, 2])

            # obs_arr = torch.rot90(obs_arr, 1, [1, 2])

            assert obs_arr[n_b:n_e + 1].shape[0] == self.time_step + 1, f"shape mismatch: obs_arr, expected {self.time_step + 1}, got {obs_arr[n_b:].shape[0]}"
            assert bact_id_arr[n_b:n_e].shape[0] == self.time_step, f"shape mismatch: bact_id_arr, expected {self.time_step}, got {bact_id_arr[n_b:].shape[0]}"
            assert lact_id_arr[n_b:n_e].shape[0] == self.time_step, f"shape mismatch: lact_id_arr, expected {self.time_step}, got {lact_id_arr[n_b:].shape[0]}"
            assert bact_val_arr[n_b:n_e].shape[0] == self.time_step, f"shape mismatch: bact_val_arr, expected {self.time_step}, got {bact_val_arr[n_b:].shape[0]}"
            assert lact_val_arr[n_b:n_e].shape[0] == self.time_step, f"shape mismatch: lact_val_arr, expected {self.time_step}, got {lact_val_arr[n_b:].shape[0]}"
            assert reward_arr[n_b:n_e].shape[0] == self.time_step, f"shape mismatch: reward_arr, expected {self.time_step}, got {reward_arr[n_b:].shape[0]}"
            assert command_arr[n_b:n_e].shape[0] == self.time_step, f"shape mismatch: command_arr, expected {self.time_step}, got {command_arr[n_b:].shape[0]}"
            # assert bact_prior_arr.shape[0] == self.time_step

            # return (
            #     # cmd_arr, obs_arr, bact_id_arr, lact_id_arr, bact_val_arr, lact_val_arr, reward_arr
            #     command_arr[:n_e].view(command_arr[:n_e].shape[0], -1),
            #     obs_arr[:n_e+1],
            #     bact_id_arr[:n_e], # cut the last 'end'
            #     lact_id_arr[:n_e, 0], # lact_id_arr[0:self.time_step],
            #     bact_val_arr[:n_e],
            #     lact_val_arr[:n_e],
            #     reward_arr[:n_e],
            #     # bact_prior_arr[0:self.time_step]
            # )

            return (
                # cmd_arr, obs_arr, bact_id_arr, lact_id_arr, bact_val_arr, lact_val_arr, reward_arr
                command_arr[n_b:n_e].view(command_arr[n_b:n_e].shape[0], -1),
                obs_arr[n_b:n_e+1],
                bact_id_arr[n_b:n_e], # cut the last 'end'
                lact_id_arr[n_b:n_e, 0], # lact_id_arr[0:self.time_step],
                bact_val_arr[n_b:n_e],
                lact_val_arr[n_b:n_e],
                reward_arr[n_b:n_e],
                # bact_prior_arr[0:self.time_step]
            )

            # return (
            #     # cmd_arr, obs_arr, bact_id_arr, lact_id_arr, bact_val_arr, lact_val_arr, reward_arr
            #     command_arr[0:self.time_step].view(command_arr[0:self.time_step].shape[0], -1),
            #     obs_arr[0:self.time_step+1],
            #     bact_id_arr[0:self.time_step], # cut the last 'end'
            #     lact_id_arr[0:self.time_step, 0], # lact_id_arr[0:self.time_step],
            #     bact_val_arr[0:self.time_step],
            #     lact_val_arr[0:self.time_step],
            #     reward_arr[0:self.time_step],
            #     # bact_prior_arr[0:self.time_step]
            # )
        except Exception as e:
            print(f"Unexpected reading error founded when loading {path}: {e}")
            return None



class MazeRandomShortDataSet(Dataset):
    def __init__(self, directory, time_step, verbose=False, max_maze=None, folder_verbose=False):
        self.folder_verbose = folder_verbose
        if(verbose):
            print("\nInitializing data set from file: %s..." % directory)
        if folder_verbose:
            print("Folder verbose is on")
        self.file_list = []
        directories = []
        if(isinstance(directory, list)):
            directories.extend(directory)
        else:
            directories.append(directory)
        self.directories = directories
        count = 0
        for d in directories:
            print(f"Loading data from {d}")
            for folder in os.listdir(d):
                folder_path = os.path.join(d, folder)
                if os.path.isdir(folder_path):
                    single_layer_flag = False
                    for file in os.listdir(folder_path):
                        if file == "observations.npy": # while...there must be a observation file right?
                            single_layer_flag = True
                            break
                        if os.path.isdir(os.path.join(folder_path, file)): # if there is a subfolder, then it is not a single layer folder
                            single_layer_flag = False
                            break
                    if max_maze != None and count >= max_maze:
                        break
                    if single_layer_flag:
                        self.file_list.append(folder_path)
                        count += 1
                    else:
                        for subfolder in os.listdir(folder_path):
                            subfolder_path = os.path.join(folder_path, subfolder)
                            if os.path.isdir(subfolder_path):
                                self.file_list.append(subfolder_path)
                                count += 1
                                if max_maze != None and count >= max_maze:
                                    break
            print(f"Before filtered Loading data from {d} finished, the number of data is {len(self.file_list)}")

        filter_file_list = []
        self.time_interval = None
        if isinstance(time_step, list):
            assert len(time_step) == 2, "time_step must be a list with length 2"
            self.time_interval = time_step.copy()
            time_step = time_step[1] - time_step[0]
        
        for file_path in self.file_list:
            if "maze" in file_path:
                filter_file_list.append(file_path)
                continue
            actions_behavior_id = np.load(file_path + '/actions_behavior_id.npy')
            if actions_behavior_id.shape[0] >= time_step:
                filter_file_list.append(file_path)
        self.file_list = filter_file_list
        
        assert len(self.file_list) > 0, "No data in the data set"
        if len(self.file_list) % 8 != 0:
            print(f"[Warning] The number of data is not divisible by 8, the number of data is {len(self.file_list)}")
            self.file_list = self.file_list[:len(self.file_list) - len(self.file_list) % 8]

        self.time_step = time_step

        if(verbose):
            print("...finished initializing data set, number of samples: %s\n" % len(self.file_list))

    def __getitem__(self, index):
        path = self.file_list[index]
        if "traj" in path.split("/")[-1] or "path" in path.split("/")[-1]:
            folder_name = os.path.join(path.split("/")[-2], path.split("/")[-1])
            # print(folder_name)
        else:
            folder_name = path.split("/")[-1]
        if "maze" in path:
            # print("dataset loading maze data")
            if self.folder_verbose:
                return self.__get_maze__(index), folder_name
            return self.__get_maze__(index)
        else:
            # print("dataset loading procthor data")
            # if self.folder_verbose:
            #     return self.__get_procthor__(index), folder_name
            # return self.__get_procthor__(index)
            raise NotImplementedError
            # if self.folder_verbose:
            #     return self.__get_procthor_short__(index), path
            # return self.__get_procthor_short__(index)
    
    def __len__(self):
        return len(self.file_list)
    
    def __get_maze__(self, index):
        path = self.file_list[index]
        try:
            cmds = np.load(path + '/commands.npy')
            observations = np.load(path + '/observations.npy')
            actions_behavior_id = np.load(path + '/actions_behavior_id.npy')
            actions_label_id = np.load(path + '/actions_label_id.npy')
            actions_behavior_val = np.load(path + '/actions_behavior_val.npy')
            actions_label_val = np.load(path + '/actions_label_val.npy')
            rewards = np.load(path + '/rewards.npy')
            # bevs = np.load(path + '/BEVs.npy')
            # if os.path.exists(path + '/actions_behavior_prior.npy'):
            #     actions_behavior_prior = np.load(path + '/actions_behavior_prior.npy')
            max_t = actions_behavior_id.shape[0]

            # Shape Check
            assert max_t == rewards.shape[0], f"Shape mismatch: rewards, expected {max_t}, got {rewards.shape[0]}"
            assert max_t == actions_behavior_val.shape[0], f"Shape mismatch: actions_behavior_val, expected {max_t}, got {actions_behavior_val.shape[0]}"
            assert max_t == actions_label_id.shape[0], f"Shape mismatch: actions_label_id, expected {max_t}, got {actions_label_id.shape[0]}"
            assert max_t == actions_label_val.shape[0], f"Shape mismatch: actions_label_val, expected {max_t}, got {actions_label_val.shape[0]}"
            # assert max_t == bevs.shape[0]
            assert max_t + 1 == observations.shape[0], f"Shape mismatch: observations, expected {max_t + 1}, got {observations.shape[0]}"

            if(self.time_step > max_t):
                print(f'[Warning] Load samples from {path} that is shorter ({max_t}) than specified time step ({self.time_step})')
                n_b = 0
                n_e = max_t
            else:
                # ne = nb + self.time_step, and ne + 1 < max_t, so that 
                n_b = np.random.randint(0, max_t - self.time_step + 1)
                n_e = n_b + self.time_step
            cmd_arr = torch.from_numpy(cmds).float()
            
            # Normalize command to [B, 16*16*3]
            if(cmd_arr.dim() == 2): # Normalize to [B，16，16，3]
                cmd_arr = np.repeat(cmd_arr, 256, axis=1)
            elif(cmd_arr.dim() == 4):
                cmd_arr = cmd_arr.reshape(cmd_arr.shape[0], -1)
            
            cmd_arr = cmd_arr[n_b:(n_e)]
            obs_arr = torch.from_numpy(observations[n_b:(n_e + 1)]).float() 
            bact_id_arr = torch.from_numpy(actions_behavior_id[n_b:n_e]).long() 
            lact_id_arr = torch.from_numpy(actions_label_id[n_b:n_e]).long() 
            bact_val_arr = torch.from_numpy(actions_behavior_val[n_b:n_e]).float() 
            lact_val_arr = torch.from_numpy(actions_label_val[n_b:n_e]).float() 
            reward_arr = torch.from_numpy(rewards[n_b:n_e]).float()
            # bev_arr = torch.from_numpy(bevs[n_b:n_e]).float()
            
            return cmd_arr, obs_arr, bact_id_arr, lact_id_arr, bact_val_arr, lact_val_arr, reward_arr#, bev_arr
        except Exception as e:
            print(f"Unexpected reading error founded when loading {path}: {e}")
            return None

class ProcthorOnlineDataSet(Dataset):
    def __init__(self, directory, time_step, verbose=False, max_maze=None, folder_verbose=False):
        self.folder_verbose = folder_verbose
        if(verbose):
            print("\nInitializing data set from file: %s..." % directory)
        if folder_verbose:
            print("Folder verbose is on")
        self.file_list = []
        directories = []
        if(isinstance(directory, list)):
            directories.extend(directory)
        else:
            directories.append(directory)
        self.directories = directories
        for d in directories:
            print(f"Loading data from {d}")
            count = 0
            for folder in os.listdir(d):
                folder_path = os.path.join(d, folder)
                if os.path.isdir(folder_path):
                    single_layer_flag = False
                    for file in os.listdir(folder_path):
                        if file == "house.json": # while...there must be a observation file right?
                            single_layer_flag = True
                            break
                        if os.path.isdir(os.path.join(folder_path, file)): # if there is a subfolder, then it is not a single layer folder
                            single_layer_flag = False
                            break
                    if max_maze != None and count >= max_maze:
                        break
                    if single_layer_flag:
                        self.file_list.append(folder_path)
                        count += 1
                    else:
                        for subfolder in os.listdir(folder_path):
                            subfolder_path = os.path.join(folder_path, subfolder)
                            if os.path.isdir(subfolder_path):
                                self.file_list.append(subfolder_path)
                                count += 1
                                if max_maze != None and count >= max_maze:
                                    break
            print(f"Before filtered Loading data from {d} finished, the number of data is {len(self.file_list)}")

        
        assert len(self.file_list) > 0, "No data in the data set"
        if len(self.file_list) % 8 != 0:
            print(f"[Warning] The number of data is not divisible by 8, the number of data is {len(self.file_list)}")
            self.file_list = self.file_list[:len(self.file_list) - len(self.file_list) % 8]

        self.time_step = time_step

        if(verbose):
            print("...finished initializing data set, number of samples: %s\n" % len(self.file_list))

    def __getitem__(self, index):
        path = self.file_list[index]
        if "traj" in path.split("/")[-1] or "path" in path.split("/")[-1]:
            folder_name = os.path.join(path.split("/")[-2], path.split("/")[-1])
            # print(folder_name)
        else:
            folder_name = path.split("/")[-1]
        if "maze" in path:
            raise NotImplementedError
        else:
            return path
    
    def __len__(self):
        return len(self.file_list)




class MazeDataSetRandomActionTest(Dataset):
    def __init__(self, directory, time_step, verbose=False, max_maze=None, folder_verbose=False):
        self.folder_verbose = folder_verbose
        if(verbose):
            print("\nInitializing data set from file: %s..." % directory)
        if folder_verbose:
            print("Folder verbose is on")
        self.file_list = []
        directories = []
        if(isinstance(directory, list)):
            directories.extend(directory)
        else:
            directories.append(directory)
        self.directories = directories
        for d in directories:
            print(f"Loading data from {d}")
            count = 0
            for folder in os.listdir(d):
                folder_path = os.path.join(d, folder)
                if os.path.isdir(folder_path):
                    single_layer_flag = False
                    for file in os.listdir(folder_path):
                        if file == "observations.npy": # while...there must be a observation file right?
                            single_layer_flag = True
                            break
                        if os.path.isdir(os.path.join(folder_path, file)): # if there is a subfolder, then it is not a single layer folder
                            single_layer_flag = False
                            break
                    if max_maze != None and count >= max_maze:
                        break
                    if single_layer_flag:
                        self.file_list.append(folder_path)
                        count += 1
                    else:
                        for subfolder in os.listdir(folder_path):
                            subfolder_path = os.path.join(folder_path, subfolder)
                            if os.path.isdir(subfolder_path):
                                self.file_list.append(subfolder_path)
                                count += 1
                                if max_maze != None and count >= max_maze:
                                    break
            print(f"Loading data from {d} finished, the number of data is {len(self.file_list)}")

        filter_file_list = []
        for file_path in self.file_list:
            if "maze" in file_path:
                filter_file_list.append(file_path)
                continue
            actions_behavior_id = np.load(file_path + '/actions_behavior_id.npy')
            if actions_behavior_id.shape[0] > time_step:
                filter_file_list.append(file_path)
        self.file_list = filter_file_list
        
        assert len(self.file_list) > 0, "No data in the data set"
        if len(self.file_list) % 8 != 0:
            print(f"[Warning] The number of data is not divisible by 8, the number of data is {len(self.file_list)}")
            self.file_list = self.file_list[:len(self.file_list) - len(self.file_list) % 8]
        self.time_step = time_step

        if(verbose):
            print("...finished initializing data set, number of samples: %s\n" % len(self.file_list))

    def __getitem__(self, index):
        path = self.file_list[index]
        if "traj" in path.split("/")[-1] or "path" in path.split("/")[-1]:
            folder_name = os.path.join(path.split("/")[-2], path.split("/")[-1])
            # print(folder_name)
        else:
            folder_name = path.split("/")[-1]
        if "maze" in path:
            # print("dataset loading maze data")
            if self.folder_verbose:
                return self.__get_maze__(index), folder_name
            return self.__get_maze__(index)
        else:
            # print("dataset loading procthor data")
            # if self.folder_verbose:
            #     return self.__get_procthor__(index), folder_name
            # return self.__get_procthor__(index)

            if self.folder_verbose:
                return self.__get_procthor_short__(index), folder_name
            return self.__get_procthor_short__(index)
    
    def __len__(self):
        return len(self.file_list)
    
    def __get_maze__(self, index):
        path = self.file_list[index]
        try:
            cmds = np.load(path + '/commands.npy')
            observations = np.load(path + '/observations.npy')
            actions_behavior_id = np.load(path + '/actions_behavior_id.npy')
            actions_label_id = np.load(path + '/actions_label_id.npy')
            actions_behavior_val = np.load(path + '/actions_behavior_val.npy')
            actions_label_val = np.load(path + '/actions_label_val.npy')
            rewards = np.load(path + '/rewards.npy')
            # bevs = np.load(path + '/BEVs.npy')
            # if os.path.exists(path + '/actions_behavior_prior.npy'):
            #     actions_behavior_prior = np.load(path + '/actions_behavior_prior.npy')
            max_t = actions_behavior_id.shape[0]

            # Shape Check
            assert max_t == rewards.shape[0], f"Shape mismatch: rewards, expected {max_t}, got {rewards.shape[0]}"
            assert max_t == actions_behavior_val.shape[0], f"Shape mismatch: actions_behavior_val, expected {max_t}, got {actions_behavior_val.shape[0]}"
            assert max_t == actions_label_id.shape[0], f"Shape mismatch: actions_label_id, expected {max_t}, got {actions_label_id.shape[0]}"
            assert max_t == actions_label_val.shape[0], f"Shape mismatch: actions_label_val, expected {max_t}, got {actions_label_val.shape[0]}"
            # assert max_t == bevs.shape[0]
            assert max_t + 1 == observations.shape[0], f"Shape mismatch: observations, expected {max_t + 1}, got {observations.shape[0]}"

            if(self.time_step > max_t):
                print(f'[Warning] Load samples from {path} that is shorter ({max_t}) than specified time step ({self.time_step})')
                n_b = 0
                n_e = max_t
            else:
                n_b = 0
                n_e = self.time_step
            cmd_arr = torch.from_numpy(cmds).float()
            
            # Normalize command to [B, 16*16*3]
            if(cmd_arr.dim() == 2): # Normalize to [B，16，16，3]
                cmd_arr = np.repeat(cmd_arr, 256, axis=1)
            elif(cmd_arr.dim() == 4):
                cmd_arr = cmd_arr.reshape(cmd_arr.shape[0], -1)
            
            cmd_arr = cmd_arr[n_b:(n_e)]
            obs_arr = torch.from_numpy(observations[n_b:(n_e + 1)]).float() 
            bact_id_arr = torch.from_numpy(actions_behavior_id[n_b:n_e]).long() 
            lact_id_arr = torch.from_numpy(actions_label_id[n_b:n_e]).long() 
            bact_val_arr = torch.from_numpy(actions_behavior_val[n_b:n_e]).float() 
            lact_val_arr = torch.from_numpy(actions_label_val[n_b:n_e]).float() 
            reward_arr = torch.from_numpy(rewards[n_b:n_e]).float()
            # bev_arr = torch.from_numpy(bevs[n_b:n_e]).float()
            # make the actions_behavior_id and actions_label_id random shuffle in dimension 0
            bact_id_arr = bact_id_arr[torch.randperm(bact_id_arr.shape[0])]
            lact_id_arr = lact_id_arr[torch.randperm(lact_id_arr.shape[0])]
            bact_val_arr = bact_val_arr[torch.randperm(bact_val_arr.shape[0])]
            lact_val_arr = lact_val_arr[torch.randperm(lact_val_arr.shape[0])]
            reward_arr = reward_arr[torch.randperm(reward_arr.shape[0])]

            return cmd_arr, obs_arr, bact_id_arr, lact_id_arr, bact_val_arr, lact_val_arr, reward_arr#, bev_arr
        except Exception as e:
            print(f"Unexpected reading error founded when loading {path}: {e}")
            return None
    
    def __get_procthor_short__(self, index):

        path = self.file_list[index]
        try:
            observations = np.load(path + "/observations.npy").astype(np.uint8)
            actions_behavior_id = np.load(path + "/actions_behavior_id.npy").astype(np.int32)
            actions_behavior_val = np.load(path + "/actions_behavior_val.npy").astype(np.float32)
            actions_label_id = np.load(path + "/actions_label_id.npy").astype(np.int32)
            actions_label_val = np.load(path + "/actions_label_val.npy").astype(np.float32)
            if os.path.exists(path + "/actions_behavior_prior.npy"):
                actions_behavior_prior = np.load(path + "/actions_behavior_prior.npy").astype(np.int32)

            # for i in range(len(actions_label_id)):
            #     if actions_label_id[i][0] == 16:
            #         actions_label_id[i][0] = 17
                    
            rewards = np.load(path + "/rewards.npy").astype(np.float32)
            if os.path.exists(path + "/commands.npy"):
                command = np.load(path + "/commands.npy").astype(np.uint8)
            else:
                assert False, "WE MUST HAVE COMMAND!, No command found in %s" % path
                command = np.zeros((len(observations), 16, 16, 3)).astype(np.uint8)
            
            # 1800 length by now
            # percent = self.time_step / len(observations)
            # assert self.time_step <= 1800, "The length of the trajectory is longer than 1800"
            # get the last time steps' index
            n_b = observations.shape[0] - (self.time_step + 1)
            n_e = observations.shape[0]

            # Ensure that all arrays are of correct dtype
            obs_arr = torch.from_numpy(observations).float()
            bact_id_arr = torch.from_numpy(
                actions_behavior_id
            ).long() 
            bact_val_arr = torch.from_numpy(
                actions_behavior_val
            ).float() 
            lact_id_arr = torch.from_numpy(
                actions_label_id
            ).long() 
            lact_val_arr = torch.from_numpy(
                actions_label_val
            ).float() 
            reward_arr = torch.from_numpy(rewards).float() 
            
            command_arr = torch.from_numpy(command).float() 
            if actions_behavior_prior is not None and len(actions_behavior_prior) > 0:
                bact_prior_arr = torch.from_numpy(actions_behavior_prior).float() 

            # print(obs_arr.shape)
            # print(self.time_step)
            obs_arr = obs_arr.permute(0, 2, 1, 3)
            # rotate the image by 90 degrees 
            obs_arr = torch.rot90(obs_arr, 2, [1, 2])

            # make the actions_behavior_id and actions_label_id random shuffle in dimension 0
            bact_id_arr = bact_id_arr[torch.randperm(bact_id_arr.shape[0])]
            lact_id_arr = lact_id_arr[torch.randperm(lact_id_arr.shape[0])]
            bact_val_arr = bact_val_arr[torch.randperm(bact_val_arr.shape[0])]
            lact_val_arr = lact_val_arr[torch.randperm(lact_val_arr.shape[0])]
            reward_arr = reward_arr[torch.randperm(reward_arr.shape[0])]

            assert obs_arr[n_b:].shape[0] == self.time_step + 1, f"shape mismatch: obs_arr, expected {self.time_step + 1}, got {obs_arr[n_b:].shape[0]}"
            assert bact_id_arr[n_b:-1].shape[0] == self.time_step, f"shape mismatch: bact_id_arr, expected {self.time_step}, got {bact_id_arr[n_b:-1].shape[0]}"
            assert lact_id_arr[n_b:-1].shape[0] == self.time_step, f"shape mismatch: lact_id_arr, expected {self.time_step}, got {lact_id_arr[n_b:-1].shape[0]}"
            assert bact_val_arr[n_b:-1].shape[0] == self.time_step, f"shape mismatch: bact_val_arr, expected {self.time_step}, got {bact_val_arr[n_b:-1].shape[0]}"
            assert lact_val_arr[n_b:-1].shape[0] == self.time_step, f"shape mismatch: lact_val_arr, expected {self.time_step}, got {lact_val_arr[n_b:-1].shape[0]}"
            assert reward_arr[n_b:-1].shape[0] == self.time_step, f"shape mismatch: reward_arr, expected {self.time_step}, got {reward_arr[n_b:-1].shape[0]}"
            assert command_arr[n_b:-1].shape[0] == self.time_step, f"shape mismatch: command_arr, expected {self.time_step}, got {command_arr[n_b:-1].shape[0]}"
            # assert bact_prior_arr.shape[0] == self.time_step

            return (
                # cmd_arr, obs_arr, bact_id_arr, lact_id_arr, bact_val_arr, lact_val_arr, reward_arr
                command_arr[n_b:-1].view(command_arr[n_b:-1].shape[0], -1),
                obs_arr[n_b:],
                bact_id_arr[n_b:-1], # cut the last 'end'
                lact_id_arr[n_b:-1, 0], # lact_id_arr[0:self.time_step],
                bact_val_arr[n_b:-1],
                lact_val_arr[n_b:-1],
                reward_arr[n_b:-1],
                # bact_prior_arr[0:self.time_step]
            )

            # return (
            #     # cmd_arr, obs_arr, bact_id_arr, lact_id_arr, bact_val_arr, lact_val_arr, reward_arr
            #     command_arr[0:self.time_step].view(command_arr[0:self.time_step].shape[0], -1),
            #     obs_arr[0:self.time_step+1],
            #     bact_id_arr[0:self.time_step], # cut the last 'end'
            #     lact_id_arr[0:self.time_step, 0], # lact_id_arr[0:self.time_step],
            #     bact_val_arr[0:self.time_step],
            #     lact_val_arr[0:self.time_step],
            #     reward_arr[0:self.time_step],
            #     # bact_prior_arr[0:self.time_step]
            # )
        except Exception as e:
            print(f"Unexpected reading error founded when loading {path}: {e}")
            return None

    def __get_procthor__(self, index):

        path = self.file_list[index]
        try:
            observations = np.load(path + "/observations.npy").astype(np.uint8)
            actions_behavior_id = np.load(path + "/actions_behavior_id.npy").astype(np.int32)
            actions_behavior_val = np.load(path + "/actions_behavior_val.npy").astype(np.float32)
            actions_label_id = np.load(path + "/actions_label_id.npy").astype(np.int32)
            actions_label_val = np.load(path + "/actions_label_val.npy").astype(np.float32)
            if os.path.exists(path + "/actions_behavior_prior.npy"):
                actions_behavior_prior = np.load(path + "/actions_behavior_prior.npy").astype(np.int32)

            rewards = np.load(path + "/rewards.npy").astype(np.float32)
            if os.path.exists(path + "/commands.npy"):
                command = np.load(path + "/commands.npy").astype(np.uint8)
            # elif os.path.exists(path + "/target.npy"):
            #     command = np.load(path + "/target.npy").astype(np.uint8)
            else:
                assert False, "WE MUST HAVE COMMAND!, No command found in %s" % path
                command = np.zeros((len(observations), 16, 16, 3)).astype(np.uint8)
            
            # FIXED
            for i in range(len(actions_label_id)):
                if actions_label_id[i][1] == 16:
                    actions_label_id[i][1] = 17

            # print(len(observations))
            # 1800 length by now
            percent = self.time_step / len(observations)
            percent = 10000 / len(observations)
            if percent < 1:
                (
                    observations,
                    actions_behavior_id,
                    actions_behavior_val,
                    actions_label_id,
                    actions_label_val,
                    rewards,
                    command,
                    actions_behavior_prior
                ) = cut_data(
                    observations,
                    actions_behavior_id,
                    actions_behavior_val,
                    actions_label_id,
                    actions_label_val,
                    rewards,
                    command,
                    actions_behavior_prior,
                    percentage=percent,
                    time_step=self.time_step,
                )
            else:
                (
                    observations,
                    actions_behavior_id,
                    actions_behavior_val,
                    actions_label_id,
                    actions_label_val,
                    rewards,
                    command,
                    actions_behavior_prior
                ) = expend_data(
                    observations,
                    actions_behavior_id,
                    actions_behavior_val,
                    actions_label_id,
                    actions_label_val,
                    rewards,
                    command,
                    actions_behavior_prior, # TODO
                    percentage=percent,
                )


            # Ensure that all arrays are of correct dtype
            obs_arr = torch.from_numpy(observations).float()
            bact_id_arr = torch.from_numpy(
                actions_behavior_id
            ).long() 
            bact_val_arr = torch.from_numpy(
                actions_behavior_val
            ).float() 
            lact_id_arr = torch.from_numpy(
                actions_label_id
            ).long() 
            lact_val_arr = torch.from_numpy(
                actions_label_val
            ).float() 
            reward_arr = torch.from_numpy(rewards).float() 
            
            command_arr = torch.from_numpy(command).float() 
            if actions_behavior_prior is not None and len(actions_behavior_prior) > 0:
                bact_prior_arr = torch.from_numpy(actions_behavior_prior).float() 

            # print(obs_arr.shape)
            # print(self.time_step)
            obs_arr = obs_arr.permute(0, 2, 1, 3)
            # rotate the image by 90 degrees 
            obs_arr = torch.rot90(obs_arr, 2, [1, 2])
            # return (
            #     # cmd_arr, obs_arr, bact_id_arr, lact_id_arr, bact_val_arr, lact_val_arr, reward_arr
            #     command_arr[0:1800].view(command_arr[0:1800].shape[0], -1),
            #     obs_arr[0:1801],
            #     bact_id_arr[0:1800], # cut the last 'end'
            #     lact_id_arr[0:1800, 0], # lact_id_arr[0:self.time_step],
            #     bact_val_arr[0:1800],
            #     lact_val_arr[0:1800],
            #     reward_arr[0:1800],
            #     # bact_prior_arr[0:self.time_step]
            # )
            return (
                # cmd_arr, obs_arr, bact_id_arr, lact_id_arr, bact_val_arr, lact_val_arr, reward_arr
                command_arr[0:self.time_step].view(command_arr[0:self.time_step].shape[0], -1),
                obs_arr[0:self.time_step+1],
                bact_id_arr[0:self.time_step], # cut the last 'end'
                lact_id_arr[0:self.time_step, 1], # lact_id_arr[0:self.time_step],
                bact_val_arr[0:self.time_step],
                lact_val_arr[0:self.time_step],
                reward_arr[0:self.time_step],
                # bact_prior_arr[0:self.time_step]
            )
        except Exception as e:
            print(f"Unexpected reading error founded when loading {path}: {e}")
            return None






class ProcthorDataSet(Dataset):
    def __init__(self, directory, time_step, verbose=False, max_maze=None, folder_verbose=False):
        self.folder_verbose = folder_verbose
        if(verbose):
            print("\nInitializing data set from file: %s..." % directory)
        if folder_verbose:
            print("Folder verbose is on")
        self.file_list = []
        directories = []
        if(isinstance(directory, list)):
            directories.extend(directory)
        else:
            directories.append(directory)
        self.directories = directories
        for d in directories:
            print(f"Loading data from {d}")
            count = 0
            for folder in os.listdir(d):
                folder_path = os.path.join(d, folder)
                if os.path.isdir(folder_path):
                    single_layer_flag = False
                    for file in os.listdir(folder_path):
                        if file == "observations.npy": # while...there must be a observation file right?
                            single_layer_flag = True
                            break
                        if os.path.isdir(os.path.join(folder_path, file)): # if there is a subfolder, then it is not a single layer folder
                            single_layer_flag = False
                            break
                    if max_maze != None and count >= max_maze:
                        break
                    if single_layer_flag:
                        self.file_list.append(folder_path)
                        count += 1
                    else:
                        for subfolder in os.listdir(folder_path):
                            subfolder_path = os.path.join(folder_path, subfolder)
                            if os.path.isdir(subfolder_path):
                                self.file_list.append(subfolder_path)
                                count += 1
                                if max_maze != None and count >= max_maze:
                                    break
            print(f"Loading data from {d} finished, the number of data is {len(self.file_list)}")
            # file_list = os.listdir(d)
            # self.file_list.extend([os.path.join(d, file) for file in file_list])
        if len(self.file_list) % 8 != 0:
            print(f"[Warning] The number of data is not divisible by 8, the number of data is {len(self.file_list)}")
            self.file_list = self.file_list[:len(self.file_list) - len(self.file_list) % 8]
        self.time_step = time_step

        if(verbose):
            print("...finished initializing data set, number of samples: %s\n" % len(self.file_list))

    def __getitem__(self, index):

        import pickle
        # path = self.file_list[index]
        # return path

        path = self.file_list[index]
        # if "traj" in path.split("/")[-1] or "path" in path.split("/")[-1]:
        #     folder_name = os.path.join(path.split("/")[-2], path.split("/")[-1])
        #     # print(folder_name)
        # else:
        #     folder_name = path.split("/")[-1]
        # if "maze" in path:
        #     print(f"Maze dataset in {path} is not supported in ProcthorDataSet, please use MazeDataSet instead")
        #     return None
        #     # if self.folder_verbose:
        #     #     return self.__get_maze__(index), folder_name
        #     # return self.__get_maze__(index)
        # else:
        #     # print("dataset loading procthor data")
        if self.folder_verbose:
            
            return self.__get_procthor__(index), path
        return self.__get_procthor__(index)
    
    def __len__(self):
        return len(self.file_list)

    def __get_maze__(self, index):
        path = self.file_list[index]
        try:
            cmds = np.load(path + '/commands.npy')
            observations = np.load(path + '/observations.npy')
            actions_behavior_id = np.load(path + '/actions_behavior_id.npy')
            actions_label_id = np.load(path + '/actions_label_id.npy')
            actions_behavior_val = np.load(path + '/actions_behavior_val.npy')
            actions_label_val = np.load(path + '/actions_label_val.npy')
            rewards = np.load(path + '/rewards.npy')
            # bevs = np.load(path + '/BEVs.npy')
            # if os.path.exists(path + '/actions_behavior_prior.npy'):
            #     actions_behavior_prior = np.load(path + '/actions_behavior_prior.npy')
            max_t = actions_behavior_id.shape[0]

            # Shape Check
            assert max_t == rewards.shape[0]
            assert max_t == actions_behavior_val.shape[0]
            assert max_t == actions_label_id.shape[0]
            assert max_t == actions_label_val.shape[0]
            # assert max_t == bevs.shape[0]
            assert max_t + 1 == observations.shape[0]

            if(self.time_step > max_t):
                print(f'[Warning] Load samples from {path} that is shorter ({max_t}) than specified time step ({self.time_step})')
                n_b = 0
                n_e = max_t
            else:
                n_b = 0
                n_e = self.time_step
            cmd_arr = torch.from_numpy(cmds).float()
            
            # Normalize command to [B, 16*16*3]
            if(cmd_arr.dim() == 2): # Normalize to [B，16，16，3]
                cmd_arr = np.repeat(cmd_arr, 256, axis=1)
            elif(cmd_arr.dim() == 4):
                cmd_arr = cmd_arr.reshape(cmd_arr.shape[0], -1)
            
            cmd_arr = cmd_arr[n_b:(n_e)]
            obs_arr = torch.from_numpy(observations[n_b:(n_e + 1)]).float() 
            bact_id_arr = torch.from_numpy(actions_behavior_id[n_b:n_e]).long() 
            lact_id_arr = torch.from_numpy(actions_label_id[n_b:n_e]).long() 
            bact_val_arr = torch.from_numpy(actions_behavior_val[n_b:n_e]).float() 
            lact_val_arr = torch.from_numpy(actions_label_val[n_b:n_e]).float() 
            reward_arr = torch.from_numpy(rewards[n_b:n_e]).float()
            # bev_arr = torch.from_numpy(bevs[n_b:n_e]).float()
            
            return cmd_arr, obs_arr, bact_id_arr, lact_id_arr, bact_val_arr, lact_val_arr, reward_arr#, bev_arr
        except Exception as e:
            print(f"Unexpected reading error founded when loading {path}: {e}")
            return None
    
    
    def __get_procthor__(self, index):

        path = self.file_list[index]
        try:
            observations = np.load(path + "/observations.npy").astype(np.uint8)
            actions_behavior_id = np.load(path + "/actions_behavior_id.npy").astype(np.int32)
            actions_behavior_val = np.load(path + "/actions_behavior_val.npy").astype(np.float32)
            actions_label_id = np.load(path + "/actions_label_id.npy").astype(np.int32)
            actions_label_val = np.load(path + "/actions_label_val.npy").astype(np.float32)
            if os.path.exists(path + "/actions_behavior_prior.npy"):
                actions_behavior_prior = np.load(path + "/actions_behavior_prior.npy").astype(np.int32)

            rewards = np.load(path + "/rewards.npy").astype(np.float32)
            if os.path.exists(path + "/commands.npy"):
                command = np.load(path + "/commands.npy").astype(np.uint8)
            else:
                assert False, "WE MUST HAVE COMMAND!, No command found in %s" % path
                command = np.zeros((len(observations), 16, 16, 3)).astype(np.uint8)
            
            # print(len(observations))

            percent = self.time_step / len(observations)
            if percent < 1:
                (
                    observations,
                    actions_behavior_id,
                    actions_behavior_val,
                    actions_label_id,
                    actions_label_val,
                    rewards,
                    command,
                    actions_behavior_prior
                ) = cut_data(
                    observations,
                    actions_behavior_id,
                    actions_behavior_val,
                    actions_label_id,
                    actions_label_val,
                    rewards,
                    command,
                    actions_behavior_prior,
                    percentage=percent,
                    time_step=self.time_step,
                )
            else:
                (
                    observations,
                    actions_behavior_id,
                    actions_behavior_val,
                    actions_label_id,
                    actions_label_val,
                    rewards,
                    command,
                    actions_behavior_prior
                ) = expend_data(
                    observations,
                    actions_behavior_id,
                    actions_behavior_val,
                    actions_label_id,
                    actions_label_val,
                    rewards,
                    command,
                    actions_behavior_prior, # TODO
                    percentage=percent,
                )


            # Ensure that all arrays are of correct dtype
            obs_arr = torch.from_numpy(observations).float()
            bact_id_arr = torch.from_numpy(
                actions_behavior_id
            ).long() 
            bact_val_arr = torch.from_numpy(
                actions_behavior_val
            ).float() 
            lact_id_arr = torch.from_numpy(
                actions_label_id
            ).long() 
            lact_val_arr = torch.from_numpy(
                actions_label_val
            ).float() 
            reward_arr = torch.from_numpy(rewards).float() 
            
            command_arr = torch.from_numpy(command).float() 
            if actions_behavior_prior is not None and len(actions_behavior_prior) > 0:
                bact_prior_arr = torch.from_numpy(actions_behavior_prior).float() 

            # print(obs_arr.shape)
            # print(self.time_step)
            obs_arr = obs_arr.permute(0, 2, 1, 3)
            # rotate the image by 90 degrees 
            obs_arr = torch.rot90(obs_arr, 2, [1, 2])
            return (
                # cmd_arr, obs_arr, bact_id_arr, lact_id_arr, bact_val_arr, lact_val_arr, reward_arr
                command_arr[0:self.time_step].view(command_arr[0:self.time_step].shape[0], -1),
                obs_arr[0:self.time_step+1],
                bact_id_arr[0:self.time_step], # cut the last 'end'
                lact_id_arr[0:self.time_step, 0], # lact_id_arr[0:self.time_step],
                bact_val_arr[0:self.time_step],
                lact_val_arr[0:self.time_step],
                reward_arr[0:self.time_step],
                # bact_prior_arr[0:self.time_step]
            )
        except Exception as e:
            print(f"Unexpected reading error founded when loading {path}: {e}")
            return None




class MazeTaskDataSet(Dataset):
    def __init__(self, directory, time_step, verbose=False, max_maze=None, world_size=8, folder_verbose=False):
        self.folder_verbose = folder_verbose
        if(verbose):
            print("\nInitializing data set from file: %s..." % directory)
        if folder_verbose:
            print("Folder verbose is on")
        self.file_list = []
        directories = []
        if(isinstance(directory, list)):
            directories.extend(directory)
        else:
            directories.append(directory)
        self.directories = directories
        for d in directories:
            print(f"Loading data from {d}")
            count = 0
            for folder in os.listdir(d):
                folder_path = os.path.join(d, folder)
                if os.path.isdir(folder_path):
                    single_layer_flag = False
                    for file in os.listdir(folder_path):
                        if file == "observations.npy" or file == "task.pkl": # while...there must be a observation file right?
                            single_layer_flag = True
                            break
                        if os.path.isdir(os.path.join(folder_path, file)): # if there is a subfolder, then it is not a single layer folder
                            single_layer_flag = False
                            break
                    if max_maze != None and count >= max_maze:
                        break
                    if single_layer_flag:
                        self.file_list.append(folder_path)
                        count += 1
                    else:
                        for subfolder in os.listdir(folder_path):
                            subfolder_path = os.path.join(folder_path, subfolder)
                            if os.path.isdir(subfolder_path):
                                self.file_list.append(subfolder_path)
                                count += 1
                                if max_maze != None and count >= max_maze:
                                    break
            print(f"Loading data from {d} finished, the number of data is {len(self.file_list)}")
            # file_list = os.listdir(d)
            # self.file_list.extend([os.path.join(d, file) for file in file_list])
        if len(self.file_list) % world_size != 0:
            print(f"[Warning] The number of data is not divisible by {world_size}, the number of data is {len(self.file_list)}")
            self.file_list = self.file_list[:len(self.file_list) - len(self.file_list) % world_size]
        self.time_step = time_step

        if(verbose):
            print("...finished initializing data set, number of samples: %s\n" % len(self.file_list))
    def __getitem__(self, index):
        import pickle
        path = self.file_list[index]
        task_path = os.path.join(path, "task.pkl")
        # task = pickle.load(open(path + '/task.pkl', 'rb'))
        if "traj" in path.split("/")[-1] or "path" in path.split("/")[-1]:
            folder_name = os.path.join(path.split("/")[-2], path.split("/")[-1])
            # print(folder_name)
        else:
            folder_name = path.split("/")[-1]
        if self.folder_verbose:
            return task_path, folder_name
        return task_path
    def __len__(self):
        return len(self.file_list)

class MazeDataSetShort(Dataset):
    def __init__(self, directory, time_step, verbose=False, max_maze=None, folder_verbose=False):
        self.folder_verbose = folder_verbose
        if(verbose):
            print("\nInitializing data set from file: %s..." % directory)
        if folder_verbose:
            print("Folder verbose is on")
        self.file_list = []
        directories = []
        if(isinstance(directory, list)):
            directories.extend(directory)
        else:
            directories.append(directory)
        self.directories = directories
        for d in directories:
            print(f"Loading data from {d}")
            count = 0
            for folder in os.listdir(d):
                folder_path = os.path.join(d, folder)
                if os.path.isdir(folder_path):
                    single_layer_flag = False
                    for file in os.listdir(folder_path):
                        if file == "observations.npy": # while...there must be a observation file right?
                            single_layer_flag = True
                            break
                        if os.path.isdir(os.path.join(folder_path, file)): # if there is a subfolder, then it is not a single layer folder
                            single_layer_flag = False
                            break
                    if max_maze != None and count >= max_maze:
                        break
                    if single_layer_flag:
                        self.file_list.append(folder_path)
                        count += 1
                    else:
                        for subfolder in os.listdir(folder_path):
                            subfolder_path = os.path.join(folder_path, subfolder)
                            if os.path.isdir(subfolder_path):
                                self.file_list.append(subfolder_path)
                                count += 1
                                if max_maze != None and count >= max_maze:
                                    break
            print(f"Loading data from {d} finished, the number of data is {len(self.file_list)}")
            # file_list = os.listdir(d)
            # self.file_list.extend([os.path.join(d, file) for file in file_list])
        if len(self.file_list) % 4 != 0:
            print(f"[Warning] The number of data is not divisible by 4, the number of data is {len(self.file_list)}")
            self.file_list = self.file_list[:len(self.file_list) - len(self.file_list) % 4]
        self.time_step = time_step
        self.cutting_length = 100
        if(verbose):
            print("...finished initializing data set, number of samples: %s\n" % len(self.file_list))

    def __getitem__(self, index):
        cutting_length = self.cutting_length
        true_index = int(index // cutting_length)
        path = self.file_list[true_index]
        if "traj" in path.split("/")[-1] or "path" in path.split("/")[-1]:
            folder_name = os.path.join(path.split("/")[-2], path.split("/")[-1])
        else:
            folder_name = path.split("/")[-1]
        if "maze" in path:
            # print("dataset loading maze data")
            if self.folder_verbose:
                return self.__get_maze__(index), folder_name
            return self.__get_maze__(index)
        else:
            # print("dataset loading procthor data")
            if self.folder_verbose:
                return self.__get_procthor__(index), folder_name
            return self.__get_procthor__(index)
    
    def __len__(self):
        if self.time_step <= self.cutting_length:
            return len(self.file_list)*self.cutting_length
        return len(self.file_list)
    
    def __get_maze__(self, index):
        cutting_length = self.cutting_length
        true_index = int(index // cutting_length)
        overflow = index % cutting_length
        assert true_index*cutting_length + overflow == index
        path = self.file_list[true_index]
        try:
            cmds = np.load(path + '/commands.npy')
            observations = np.load(path + '/observations.npy')
            actions_behavior_id = np.load(path + '/actions_behavior_id.npy')
            actions_label_id = np.load(path + '/actions_label_id.npy')
            actions_behavior_val = np.load(path + '/actions_behavior_val.npy')
            actions_label_val = np.load(path + '/actions_label_val.npy')
            rewards = np.load(path + '/rewards.npy')
            # bevs = np.load(path + '/BEVs.npy')
            # if os.path.exists(path + '/actions_behavior_prior.npy'):
            #     actions_behavior_prior = np.load(path + '/actions_behavior_prior.npy')
            max_t = actions_behavior_id.shape[0]

            # Shape Check
            assert max_t == rewards.shape[0]
            assert max_t == actions_behavior_val.shape[0]
            assert max_t == actions_label_id.shape[0]
            assert max_t == actions_label_val.shape[0]
            # assert max_t == bevs.shape[0]
            assert max_t + 1 == observations.shape[0]

            if(self.time_step > max_t):
                print(f'[Warning] Load samples from {path} that is shorter ({max_t}) than specified time step ({self.time_step})')
                n_b = 0
                n_e = max_t
            else:
                n_b = 0
                n_e = self.time_step
            # 101, maze 1, (100, 200), 
            # print(f"true_index: {true_index}, cutting_length: {cutting_length}, overflow: {overflow}")
            if self.time_step <= self.cutting_length:
                n_b = int(overflow) * cutting_length
                n_e = n_b + self.cutting_length
            # print(f"n_b: {n_b}, n_e: {n_e}")
            cmd_arr = torch.from_numpy(cmds).float()
            
            # Normalize command to [B, 16*16*3]
            if(cmd_arr.dim() == 2): # Normalize to [B，16，16，3]
                cmd_arr = np.repeat(cmd_arr, 256, axis=1)
            elif(cmd_arr.dim() == 4):
                cmd_arr = cmd_arr.reshape(cmd_arr.shape[0], -1)
            
            cmd_arr = cmd_arr[n_b:(n_e)]
            obs_arr = torch.from_numpy(observations[n_b:(n_e + 1)]).float() 
            bact_id_arr = torch.from_numpy(actions_behavior_id[n_b:n_e]).long() 
            lact_id_arr = torch.from_numpy(actions_label_id[n_b:n_e]).long() 
            bact_val_arr = torch.from_numpy(actions_behavior_val[n_b:n_e]).float() 
            lact_val_arr = torch.from_numpy(actions_label_val[n_b:n_e]).float() 
            reward_arr = torch.from_numpy(rewards[n_b:n_e]).float()
            # bev_arr = torch.from_numpy(bevs[n_b:n_e]).float()
            
            return cmd_arr, obs_arr, bact_id_arr, lact_id_arr, bact_val_arr, lact_val_arr, reward_arr#, bev_arr
        except Exception as e:
            print(f"Unexpected reading error founded when loading {path}: {e}")
            return None
    def __get_procthor__(self, index):
        cutting_length = self.cutting_length
        true_index = int(index // cutting_length)
        overflow = index % cutting_length
        assert true_index*cutting_length + overflow == index
        path = self.file_list[true_index]
        try:
            observations = np.load(path + "/observations.npy").astype(np.uint8)
            actions_behavior_id = np.load(path + "/actions_behavior_id.npy").astype(np.int32)
            actions_behavior_val = np.load(path + "/actions_behavior_val.npy").astype(np.float32)
            actions_label_id = np.load(path + "/actions_label_id.npy").astype(np.int32)
            actions_label_val = np.load(path + "/actions_label_val.npy").astype(np.float32)
            if os.path.exists(path + "/actions_behavior_prior.npy"):
                actions_behavior_prior = np.load(path + "/actions_behavior_prior.npy").astype(np.int32)

            rewards = np.load(path + "/rewards.npy").astype(np.float32)
            if os.path.exists(path + "/commands.npy"):
                command = np.load(path + "/commands.npy").astype(np.uint8)
            elif os.path.exists(path + "/target.npy"):
                command = np.load(path + "/target.npy").astype(np.uint8)
            else:
                assert False, "WE MUST HAVE COMMAND!, No command found in %s" % path
                command = np.zeros((len(observations), 16, 16, 3)).astype(np.uint8)
            
            # print(len(observations))

            percent = self.time_step / len(observations)
            if percent < 1:
                (
                    observations,
                    actions_behavior_id,
                    actions_behavior_val,
                    actions_label_id,
                    actions_label_val,
                    rewards,
                    command,
                    actions_behavior_prior
                ) = cut_data(
                    observations,
                    actions_behavior_id,
                    actions_behavior_val,
                    actions_label_id,
                    actions_label_val,
                    rewards,
                    command,
                    actions_behavior_prior,
                    percentage=percent,
                    time_step=self.time_step,
                )
            else:
                (
                    observations,
                    actions_behavior_id,
                    actions_behavior_val,
                    actions_label_id,
                    actions_label_val,
                    rewards,
                    command,
                    actions_behavior_prior
                ) = expend_data(
                    observations,
                    actions_behavior_id,
                    actions_behavior_val,
                    actions_label_id,
                    actions_label_val,
                    rewards,
                    command,
                    actions_behavior_prior, # TODO
                    percentage=percent,
                )


            # Ensure that all arrays are of correct dtype
            obs_arr = torch.from_numpy(observations).float()
            bact_id_arr = torch.from_numpy(
                actions_behavior_id
            ).long() 
            bact_val_arr = torch.from_numpy(
                actions_behavior_val
            ).float() 
            lact_id_arr = torch.from_numpy(
                actions_label_id
            ).long() 
            lact_val_arr = torch.from_numpy(
                actions_label_val
            ).float() 
            reward_arr = torch.from_numpy(rewards).float() 
            
            command_arr = torch.from_numpy(command).float() 
            if actions_behavior_prior is not None and len(actions_behavior_prior) > 0:
                bact_prior_arr = torch.from_numpy(actions_behavior_prior).float() 

            # print(obs_arr.shape)
            # print(self.time_step)
            obs_arr = obs_arr.permute(0, 2, 1, 3)
            # rotate the image by 90 degrees 
            obs_arr = torch.rot90(obs_arr, 2, [1, 2])
            return (
                # cmd_arr, obs_arr, bact_id_arr, lact_id_arr, bact_val_arr, lact_val_arr, reward_arr
                command_arr[0:self.time_step].view(command_arr[0:self.time_step].shape[0], -1),
                obs_arr[0:self.time_step+1],
                bact_id_arr[0:self.time_step], # cut the last 'end'
                lact_id_arr[0:self.time_step, 0], # lact_id_arr[0:self.time_step],
                bact_val_arr[0:self.time_step],
                lact_val_arr[0:self.time_step],
                reward_arr[0:self.time_step],
                # bact_prior_arr[0:self.time_step]
            )
        except Exception as e:
            print(f"Unexpected reading error founded when loading {path}: {e}")
            return None





class ProcthorPaddingDataSet(Dataset):
    def __init__(self, directory, time_step, verbose=False, max_maze=None, folder_verbose=False, use_feat=False):
        self.folder_verbose = folder_verbose
        if(verbose):
            print("\nInitializing data set from file: %s..." % directory)
        if folder_verbose:
            print("Folder verbose is on")
        self.file_list = []
        directories = []
        if(isinstance(directory, list)):
            directories.extend(directory)
        else:
            directories.append(directory)
        self.directories = directories
        count = 0
        if use_feat:
            self.state_file_name = "vision_feat.npy"
        else:
            self.state_file_name = "observations.npy"
        for d in directories:
            print(f"Loading data from {d}")
            for folder in os.listdir(d):
                folder_path = os.path.join(d, folder)
                if os.path.isdir(folder_path):
                    single_layer_flag = False
                    for file in os.listdir(folder_path):
                        if file == self.state_file_name: # while...there must be a observation file right?
                            single_layer_flag = True
                            break
                        if os.path.isdir(os.path.join(folder_path, file)): # if there is a subfolder, then it is not a single layer folder
                            single_layer_flag = False
                            break
                    if max_maze != None and count >= max_maze:
                        break
                    if single_layer_flag:
                        self.file_list.append(folder_path)
                        count += 1
                    else:
                        for subfolder in os.listdir(folder_path):
                            subfolder_path = os.path.join(folder_path, subfolder)
                            if os.path.isdir(subfolder_path):
                                self.file_list.append(subfolder_path)
                                count += 1
                                if max_maze != None and count >= max_maze:
                                    break
            print(f"Before filtered Loading data from {d} finished, the number of data is {len(self.file_list)}")

        filter_file_list = []
        self.time_interval = None
        if isinstance(time_step, list):
            assert len(time_step) == 2, "time_step must be a list with length 2"
            self.time_interval = time_step.copy()
            time_step = time_step[1] - time_step[0]
        self.time_step = time_step

        for file_path in self.file_list:
            if "maze" in file_path:
                filter_file_list.append(file_path)
                continue
            actions_behavior_id = np.load(file_path + '/actions_behavior_id.npy')
            if actions_behavior_id.shape[0] >= 5:
                filter_file_list.append(file_path)
        self.file_list = filter_file_list
        
        assert len(self.file_list) > 0, "No data in the data set"
        if len(self.file_list) % 8 != 0:
            print(f"[Warning] The number of data is not divisible by 8, the number of data is {len(self.file_list)}")
            self.file_list = self.file_list[:len(self.file_list) - len(self.file_list) % 8]



        if(verbose):
            print("...finished initializing data set, number of samples: %s\n" % len(self.file_list))

    def __getitem__(self, index):
        path = self.file_list[index]
        if "traj" in path.split("/")[-1] or "path" in path.split("/")[-1]:
            folder_name = os.path.join(path.split("/")[-2], path.split("/")[-1])
        else:
            folder_name = path.split("/")[-1]
        
        if "maze" in path:
            raise print("")
        else:
            result = self.__get_procthor_short__(index)
            if self.folder_verbose:
                return result, path
            return result
    
    def __len__(self):
        return len(self.file_list)
    def __get_procthor_short__(self, index):
        path = self.file_list[index]
        try:
                observations = np.load(path + f"/{self.state_file_name}").astype(np.float32)
                actions_behavior_id = np.load(path + "/actions_behavior_id.npy").astype(np.int32)
                actions_behavior_val = np.load(path + "/actions_behavior_val.npy").astype(np.float32)
                actions_label_id = np.load(path + "/actions_label_id.npy").astype(np.int32)
                actions_label_val = np.load(path + "/actions_label_val.npy").astype(np.float32)
                actions_behavior_prior = None
                if os.path.exists(path + "/actions_behavior_prior.npy"):
                    actions_behavior_prior = np.load(path + "/actions_behavior_prior.npy").astype(np.int32)

                rewards = np.load(path + "/rewards.npy").astype(np.float32)
                if os.path.exists(path + "/commands.npy"):
                    command = np.load(path + "/commands.npy").astype(np.uint8)
                else:
                    assert False, "WE MUST HAVE COMMAND!, No command found in %s" % path
                    command = np.zeros((len(observations), 16, 16, 3)).astype(np.uint8)
                
                max_t = actions_behavior_id.shape[0]
                
                # 处理变长序列
                padding_length = 0
                if self.time_interval is not None:
                    n_b = self.time_interval[0]
                    n_e = self.time_interval[1]
                    if n_e > max_t:
                        padding_length = n_e - max_t
                else:
                    if max_t < self.time_step:
                        n_b = 0
                        n_e = self.time_step
                        padding_length = self.time_step - max_t
                    else:
                        n_b = max_t - self.time_step
                        n_e = max_t
                assert n_b >= 0 and n_e - n_b == self.time_step, f"n_b: {n_b}, n_e: {n_e}, max_t: {max_t}, self.time_step: {self.time_step}"

                # 获取实际数据
                actual_obs = torch.from_numpy(observations).float()
                actual_bact_id = torch.from_numpy(actions_behavior_id).long()
                actual_bact_val = torch.from_numpy(actions_behavior_val).float()
                actual_lact_id = torch.from_numpy(actions_label_id).long()
                actual_lact_val = torch.from_numpy(actions_label_val).float()
                actual_reward = torch.from_numpy(rewards).float()
                actual_command = torch.from_numpy(command).float()
                
                # # padding 
                # for i in range(padding_length):
                #     padding_obs = - torch.ones_like(actual_obs[0])
                #     actual_obs = torch.cat((actual_obs, padding_obs.unsqueeze(0)), dim=0)
                #     padding_action_id = -1
                #     padding_action_val = - torch.ones_like(actual_bact_val[0])
                #     actual_bact_id = torch.cat((actual_bact_id, padding_action_id.unsqueeze(0)), dim=0)
                #     actual_bact_val = torch.cat((actual_bact_val, padding_action_val.unsqueeze(0)), dim=0)
                #     padding_laction_val = - torch.ones_like(actual_lact_val[0])
                #     actual_lact_id = torch.cat((actual_bact_id, padding_action_id.unsqueeze(0)), dim=0)
                #     actual_lact_val = torch.cat((actual_bact_val, padding_laction_val.unsqueeze(0)), dim=0)
                #     padding_reward = - torch.ones_like(actual_reward[0])
                #     actual_reward = torch.cat((actual_reward, padding_reward.unsqueeze(0)), dim=0)
                #     padding_command = - torch.ones_like(actual_command[0])
                #     actual_command = torch.cat((actual_command, padding_command.unsqueeze(0)), dim=0)

                if padding_length > 0:
                    # Pad observations
                    actual_obs = torch.cat([actual_obs] + [-torch.ones_like(actual_obs[0]).unsqueeze(0)] * padding_length, dim=0)
                    # Pad behavior action IDs (scalar)
                    actual_bact_id = torch.cat([actual_bact_id] + [-torch.ones_like(actual_bact_id[0]).unsqueeze(0)] * padding_length, dim=0)
                    # Pad behavior action values
                    actual_bact_val = torch.cat([actual_bact_val] + [-torch.ones_like(actual_bact_val[0]).unsqueeze(0)] * padding_length, dim=0)
                    # Pad label action IDs
                    actual_lact_id = torch.cat([actual_lact_id] + [-torch.ones_like(actual_lact_id[0]).unsqueeze(0)] * padding_length, dim=0)
                    # Pad label action values
                    actual_lact_val = torch.cat([actual_lact_val] + [-torch.ones_like(actual_lact_val[0]).unsqueeze(0)] * padding_length, dim=0)
                    # Pad rewards
                    actual_reward = torch.cat([actual_reward] + [-torch.ones_like(actual_reward[0]).unsqueeze(0)] * padding_length, dim=0)
                    # Pad commands
                    actual_command = torch.cat([actual_command] + [-torch.ones_like(actual_command[0]).unsqueeze(0)] * padding_length, dim=0)

                # # 应用旋转
                # actual_obs = actual_obs.permute(0, 2, 1, 3)
                # actual_obs = torch.rot90(actual_obs, 2, [1, 2])

                # 切片获取实际数据
                sliced_obs = actual_obs[n_b:n_e + 1]
                sliced_bact_id = actual_bact_id[n_b:n_e]
                sliced_lact_id = actual_lact_id[n_b:n_e, 0]  # 只取第一个维度
                sliced_bact_val = actual_bact_val[n_b:n_e]
                sliced_lact_val = actual_lact_val[n_b:n_e]
                sliced_reward = actual_reward[n_b:n_e]
                sliced_command = actual_command[n_b:n_e]

                return (sliced_command.view(sliced_command.shape[0], -1),
                        sliced_obs,
                        sliced_bact_id,
                        sliced_lact_id,
                        sliced_bact_val,
                        sliced_lact_val,
                        sliced_reward)
        except Exception as e:
            print(f"Unexpected reading error founded when loading {path}: {e}")
            return None

def procthor_collate_fn(batch):
    """处理变长序列的collate_fn函数[1,2](@ref)"""
    # 过滤掉None值
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    
    # 检查是否是带有文件夹名的返回格式
    if isinstance(batch[0], tuple) and len(batch[0]) == 2 and isinstance(batch[0][1], str):
        # 带有文件夹名的情况
        data_items = [item[0] for item in batch]
        folder_names = [item[1] for item in batch]
        return_data = True
    else:
        data_items = batch
        folder_names = None
        return_data = False
    
    # 从第一个样本推断time_step
    time_step = None
    for item in data_items:
        if len(item) >= 3 and hasattr(item[2], 'shape') and len(item[2].shape) > 0:
            time_step = item[2].shape[0] if time_step is None else max(time_step, item[2].shape[0])
            break
    
    if time_step is None:
        raise ValueError("Cannot infer time_step from batch data")
    
    # 初始化结果列表
    padded_results = []
    
    # 对每个数据字段进行处理
    num_fields = len(data_items[0])
    for field_idx in range(num_fields):
        field_data = [item[field_idx] for item in data_items]
        
        # 获取当前字段的形状信息
        sample_shape = field_data[0].shape
        field_rank = len(sample_shape)  # 张量的维度数
        
        # 对于observations(索引1)，需要特殊处理，因为它的长度比其他字段多1[1](@ref)
        if field_idx == 1:  # observations字段
            target_length = time_step + 1
        else:
            target_length = time_step
        
        # 对每个样本进行处理
        padded_field = []
        for data in field_data:
            actual_len = data.shape[0]
            
            # 如果实际长度小于目标长度，需要填充[8](@ref)
            if actual_len < target_length:
                pad_length = target_length - actual_len
                
                # 根据字段的维度构建padding形状[8](@ref)
                if field_rank == 1:
                    # 1D张量: [length]
                    pad_shape = (0, pad_length)
                elif field_rank == 2:
                    # 2D张量: [length, features]
                    pad_shape = (0, 0, 0, pad_length)
                elif field_rank == 4:
                    # 4D张量: [length, channels, height, width]
                    pad_shape = (0, 0, 0, 0, 0, pad_length)
                else:
                    # 通用情况: 只在第一个维度(时间维度)填充
                    pad_shape = (0, 0) * (field_rank - 1) + (0, pad_length)
                
                # 使用-1进行填充[8](@ref)
                padded_data = torch.nn.functional.pad(data, pad_shape, value=-1)
                padded_field.append(padded_data)
            elif actual_len > target_length:
                # 如果实际长度大于目标长度，取最后target_length个时间步
                if field_rank == 1:
                    padded_data = data[-target_length:]
                elif field_rank == 2:
                    padded_data = data[-target_length:, :]
                elif field_rank == 4:
                    padded_data = data[-target_length:, :, :, :]
                else:
                    slices = [slice(-target_length, None)] + [slice(None)] * (field_rank - 1)
                    padded_data = data[slices]
                padded_field.append(padded_data)
            else:
                # 长度正好，直接使用
                padded_field.append(data)
        
        # 将填充后的数据堆叠成batch
        try:
            stacked_data = torch.stack(padded_field)
            padded_results.append(stacked_data)
        except Exception as e:
            print(f"Error stacking field {field_idx}: {e}")
            # 如果stack失败，尝试其他方法
            try:
                # 对于不同形状的数据，使用pad_sequence[2,8](@ref)
                from torch.nn.utils.rnn import pad_sequence
                if field_rank == 1:
                    stacked_data = pad_sequence(padded_field, batch_first=True, padding_value=-1)
                else:
                    # 对于高维数据，手动处理
                    max_shape = [target_length] + list(padded_field[0].shape[1:])
                    stacked_data = torch.full((len(padded_field),) + tuple(max_shape), -1.0, dtype=padded_field[0].dtype)
                    for i, data in enumerate(padded_field):
                        if field_rank == 2:
                            stacked_data[i, :data.shape[0]] = data
                        elif field_rank == 4:
                            stacked_data[i, :data.shape[0], :, :, :] = data
                padded_results.append(stacked_data)
            except Exception as e2:
                print(f"Alternative padding also failed for field {field_idx}: {e2}")
                padded_results.append(padded_field)
    
    # 如果有文件夹名，一起返回
    if return_data and folder_names is not None:
        return tuple(padded_results), folder_names
    else:
        return tuple(padded_results)

# Test Maze Data Set
if __name__=="__main__":
    data_path = ["/home/libo/program/wordmodel/libo/for_train_word_model"]
    dataset = MazeDataSet(data_path, 1280, verbose=True)
    print("The number of data is: %s" % len(dataset))
    obs, bact, lact, bactv, lactv, rewards, bevs = dataset[0]
    print(obs.shape, bact.shape, lact.shape, bactv.shape, lactv.shape, rewards.shape, bevs.shape)
