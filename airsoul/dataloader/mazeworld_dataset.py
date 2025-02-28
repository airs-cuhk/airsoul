import os
import sys
import torch
import numpy as np
from numpy import random
from torch.utils.data import DataLoader, Dataset


class MazeDataSet(Dataset):
    def __init__(self, directory, time_step, verbose=False):
        if(verbose):
            print("\nInitializing data set from file: %s..." % directory)
        self.file_list = []
        directories = []
        if(isinstance(directory, list)):
            directories.extend(directory)
        else:
            directories.append(directory)
        for d in directories:
            file_list = os.listdir(d)
            self.file_list.extend([os.path.join(d, file) for file in file_list])
            
        self.time_step = time_step

        if(verbose):
            print("...finished initializing data set, number of samples: %s\n" % len(self.file_list))

    def __getitem__(self, index):
        path = self.file_list[index]

        try:
            observations = np.load(path + '/observations.npy')
            actions_behavior_id = np.load(path + '/actions_behavior_id.npy')
            actions_label_id = np.load(path + '/actions_label_id.npy')
            actions_behavior_val = np.load(path + '/actions_behavior_val.npy')
            actions_label_val = np.load(path + '/actions_label_val.npy')
            rewards = np.load(path + '/rewards.npy')
            bevs = np.load(path + '/BEVs.npy')
            max_t = actions_behavior_id.shape[0]

            # Shape Check
            assert max_t == rewards.shape[0]
            assert max_t == actions_behavior_val.shape[0]
            assert max_t == actions_label_id.shape[0]
            assert max_t == actions_label_val.shape[0]
            assert max_t == bevs.shape[0]
            assert max_t + 1 == observations.shape[0]

            if(self.time_step > max_t):
                print(f'[Warning] Load samples from {path} that is shorter ({max_t}) than specified time step ({self.time_step})')
                n_b = 0
                n_e = max_t
            else:
                n_b = 0
                n_e = self.time_step
            obs_arr = torch.from_numpy(observations[n_b:(n_e + 1)]).float() 
            bact_id_arr = torch.from_numpy(actions_behavior_id[n_b:n_e]).long() 
            lact_id_arr = torch.from_numpy(actions_label_id[n_b:n_e]).long() 
            bact_val_arr = torch.from_numpy(actions_behavior_val[n_b:n_e]).float() 
            lact_val_arr = torch.from_numpy(actions_label_val[n_b:n_e]).float() 
            reward_arr = torch.from_numpy(rewards[n_b:n_e]).float()
            bev_arr = torch.from_numpy(bevs[n_b:n_e]).float()
            return obs_arr, bact_id_arr, lact_id_arr, bact_val_arr, lact_val_arr, reward_arr, bev_arr
        except Exception as e:
            print(f"Unexpected reading error founded when loading {path}: {e}")
            return None

    def __len__(self):
        return len(self.file_list)


# Test Maze Data Set
if __name__=="__main__":
    data_path = sys.argv[1]
    dataset = MazeDataSet(data_path, 1280, verbose=True)
    print("The number of data is: %s" % len(dataset))
    obs, bact, lact, bactv, lactv, rewards, bevs = dataset[0]
    print(obs.shape, bact.shape, lact.shape, bactv.shape, lactv.shape, rewards.shape, bevs.shape)
