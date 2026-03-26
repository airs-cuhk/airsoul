import os
import sys
import random
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

class SmallBatchDataSetBase(Dataset):
    def __init__(self, directory, time_step, file_size=-1, verbose=False):
        # For file_size!=None, each file contains multiple samples
        # For file_size=None, each file contains one sample
        if(verbose):
            print("\nInitializing data set from file: %s..." % directory)
        self.file_list = []
        self.file_size = file_size

        directories = []
        if(isinstance(directory, list)):
            directories.extend(directory)
        else:
            directories.append(directory)
        for d in directories:
            file_list = os.listdir(d)
            if(self.file_size is None):
                self.file_list.extend([os.path.join(d, file) for file in file_list])
            else:
                for file in file_list:
                    if(self.file_size < 0):
                        file_size = np.load(os.path.join(d, file_list[0]) + '/observations.npy').shape[0]
                    else:
                        file_size = self.file_size
                    self.file_list.extend([(os.path.join(d, file), idx) for idx in range(file_size)])            
            
        self.time_step = time_step

        if(verbose):
            print("...finished initializing data set, number of samples: %s\n" % len(self.file_list))

    def __len__(self):
        return len(self.file_list)

    def _load_and_process_data(self, path):
        if(isinstance(path, tuple)):
            path, idx = path
        else:
            idx = None
        try:
            observations = np.load(path + '/observations.npy')
            actions_behavior = np.load(path + '/actions_behavior.npy')
            actions_label = np.load(path + '/actions_label.npy')
            rewards = np.load(path + '/rewards.npy')
            
            if(idx is not None):
                observations = observations[idx]
                actions_behavior = actions_behavior[idx]
                actions_label = actions_label[idx]
                rewards = rewards[idx]

            max_t = min(actions_label.shape[0], 
                        rewards.shape[0], 
                        actions_behavior.shape[0],
                        observations.shape[0] - 1)

            # Shape Check
            if(self.time_step > max_t):
                print(f'[Warning] Load samples from {path} that is shorter ({max_t}) than specified time step ({self.time_step})')
                n_b = 0
                n_e = max_t
            else:
                n_b = 0
                n_e = self.time_step

            return (observations[n_b:n_e + 1], 
                    actions_behavior[n_b:n_e], 
                    rewards[n_b:n_e],
                    actions_label[n_b:n_e])
        except Exception as e:
            print(f"Unexpected reading error founded when loading {path}: {e}")
            return (None,) * 4

    def __getitem__(self, index):
        path = self.file_list[index]

        data = self._load_and_process_data(path)
        
        if any(arr is None for arr in data):
            return None

        obs_arr = torch.from_numpy(data[0].astype("float32")).float() 
        bact_arr = torch.from_numpy(data[1].astype("int32")).long()
        rwd_arr = torch.from_numpy(data[2]).float()
        lact_arr = torch.from_numpy(data[3].astype("int32")).long()

        # Orders: O-P-T-A-R and Action Label
        return obs_arr, bact_arr, rwd_arr, lact_arr

# Test Maze Data Set
if __name__=="__main__":
    data_path = sys.argv[1]
    dataset = SmallBatchDataSetBase(data_path, 200, file_size=-1, verbose=True)
    print("The number of data is: %s" % len(dataset))
    obs, bact, rewards, lact = dataset[0]
    if obs is not None:
        print(obs.shape, bact.shape, lact.shape, rewards.shape)
    else:
        print("Failed to load the first sample.")