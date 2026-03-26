import os
import sys
import random
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import tqdm

class LMDataSet(Dataset):
    def __init__(self, directory, max_length, world_size=None, max_data=None, verbose=False):
        if(verbose):
            print("\nInitializing data set from file: %s..." % directory)
        self.file_list = []
        self.max_length = max_length
        self.sequential = False
        self.start_index = 0
        directories = []
        if(isinstance(directory, list)):
            directories.extend(directory)
        else:
            directories.append(directory)
        for d in directories:
            for root, _, files in os.walk(d):
                self.file_list.extend([os.path.join(root, file) for file in files])
        self.file_list = sorted(self.file_list)        
        self.data_list = []
        
        for file in self.file_list:
            data = np.load(file)
            assert data.ndim == 3 and (data.shape[1] == 2 or data.shape[1] == 3), \
                    f"Expect the data shape of meta_lm being (Bsz, 2, Length), get {data.shape}"
            file_size = data.shape[0]
            self.data_list.extend([(file, i) for i in range(file_size)])
        
        assert len(self.data_list) > 0, "No data in the data set"
        
        if max_data is not None and len(self.data_list) > max_data:
            self.data_list = self.data_list[:max_data]
            
        ws = 32
        if len(self.data_list) % ws != 0:
            print(f"[Warning] The number of data is not divisible by 8, the number of data is {len(self.data_list)}")
            self.data_list = self.data_list[:len(self.data_list) - len(self.data_list) % ws]
            
        if(verbose):
            print("...finished initializing data set, number of samples: %s\n" % len(self.index_inverse_list))

    def __getitem__(self, index):
        path, sub_index = self.data_list[index]
        data = np.load(path)
        if data.shape[1] == 2:
            return torch.from_numpy(data[sub_index][0][:self.max_length]).to(torch.int64), torch.from_numpy(data[sub_index][1][:self.max_length]).to(torch.int64), torch.ones(data[sub_index][1][:self.max_length].shape).float() 
        elif data.shape[1] == 3:
            return torch.from_numpy(data[sub_index][0][:self.max_length]).to(torch.int64), torch.from_numpy(data[sub_index][1][:self.max_length]).to(torch.int64), torch.from_numpy(data[sub_index][2][:self.max_length]).float() 
        else: 
            raise ValueError(f"Expect the data shape of meta_lm being (Bsz, 2 or 3, Length), get {data.shape}")

    def __len__(self):
        return len(self.data_list)

# Test Maze Data Set
if __name__=="__main__":
    data_path = sys.argv[1]
    dataset = LMDataSet(data_path, 1280)
    print("The number of data is: %s" % len(dataset))
    fea, lab = dataset[0]
    print(fea.shape, lab.shape)
