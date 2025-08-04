import os
import sys
import random
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset


class LMDataSet(Dataset):
    def __init__(self, directory, file_size, verbose=False):
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
            self.file_list.extend([os.path.join(d, file) for file in file_list])
        self.data_list = []
        for file in self.file_list:
            self.data_list.extend([(file, i) for i in range(self.file_size)])
        if(verbose):
            print("...finished initializing data set, number of samples: %s\n" % len(self.index_inverse_list))

    def __getitem__(self, index):
        path, sub_index = self.data_list[index]
        data = np.load(path)
        return torch.from_numpy(data[sub_index][:-1]).to(torch.int64), torch.from_numpy(data[sub_index][1:]).to(torch.int64)

    def __len__(self):
        return len(self.data_list)

# Test Maze Data Set
if __name__=="__main__":
    data_path = sys.argv[1]
    dataset = LMDataSet(data_path, 1280)
    print("The number of data is: %s" % len(dataset))
    fea, lab = dataset[0]
    print(fea.shape, lab.shape)
