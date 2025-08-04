import os
import sys
import random
import torch
import itertools
import queue
import numpy as np
import multiprocessing
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate as torch_collate
from airsoul.utils.tools import Logger, log_progress, log_debug, log_warn, log_fatal

class BaseDataLoader(DataLoader):
    def __init__(self, dataset, rank=0, world_size=1, batch_size=4, collate_fn=torch_collate):
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.rank = rank
        self.world_size = world_size
        self.iter = 12345
        self.index = 0
        self.local_index = 0
        self.data_volume = len(self.dataset)
        self.length = (self.data_volume - 1) // (self.batch_size * self.world_size) + 1

    def __iter__(self):
        self.index = self.rank
        self.local_index = 0
        # Set the random shuffler for the data loader
        self.iter += 1
        torch.manual_seed(self.iter)
        self.index_shuffler = torch.randperm(self.data_volume).tolist()
        return self

    def __next__(self):
        if self.local_index >= self.length or self.index >= self.data_volume:
            # stop iteration once index is out of bounds
            raise StopIteration
        data = []
        sub_iter = 0
        while sub_iter < self.batch_size:
            # In case the data is invalid, fetch further data
            try:
                sub_data = self.get()
            except Exception as e:
                log_warn("Error fetching data from dataset: ", e)
                continue
            if(sub_data is not None):
                data.append(sub_data)
                sub_iter += 1
        self.local_index += 1
        return self.collate_fn(data)

    def get(self):
        item = self.dataset[self.index_shuffler[self.index]]
        self.index += self.world_size
        return item

    def __len__(self):
        return self.length

def worker_fn(dataset, length, index_queue, output_queue):
    while True:
        try:
            index = index_queue.get(timeout=0)
        except queue.Empty:
            continue
        if index is None:
            break

        if index > length - 1:
            # Allowing fetch index to exceed max length to accommodate certain mistake
            real_idx = index % length
        else:
            real_idx = index
        try:
            output_queue.put((real_idx, dataset[real_idx]))
        except Exception as e:
            log_warn(f"DataLoader:unexpected error when getting {real_idx}:{e}")
            output_queue.put((real_idx, None))

class PrefetchDataLoader(BaseDataLoader):
    def __init__(
        self,
        dataset,
        rank=0,
        world_size=1,
        batch_size=4,
        num_workers=2,
        prefetch_batches=2,
        collate_fn=torch_collate,
    ):
        super().__init__(dataset, 
            batch_size=batch_size, 
            rank=rank, 
            world_size=world_size, 
            collate_fn=collate_fn)

        self.num_workers = num_workers
        self.prefetch_batches = prefetch_batches
        self.output_queue = multiprocessing.Queue()
        self.index_queues = []
        self.workers = []
        self.worker_cycle = itertools.cycle(range(num_workers))
        self.cache = {}

        for _ in range(num_workers):
            index_queue = multiprocessing.Queue()
            worker = multiprocessing.Process(
                target=worker_fn, args=(self.dataset, self.data_volume, index_queue, self.output_queue)
            )
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
            self.index_queues.append(index_queue)

    def prefetch(self):
        """
        Index shuffler introduced at prefetch stage
        """
        while (self.prefetch_index < self.index + self.prefetch_batches * self.batch_size * self.world_size):
            real_prefetch_index = self.index_shuffler[self.prefetch_index % self.data_volume]
            self.index_queues[next(self.worker_cycle)].put(real_prefetch_index)
            self.prefetch_index += self.world_size

    def get(self):
        self.prefetch()
        sys.stdout.flush()
        real_index = self.index_shuffler[self.index % self.data_volume]
        if real_index in self.cache:
            item = self.cache.pop(real_index)
        else:
            try:
                (fetch_index, data) = self.output_queue.get(timeout=60)
                if real_index == fetch_index:
                    item = data
                else:
                    self.cache[fetch_index] = data
                    return self.get()
            except queue.Empty:
                raise StopIteration("Data fetch timeout from the output queue.")
        sys.stdout.flush()

        self.index += self.world_size
        return item

    def __iter__(self):
        self.local_index = 0
        self.index = self.rank
        self.prefetch_index = self.rank

        # Set the random shuffler for the data loader
        self.iter += 1
        torch.manual_seed(self.iter)
        self.index_shuffler = torch.randperm(self.data_volume).tolist()

        self.cache = {}
        self.prefetch()
        return self

    def __del__(self):
        try:
            for i in range(len(self.workers)):
                self.index_queues[i].put(None)
            for w in self.workers:
                w.join()
            for q in self.index_queues:
                q.close()
                q.cancel_join_thread()
            self.output_queue.close()
            self.output_queue.cancel_join_thread()
        except Exception as e:
            for w in self.workers:
                if w.is_alive():
                    w.terminate()
            raise e
    
def segment_iterator(full_len, seg_len, device, *args):
    """
    Input shape: [Batch, Length, *]
    Output: list of [Batch, seg_len, *]
    args: either torch.Tensor or tuple(torch.Tensor, ext)
    """
    data_ext = []
    data = []

    # Make sure full_len is shorter than args
    for i,arg in enumerate(args):
        if(isinstance(arg, tuple) or isinstance(arg, list)):
            sdata, ext = arg
        elif(isinstance(arg, torch.Tensor) or arg is None):
            sdata = arg
            ext = 0
        else:
            log_fatal(f"Unrecognized data type {type(arg)}")
        ext = max(0, ext)
        data_ext.append(ext)
        data.append(sdata)
        if(sdata is not None):
            arg_len = sdata.shape[1]
            # Make sure we have enough space
            full_len = min(full_len, arg_len - ext)

    seg_num = (full_len - 1) // seg_len + 1
    for seg_id in range(seg_num):
        res = [seg_id]
        b = seg_id * seg_len
        e = min(b + seg_len, full_len)
        for sdata, ext in zip(data, data_ext):
            if(sdata is not None):
                res.append(sdata[:, b:e+ext].to(device))
            else:
                res.append(None)
        yield tuple(res)