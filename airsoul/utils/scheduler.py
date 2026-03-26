import torch
from torch import nn
from torch.nn import functional as F
import math

class LinearScheduler(object):
    def __init__(self, unit_steps, unit_values):
        self._unit_steps = unit_steps
        self._unit_values = unit_values
        self._step = 0

    def reset(self, step=0):
        self._step = step

    def step(self):
        self._step += 1

    def __call__(self):
        units = self._step / self._unit_steps
        unit_i = int(units)
        unit_d = units - unit_i
        if(unit_i >= len(self._unit_values) - 1):
            return self._unit_values[-1]
        else:
            return (1 - unit_d) * self._unit_values[unit_i] + unit_d * self._unit_values[unit_i + 1]



class CosineScheduler(object):
    def __init__(self, total_steps, warmup=0, lr_max=1.0, lr_min=0.0):
        self.warmup = warmup
        self.total_steps = total_steps
        self.lr_max = lr_max
        self.lr_min = lr_min
        self._step = 0

    def reset(self, step=0):
        self._step = step

    def step(self):
        self._step += 1
    def __call__(self):
        step = self._step
        if step < self.warmup:
            lr = self.lr_max * step / self.warmup
        else:
            progress = (step - self.warmup) / (self.total_steps - self.warmup)
            # lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * progress))
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + math.cos(math.pi * progress))
        return lr

def cosine_function_scheduler(it, total_steps, warmup=0, lr_max=1.0, lr_min=0.0):
    step = it
    if step < warmup:
        lr = lr_max * step / warmup
    else:
        progress = (step - warmup) / (total_steps - warmup)
        # lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * progress))
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))
    return lr


def noam_scheduler(it, warmup_steps, low=0.0):
    vit = max(it, 1)
    low = max(min(low, 1.0), 0.0)
    lr_warm = (1.0 - low) * vit / warmup_steps + low # warm up steps
    lr_decay = (((1.0 - low) * (vit / warmup_steps)) ** (-0.5)) + low
    return max(min(lr_warm, lr_decay), low)
