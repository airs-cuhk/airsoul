import torch
from torch import nn
from torch.nn import functional as F

class LinearScheduler(object):
    def __init__(self, unit_steps, unit_values):
        self._unit_steps = unit_steps
        self._unit_values = unit_values
        self._step = 0

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

def noam_scheduler(it, warmup_steps, low=0.0):
    vit = max(it, 1)
    low = max(min(low, 1.0), 0.0)
    lr_warm = (1.0 - low) * vit / warmup_steps + low # warm up steps
    lr_decay = (((1.0 - low) * (vit / warmup_steps)) ** (-0.5)) + low
    return max(min(lr_warm, lr_decay), low)
