import copy
import torch
import torch.nn as nn
from torch.nn import functional as F


class ProxyBase(nn.Module):
    """
    Take Observations and actions, output d_models
    """
    def __init__(self, config):
        raise NotImplementedError("Proxy Module must implement self.layers in __init__")

    def forward(self, *args, **kwargs):
        return self.layers(*args, **kwargs)