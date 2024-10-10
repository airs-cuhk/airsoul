import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from .res_nets import ImageDecoder, ImageEncoder, MLPEncoder, ResidualMLPDecoder
from .proxy_base import ProxyBase


class EncodeBlock(ProxyBase):
    """
    Take Observations and actions, output d_models
    """
    def __init__(self, config):
        super().__init__()

        if(config.model_type == "ResNet"):
            self.encoder = ImageEncoder(
                config.img_size,
                3,
                config.hidden_size,
                config.n_res_block
            )
        elif(config.model_type == "MLP"):
            self.encoder = MLPEncoder(
                config.input_type,
                config.hidden_size,
                config.dropout,
            )
        else:
            raise Exception("No such causal model: %s" % model_type)
        
        self.input_size = config.input_size
        self.output_size = config.hidden_size
    
class DecodeBlock(ProxyBase):
    """
    Take Observations and actions, output d_models
    """
    def __init__(self, config):
        super().__init__()

        if(config.model_type == "ResNet"):
            self.encoder = ImageDecoder(
                config.img_size,
                config.input_size,
                config.hidden_size,
                3,
                config.n_res_block
            )
        elif(config.model_type == "MLP"):
            self.encoder = ResidualMLPDecoder(
                config.input_size,
                config.hidden_size,
                config.output_type,
                dropout = config.dropout,
                layer_norm = config.layer_norm,
                residual_connect = config.residual_connect
            )
        else:
            raise Exception("No such causal model: %s" % model_type)

        self.input_size = config.input_size
        self.output_size = config.hidden_size

if __name__=='__main__':
    DT = EncodeBlock(config)
    DT = DecodeBlock(config)