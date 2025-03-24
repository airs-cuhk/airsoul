import torch
from torch import nn
from torch.nn import functional as F
from airsoul.utils import Logger, log_progress, log_debug, log_warn, log_fatal


class MLPEncoder(nn.Module):
    def __init__(self, config, reserved_ID=False):
        super().__init__()

        input_size = config.input_size
        if(config.has_attr('hidden_size')):
            hidden_size = config.hidden_size # Can be int or a list of ints
        else:
            hidden_size = None
        dropout = config.dropout
        input_type = config.input_type.lower()

        self.input_size = input_size

        if(input_type.startswith("discrete")):
            self.is_continuous = False
            assert isinstance(hidden_size, int)
            if(reserved_ID): # Reserve the last ID for special token
                self.encoder_layer = nn.Embedding(input_size + 1, hidden_size)
            else:
                self.encoder_layer = nn.Embedding(input_size, hidden_size)
            self.output_size = hidden_size
        elif(input_type.startswith("continuous")):
            self.is_continuous = True
            if(hidden_size is None):
                self.output_size = input_size
                self.encoder_layer = nn.Identity()
            elif(isinstance(hidden_size, tuple) or isinstance(hidden_size, list)):
                layers = []
                ph = input_size
                for h in hidden_size[:-1]:
                    layers.append(nn.Linear(ph, h))
                    layers.append(nn.GELU())
                    layers.append(nn.Dropout(dropout))
                    ph = h
                layers.append(nn.Linear(ph, hidden_size[-1]))
                self.encoder_layer = nn.Sequential(*layers)
                self.output_size = hidden_size[-1]
            else:
                self.encoder_layer = nn.Linear(input_size, hidden_size)
                self.output_size = hidden_size
        else:
            log_fatal(f"action input type must be ContinuousXX or DiscreteXX, unrecognized `{output_type}`")

        # Frozen the parameters
        if(config.has_attr('is_fronzen')):
            if(config.is_frozen):
                for param in self.parameters():
                    param.requires_grad_(False)

    def forward(self, input):
        return self.encoder_layer(input)

class ResidualMLPDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        output_type = config.output_type.lower()
        input_size = config.input_size
        if(config.has_attr('hidden_size')):
            hidden_size = config.hidden_size
        else:
            hidden_size = None
        dropout = config.dropout
        layer_norm = config.layer_norm
        residual_connect = config.residual_connect

        self.input_size = input_size
        
        if(layer_norm):
            self.layer_norm = nn.LayerNorm(input_size, eps=1.0e-5)
        else:
            self.layer_norm = nn.Identity()

        def get_layers(io_list, dropout):
            if(len(io_list) < 2):
                return nn.Identity()
            elif(len(io_list) < 3):
                return nn.Linear(io_list[0], io_list[1])
            else:
                layers = []
                for i in range(len(io_list) - 2):
                    layers.append(nn.Linear(io_list[i], io_list[i+1]))
                    layers.append(nn.GELU())
                    layers.append(nn.Dropout(dropout))
                layers.append(nn.Linear(io_list[-2], io_list[-1]))
                return nn.Sequential(*layers)

        self.residual_connect = residual_connect
        if(hidden_size is None):
            self.decoder_pre = nn.Identity()
            self.residual_connect = False
            self.output_size = input_size
        elif(residual_connect):
            if(isinstance(hidden_size, tuple) or isinstance(hidden_size, list)):
                self.decoder_pre = get_layers([input_size] + list(hidden_size[:-1]) + [input_size], dropout)
                self.decoder_post = get_layers([input_size, hidden_size[-1]], dropout)
                self.output_size = hidden_size[-1]
            else:
                log_fatal("if use residual connection, the hidden size must have at least two layers")
        else:
            if(isinstance(hidden_size, tuple) or isinstance(hidden_size, list)):
                self.decoder_pre = get_layers([input_size] + list(hidden_size), dropout)
                self.output_size = hidden_size[-1]
            else:
                self.decoder_pre = get_layers([input_size, hidden_size], dropout)
                self.output_size = hidden_size
            self.decoder_post = None

        if(output_type.startswith("discrete")):
            self.is_continuous = False
            self.decoder_output = nn.Softmax(dim=-1)
        elif(output_type.startswith("continuous")):
            self.is_continuous = True
            self.decoder_output = nn.Identity()
        else:
            raise log_fatal(f"action output type must be Continuous or Discrete, unrecognized `{output_type}`")

        # Frozen the parameters
        if(config.has_attr('is_fronzen')):
            if(config.is_frozen):
                for param in self.parameters():
                    param.requires_grad_(False)

    def forward(self, input, T=1.0):
        src = self.layer_norm(input)
        out = self.decoder_pre(src)
        if(self.residual_connect):
            out = self.decoder_post(out + src)
        if(not self.is_continuous):
            return self.decoder_output(out / T)
        else:
            return self.decoder_output(out)
