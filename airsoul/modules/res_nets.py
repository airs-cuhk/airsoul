import torch
from torch import nn
from torch.nn import functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channel, hidden_size):
        super().__init__()

        self.conv = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv2d(in_channel, hidden_size, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_size, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out

class ImageEncoder(nn.Module):
    """
    Change [B*NT, C_in, 128, 128] to [B*NT, C_out]
    """
    def __init__(self, config):
        super().__init__()

        img_size = config.img_size
        in_channel = 3 # Consider only RGB channels currently
        out_channel = config.hidden_size
        n_res_block = config.n_res_block
        self.output_size = out_channel

        channel_b1 = out_channel // 32
        channel_b2 = out_channel // 16
        channel_b3 = out_channel // 8
        cur_size = img_size // 8
        fin_channel = cur_size * cur_size * channel_b3

        blocks = [
            nn.Conv2d(in_channel, channel_b1, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel_b1, channel_b2, 4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel_b2, channel_b2, 3, padding=1),
        ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel_b2, out_channel))

        blocks.extend([
            nn.Conv2d(channel_b2, channel_b2, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel_b2, channel_b3, 4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel_b3, channel_b3, 3, padding=1),
        ])

        for i in range(n_res_block):
            blocks.append(ResBlock(channel_b3, out_channel))

        blocks.extend([
            nn.Conv2d(channel_b3, channel_b3, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel_b3, channel_b3, 4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel_b3, channel_b3, 3, padding=1),
        ])

        blocks.extend([
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(fin_channel, out_channel),
            nn.LeakyReLU(),
        ])

        self.blocks = nn.Sequential(*blocks)
        self.hidden_size = out_channel

    def forward(self, input):
        return self.blocks(input)

class ImageDecoder(nn.Module):
    """
    Change [B*NT, C_in] to [B*NT, C_out, 128, 128]
    """
    def __init__(self, config):
        super().__init__()

        img_size = config.img_size
        in_channel = config.input_size
        hidden_size = config.hidden_size
        out_channel = 3 # Consider only RGB channels currently
        n_res_block = config.n_res_block
        self.output_size = out_channel

        channel_b1 = hidden_size // 4
        self.ini_channel = hidden_size // 32
        self.ini_size = img_size // 8
        ini_mapping = self.ini_size * self.ini_size * self.ini_channel

        self.input_mapping = nn.Sequential(nn.Linear(in_channel, hidden_size), nn.LeakyReLU(), nn.Linear(hidden_size, ini_mapping), nn.LeakyReLU())

        blocks = [nn.Conv2d(self.ini_channel, channel_b1, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel_b1, in_channel))

        blocks.append(nn.LeakyReLU())

        blocks.extend(
            [
                nn.ConvTranspose2d(channel_b1, channel_b1 // 2, 4, stride=2, padding=1),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(channel_b1 // 2, channel_b1 // 4, 4, stride=2, padding=1),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(channel_b1 // 4, channel_b1 // 8, 4, stride=2, padding=1),
                nn.LeakyReLU(),
                nn.Conv2d(channel_b1 // 8, out_channel, 3, padding=1),
                nn.Sigmoid(),
            ]
        )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, inputs):
        img = self.input_mapping(inputs)
        img = img.view(-1, self.ini_channel, self.ini_size, self.ini_size)
        return self.blocks(img)
