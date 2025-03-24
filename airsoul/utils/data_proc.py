import numpy
import torch

def img_pro(observations):
    return observations / 255

def img_post(observations):
    return observations * 255


def rewards2go(rewards, gamma=0.98):
    """
    returns a future moving average of rewards
    """
    rolled_rewards = rewards.clone()
    r2go = rewards.clone()
    n = max(min(50, -1/numpy.log10(gamma)), 0)
    for _ in range(n):
        rolled_rewards = gamma * torch.roll(rolled_rewards, shifts=-1, dims=1)
        r2go += rolled_rewards
    return r2go

def sa_dropout(x, p=0.15, replacement=None):
    """
    drop a discrete tensor with probability p
    args:
        x: input tensor, shape (*, seq_len) or (*, seq_len, hidden)
        p: probability to drop
        if x is int, see axis=-1 as the seq, replace dropped values by -1
        if x is float, replace dropped values by 0
    returns:
        dropped out tensor
    """
    dtype=x.dtype
    device=x.device
    if(dtype in [torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64]):
        if(replacement == None):
            replacement = torch.tensor(-1, dtype=dtype).to(device)
        mask = (torch.rand(*x.shape) > p).to(device)
        x = torch.where(mask, x, replacement)
        return x
    else:
        mask = (torch.rand(*x.shape[:-1]) > p).to(device)
        mask = mask.unsqueeze(-1).expand_as(x)
        if(replacement == None):
            replacement = torch.zeros_like(x).to(device)
        else:
            for _ in range(x.dim() - 1):
                replacement = torch.unsqueeze(replacement, 0)
        x = torch.where(mask, x, replacement.to(device))
        return x

def downsample(x, downsample_length, axis=-1):
    """
    Downsample and get the mean of each segment along a given axis
    """
    if(isinstance(x, torch.Tensor)):
        shape = x.shape
        ndim = x.dim()
    else:
        shape = numpy.shape(x)
        ndim = numpy.asarray(x).ndim
    if(axis < 0):
        axis = axis + ndim
    full_len = shape[axis]
    if(downsample_length >= full_len):
        if(isinstance(x, torch.Tensor)):
            return torch.mean(x, dim=axis, keepdim=True)
        else:
            return numpy.mean(x, axis=axis, keepdims=True)
    trunc_seg = full_len // downsample_length
    trunc_len = trunc_seg * downsample_length

    new_shape = shape[:axis] + (trunc_seg, downsample_length) + shape[axis + 1:]
    # Only when left elements is large enough we add the last statistics
    need_addition = (trunc_len + downsample_length // 2 < full_len)

    if(isinstance(x, torch.Tensor)):
        ds_x = torch.mean(torch.narrow(x, axis, 0, trunc_len).view(new_shape), dim=axis + 1, keepdim=False)
        if(need_addition):
            add_x = torch.mean(torch.narrow(x, axis, trunc_len, full_len - trunc_len), dim=axis, keepdim=True)
            ds_x = torch.cat((ds_x, add_x), dim=axis)
    else:
        slc = [slice(None)] * len(shape)
        slc[axis] = slice(0, trunc_len)
        x = numpy.array(x)
        reshape_x = numpy.reshape(x[tuple(slc)], new_shape)
        ds_x = numpy.mean(reshape_x, axis=axis + 1, keepdims=False)
        if(need_addition):
            slc[axis] = slice(trunc_len, full_len)
            add_x = numpy.mean(x[tuple(slc)], axis=axis, keepdims=True)
            ds_x = numpy.concatenate((ds_x, add_x), axis=axis)
    return ds_x