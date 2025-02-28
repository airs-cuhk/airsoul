import torch
from torch import nn
from torch.nn import functional as F

def ent_loss(act_out):
    """
    this returns negative entropy p * log(p) instead of - p * log(p)
    which is directly ready to be minimized, do not add negative sign
    """
    return torch.sum(torch.log(act_out + 1.0e-10) * act_out, dim=-1)

def focal_loss(out, gt, gamma=0):
    gt_logits = F.one_hot(gt, out.shape[-1])
    preds = torch.log(out + 1.0e-10) * ((1.0 - out) ** gamma)
    return -torch.sum(preds * gt_logits, dim=-1)

def metrics(out, gt=None, loss_type='mse', **kwargs):
    if(loss_type == 'mse'):
        assert gt is not None, "Ground Truth Must Be Provided When Using MSE Loss"
        loss_array = torch.mean((out - gt) ** 2, dim=[i for i in range(2, out.ndim)])
    elif(loss_type == 'ce'):
        assert gt is not None, "Ground Truth Must Be Provided When Using Cross Entropy Loss"
        if('gamma' not in kwargs):
            loss_array = focal_loss(out, gt)
        else:
            loss_array = focal_loss(out, gt, kwargs['gamma'])
    elif(loss_type == 'ent'):
        return ent_loss(out)
    else:
        raise ValueError('Unknown loss type {}'.format(loss_type))
    return loss_array

def weighted_loss(out, loss_wht=None, reduce_dim=1, need_cnt=False, **kwargs):
    """
    input and ground truth shape: (B, T, *)
    loss_wht shape: (B, T) (or (T,))), loss weight for each sample
    loss_wht should be provided that summation over the T dimension is 1 (or close to 1)

    loss_type: 'mse', 'ce'
    reduce_dim: None - Not Reduced At All
                0 - Only reduce the batch dimension
                1 - Only reduce both the batch and the temporal dimension
    return:
        mse_loss: weighted mean of losses
        sample_cnt: number of samples corresponding to each position in mse_loss
                    mainly used for weighting in statistics
    """
    
    loss_array = metrics(out, **kwargs)

    if(reduce_dim is None): # if not reduced at all, then just return the 2-D loss array
        if(loss_wht is None):
            mean_loss = loss_array
            counts = torch.ones_like(loss_array)
        else:
            if(loss_wht.ndim == 1):
                assert loss_wht.shape[0] == loss_array.shape[1]
                loss_wht = loss_wht.unsqueeze(0).repeat(loss_array.shape[0])
                mean_loss = loss_array * loss_wht
                counts = loss_wht.unsqueeze(0).repeat(loss_array.shape[0])
            else:
                assert loss_wht.ndim == 2 and loss_wht.shape == loss_array.shape
                mean_loss = loss_array * loss_wht
                counts = loss_wht
    else:
        assert reduce_dim in [0, 1], "reduce_dim should be either None, 0 or 1."
        if(loss_wht is None): # if loss_wht is None, then all samples are AVERAGED
            # Sum over dimension 0
            counts = torch.full((loss_array.shape[1],), 1.0, 
                    dtype=loss_array.dtype, device=loss_array.device)
            mean_loss = torch.mean(loss_array, dim=[0])

            # Sum over dimension 1 if needed
            if(reduce_dim == 1):
                counts = torch.sum(counts)
                mean_loss = torch.sum(mean_loss)
        else: # if loss_wht is provided, then samples are summed with the weights
            # The returned counts is the sum of the weights
            if(loss_wht.ndim == 1):
                loss_wht = loss_wht.unsqueeze(0)
            else:
                assert loss_wht.ndim == 2
                assert loss_wht.shape[0] == loss_array.shape[0]
            assert loss_wht.shape[1] == loss_array.shape[1]
            
            # Mean over dimension 0
            loss_array = loss_array * loss_wht
            counts = torch.mean(loss_wht, dim=[0])
            mean_loss = torch.mean(loss_array, dim=[0])

            # Sum over dimension 1 if needed
            if(reduce_dim == 1):
                counts = torch.sum(counts)
                mean_loss = torch.sum(mean_loss)
    
    if(need_cnt):
        return mean_loss, counts
    else:
        return mean_loss
    
def parameters_regularization(*layers):
    norm = 0
    cnt = 0
    for layer in layers:
        for p in layer.parameters():
            if(p.requires_grad):
                norm += (p ** 2).sum()
                cnt += p.numel()
    return norm / cnt
