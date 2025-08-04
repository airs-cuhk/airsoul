import torch
from torch import nn
from torch.nn import functional as F
import os
import pathlib
from tqdm import tqdm
import numpy as np
from PIL import Image
from scipy import linalg

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
    elif(loss_type == 'psnr'):
        assert gt is not None, "Ground Truth Must Be Provided When Using PSNR Loss"
        return calculate_psnr(out, gt)
    elif(loss_type == 'fid'):
        assert gt is not None, "Ground Truth Must Be Provided When Using PSNR Loss"
        return calculate_fid(out, gt)
    else:
        raise ValueError('Unknown loss type {}'.format(loss_type))
    return loss_array

def calculate_fid(generated_image, ground_truth, model = None):
    if isinstance(ground_truth, np.ndarray):
        ground_truth = torch.tensor(ground_truth)
    if isinstance(generated_image, np.ndarray):
        generated_image = torch.tensor(generated_image)
    if ground_truth.dim() == 3:
        ground_truth = ground_truth.unsqueeze(0)
    if generated_image.dim() == 3:
        generated_image = generated_image
    if ground_truth.dim() == 5:
        ground_truth = ground_truth.view(ground_truth.shape[0]*ground_truth.shape[1], ground_truth.shape[2], ground_truth.shape[3], ground_truth.shape[4])
    if generated_image.dim() == 5:
        generated_image = generated_image.view(generated_image.shape[0]*generated_image.shape[1], generated_image.shape[2], generated_image.shape[3], generated_image.shape[4])
    if model is None:
        model = torch.hub.load('pytorch/vision:v0.6.0', 'inception_v3', pretrained=True)
        model.fc = torch.nn.Identity()
    mu1, sigma1 = calculate_activation_statistics(ground_truth, model, device="cuda", num_workers=1)
    mu2, sigma2 = calculate_activation_statistics(generated_image, model, device="cuda", num_workers=1)
    fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid

def get_activations(
    images, model, batch_size=50, dims=2048, device="cpu", num_workers=1
):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files       : batches of image files of shape (N, ~)
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()
    # get the last feature layer of the model
    model = model.to(device)
    if batch_size > len(images):
        print(
            (
                "Warning: batch size is bigger than the data size. "
                "Setting batch size to data size"
            )
        )
        batch_size = len(images)

    
    dataset = torch.utils.data.TensorDataset(images)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    pred_arr = None
    for batch in tqdm(dataloader):
        batch = batch[0].to(device)
        with torch.no_grad():
            pred = model(batch)
        if pred_arr is None:
            pred_arr = pred
        else:
            pred_arr = torch.cat((pred_arr, pred), dim=0)
    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

def calculate_activation_statistics(
    images, model, batch_size=50, device="cpu", num_workers=1
):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of images
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(images, model, batch_size, device, num_workers)
    # act in the shape of (N, ~)
    act = act.cpu().numpy()
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    # mu = torch.mean(act, dim=0)
    # sigma = torch.cov(act)
    return mu, sigma

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
    # print("loss weight:", loss_wht)
    if(reduce_dim is None): # if not reduced at all, then just return the 2-D loss array
        if(loss_wht is None):
            mean_loss = loss_array
            counts = torch.ones_like(loss_array)
            # print("counts:", counts)
        else:
            # print(loss_wht)
            # print(loss_wht.shape)
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
        # the count is from the loss_wht, which can be sum of that or average of certain position
        # if the count is zero, which means the loss wht is zero, then we set it to 1 will not affect the mean loss, and so on
        # we do not want to return zero count, which will cause the inf when calculating the mean loss
        # while...this implementation is not very efficient, but it is simple and easy to understand
        if counts.dim() == 0:
            if counts == 0:
                counts = 1
        # elif counts.dim() == 1: # (T,)
        #     for cnt in range(counts.shape[0]):
        #         if counts[cnt] == 0:
        #             counts[cnt] = 1
        elif counts.dim() == 2: # (B, T)
            for b in range(counts.shape[0]):
                for cnt in range(counts.shape[1]):
                    if counts[b][cnt] == 0:
                        counts[b][cnt] = 1
        return mean_loss, counts
    else:
        return mean_loss

def calculate_psnr(ground_truth, generated_image, max_val):
    mse = torch.mean((ground_truth - generated_image) ** 2)
    assert mse > 0, "mse is zero"
    max_i = max_val # torch.max(ground_truth)
    psnr = 20 * torch.log10(max_i / torch.sqrt(mse))
    return psnr


def parameters_regularization(*layers):
    norm = 0
    cnt = 0
    for layer in layers:
        for p in layer.parameters():
            if(p.requires_grad):
                norm += (p ** 2).sum()
                cnt += p.numel()
    return norm / cnt
