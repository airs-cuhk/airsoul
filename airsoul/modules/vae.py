import torch
from torch import nn
from torch.nn import functional as F
from airsoul.utils import weighted_loss

class VAE(nn.Module):
    def __init__(
        self,
        hidden_size,
        encoder,
        decoder):
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder = encoder
        self.decoder = decoder
        self.layer_mean = nn.Linear(encoder.output_size, hidden_size)
        self.layer_var = nn.Linear(encoder.output_size, hidden_size)

        # print("VAE layer norm is ON")
        # self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, inputs):
        # input shape: [B, NT, C, W, H]
        nB, nT, nC, nW, nH = inputs.shape
        hidden = self.encoder(inputs.reshape(nB * nT, nC, nW, nH))
        z_exp = self.layer_mean(hidden)

        # z_exp = self.layer_norm(z_exp)
        
        z_log_var = self.layer_var(hidden)
        z_exp = z_exp.reshape(nB, nT, self.hidden_size)
        z_log_var = z_log_var.reshape(nB, nT, self.hidden_size)
        return z_exp, z_log_var

    def reconstruct(self,inputs, _sigma=1.0):
        nB, nT, nC, nW, nH = inputs.shape
        z_exp, z_log_var = self.forward(inputs)
        epsilon = torch.randn_like(z_log_var).to(z_log_var.device)
        z = z_exp + _sigma * torch.exp(z_log_var / 2) * epsilon
        outputs = self.decoding(z)
        return outputs, z_exp, z_log_var

    def decoding(self, z):
        nB, nT, nH = z.shape
        outputs = self.decoder(z.reshape(nB * nT, nH))
        outputs = outputs.reshape(nB, nT, *outputs.shape[1:])
        return outputs

    def loss(self, inputs, _sigma=0.0, seq_len=None):
        # import pdb;pdb.set_trace()
        outputs, z_exp, z_log_var = self.reconstruct(inputs, _sigma = _sigma)
        import numpy as np
        # if np.isnan(np.sum(outputs.detach().cpu().numpy())):

        #     print(np.where(np.isnan(outputs.detach().cpu().numpy())))
        #     np.save("./vae_test2/vae_nan_real.npy", inputs.detach().cpu().numpy())
        #     np.save("./vae_test2/vae_nan.npy", outputs.detach().cpu().numpy())
        
        # B T 
        # print("--------------------------")
        # print(z_exp.shape)
        # kl_tmp = -0.5 * torch.sum(1 + z_log_var - torch.square(z_exp) - torch.exp(z_log_var), axis=1)
        # kl_tmp1 = -0.5 * torch.sum(1 + z_log_var - torch.square(z_exp) - torch.exp(z_log_var), axis=-1)
        # print(kl_tmp.shape)

        kl_loss = torch.mean(-0.5 * torch.sum(1 + z_log_var - torch.square(z_exp) - torch.exp(z_log_var), axis=1))

        # print(kl_tmp)
        # print(kl_tmp1)
        # print(torch.var(kl_tmp))
        # print(kl_loss)
        # print(torch.mean(kl_tmp1))
        # print("---------------------------")

        reconstruction_loss, cnt = weighted_loss(outputs, loss_type="mse", gt=inputs, reduce_dim=1, need_cnt=True)

        if(seq_len is None):
            normal_factor = 1.0
        else:
            normal_factor = 1.0 / seq_len

        reconstruction_loss *= normal_factor
        kl_loss *= normal_factor
        cnt *= normal_factor

        return {"Reconstruction-Error": reconstruction_loss,
                "KL-Divergence": kl_loss,
                "count": cnt,}
                # "inf": inf_tensor, 
                # "zero": zero_tensor}
