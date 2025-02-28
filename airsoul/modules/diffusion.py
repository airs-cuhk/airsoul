import torch
from torch import nn
from torch.nn import functional as F
from airsoul.utils import weighted_loss

class DiffusionLayers(nn.Module):
    def __init__(self, T, hidden_size, condition_size, inner_hidden_size, beta=(0.05, 0.20)):
        super().__init__()
        self.betas = torch.linspace(beta[0], beta[1], T)
        self.betas = torch.cat([torch.tensor([0.0]), self.betas], dim=0)
        self.alphas = 1 - self.betas
        self._alphas = torch.cumprod(self.alphas, axis=0, dtype=torch.float)

        self.condition_size = condition_size
        self.hidden_size = hidden_size
        self.input_size = 2 * hidden_size + condition_size
        self.pre_diffusion_norm = nn.LayerNorm(self.hidden_size, eps=1.0e-5)
        self.diffusion_layers_1 = nn.Sequential(
            nn.Linear(self.input_size, inner_hidden_size), 
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(inner_hidden_size, hidden_size),
            nn.GELU()) 
        self.diffusion_layers_2 = nn.Linear(hidden_size, hidden_size)
        self.T = T
        self.t_embedding = nn.Embedding(T, hidden_size)

    def step_forward(self, xt, t, cond):
        """
        xt: [B, NT, H], float
        t: [B, NT], int
        cond: [B, NT, HC], float
        """
        assert cond.shape[:2] == xt.shape[:2] and cond.shape[2] == self.condition_size
        t_emb = self.pre_diffusion_norm(self.t_embedding(t - 1))
        outputs = torch.cat([xt, t_emb, cond], dim=-1)
        outputs = self.diffusion_layers_1(outputs) + xt
        outputs = self.diffusion_layers_2(outputs)

        return outputs
    
    def loss_DDPM(self, x0, cond, mask=None, reduce_dim=1, t=None):
        if(t is None):
            _t = torch.randint(low=1, high=self.T + 1, size=x0.shape[:2], dtype=torch.int64, device=x0.device)
        else:
            _t = torch.full(cond.shape[:2], t, dtype=torch.int64, device=cond.device)
        x_t, eps, _ = self.diffusion_forward(x0, _t)
        eps_t = self.step_forward(x_t, _t, cond)
        loss = weighted_loss(eps_t, gt=eps, loss_type="mse", loss_wht=mask, reduce_dim=reduce_dim)
        return loss

    def forward(self, x0, cond, mask=None, reduce_dim=1, t=None):
        """
        Allows to back propagate through the whole process
        """

        if(t is None):
            _t = torch.randint(low=1, high=self.T + 1, size=x0.shape[:2], dtype=torch.int64, device=x0.device)
        else:
            _t = torch.full(cond.shape[:2], t, dtype=torch.int64, device=cond.device)
        x_t, eps, a_t = self.diffusion_forward(x0, _t)
        eps_t = self.step_forward(x_t, _t, cond)
        a_0 = torch.full(a_t.shape, 0, dtype=torch.int64, device=cond.device)
        print(x_t.shape, eps.shape, eps_t.shape, a_t.shape, a_0.shape)

        return torch.sqrt(a_0 / a_t) * x_t + (torch.sqrt((1 - a_0) / a_0) - torch.sqrt((1 - a_t) / a_t)) * eps_t

    def diffusion_forward(self, x0, t):
        eps = torch.randn_like(x0).to(x0.device)
        a_t = torch.take(self._alphas.to(x0.device), t).unsqueeze(-1)
        x_t = torch.sqrt(a_t) * x0 + torch.sqrt(1 - a_t) * eps
        return x_t, eps, a_t

    def inference(self, cond):
        assert cond.shape[2] == self.condition_size
        z_list = []
        steps = [2 * self.T // 3, self.T // 3, 1]
        with torch.no_grad():
            x_T = torch.randn(*cond.shape[:2], self.hidden_size)
            x_t = x_T.to(cond.device)
            z_list.append(x_t.detach())
            for t in range(self.T, 0, -1):
                _t = torch.full(cond.shape[:2], t, dtype=torch.int64, device=cond.device)
                a_t = self._alphas[t]
                a_t_ = self._alphas[t-1]
                b_t = self.betas[t]

                # DDPM
                #eps = torch.randn_like(x_t)
                #if(t == 1):
                #    eps = eps * 0
                #x_t = 1.0 / torch.sqrt(1 - b_t) * (x_t - (b_t / torch.sqrt(1 - a_t)) * self.step_forward(x_t, _t, cond)) + torch.sqrt(b_t) * eps

                # DDIM
                eps_z = torch.randn_like(x_t)
                eps_t = self.step_forward(x_t, _t, cond)
                sigma_t = 0.01 * torch.sqrt((1 - a_t_) / (1 - a_t)) * torch.sqrt(1 - a_t/a_t_)
                x_t = torch.sqrt(a_t_ / a_t) * (x_t - torch.sqrt(1 - a_t) * eps_t) + torch.sqrt(1 - a_t_ - sigma_t ** 2) * eps_t + sigma_t * eps_z
                if(t in steps):
                    z_list.append(x_t.detach())
        return z_list

if __name__=="__main__":
    layer = DiffusionLayers(24, 32, 512, 512)
    x0 = torch.randn(4, 8, 32)
    cond = torch.randn(4, 8, 512)
    loss = layer.loss(x0, cond)
    output = layer.inference(cond)
    print("Loss:", loss)
    print("Output:", output.shape)
