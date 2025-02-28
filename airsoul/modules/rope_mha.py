import copy
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, scale=None) -> torch.Tensor:
    # Efficient implementation equivalent to the following:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    if attn_mask is not None:
        attn_weight += attn_mask
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)

    return attn_weight @ value

# Modified from facebookresearch/llama/model.py
def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    q0_pos: int = 0,
    k0_pos: int = 0):
    """
    q0_pos: the start position of xq
    k0_pos: the start position of xk
    """
    bsz, q_len, _, _ = xq.shape
    k_len = xk.shape[1]

    freqs_cis = freqs_cis.to(xq.device)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    xqm_ = freqs_cis[:, q0_pos:(q0_pos + q_len)]
    xkm_ = freqs_cis[:, k0_pos:(k0_pos + k_len)]
    xq_out = torch.view_as_real(xq_ * xqm_).flatten(3)
    xk_out = torch.view_as_real(xk_ * xkm_).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)

# Taken from facebookresearch/llama/model.py
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
    )
    t = torch.arange(end)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    # Modify to dimension: [1, seq, 1, d_head]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs).unsqueeze(0).unsqueeze(2)  # complex64
    freqs_cis.requires_grad_(False)

    return freqs_cis

class RoPEMultiheadAttention(nn.Module):
    def __init__(self, d_model, nheads, dropout=0.10):
        super().__init__()

        self.drop_p = dropout
        self.n_heads = nheads
        self.d_head = d_model // nheads

        # Attention
        self.qlayer = nn.Linear(
            in_features=d_model,
            out_features=d_model,
            bias=False,
        )
        self.klayer = nn.Linear(
            in_features=d_model,
            out_features=d_model,
            bias=False,
        )
        self.vlayer = nn.Linear(
            in_features=d_model,
            out_features=d_model,
            bias=False,
        )
        self.att_proj_linear = nn.Linear(
            in_features=d_model,
            out_features=d_model,
        )
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, freqs_cis: torch.Tensor, attn_mask=None, q0_pos=0, k0_pos=0):
        """
        x_shape: [bs, seq, hidden]
        """
        batch_size, q_len, _ = q.shape
        batch_size, k_len, _ = k.shape
        batch_size, v_len, _ = v.shape
        xq, xk, xv = self.qlayer(q), self.klayer(k), self.vlayer(v)

        # Reshape for rotary embeddings
        xq = xq.view(batch_size, q_len, self.n_heads, self.d_head)
        xk = xk.view(batch_size, k_len, self.n_heads, self.d_head)
        xv = xv.view(batch_size, v_len, self.n_heads, self.d_head)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis, q0_pos=q0_pos, k0_pos=k0_pos)

        # Reshape for attention calculation: (b_sz, n_head, s_len, d_head)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Required as we are not using a nn.Dropout layer
        if self.training:
            att_dropout = self.drop_p
        else:
            att_dropout = 0.0

        # Using beta torch functionality (subject to change)
        # See - https://shorturl.at/jtI17
        att = scaled_dot_product_attention(
            query=xq,
            key=xk,
            value=xv,
            attn_mask=attn_mask,
            dropout_p=att_dropout,
        )

        # Shape (b_sz, s_len, n_head, d_head)
        out = att.transpose(1, 2).contiguous()
        out = out.view(batch_size, q_len, self.n_heads * self.d_head)

        return self.resid_dropout(self.att_proj_linear(out))

if __name__=="__main__":
    rmha = RoPEMultiheadAttention(128, 8, 0.10)
    freqs_cis = precompute_freqs_cis(128 // 8, 1024)
    q = torch.randn(4, 512, 128)
    k = torch.randn(4, 1024, 128)
    v = torch.randn(4, 1024, 128)
    output = rmha(q, k, v, freqs_cis, attn_mask=None, q0_pos=512, k0_pos=0)
    print(output.shape)
