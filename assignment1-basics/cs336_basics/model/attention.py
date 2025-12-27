from cs336_basics.model.scp_attention import SCPAttention
import torch.nn as nn
import torch
from .linear import Linear
from torch import Tensor
from .rope import RotaryPositionalEmbedding
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope: RotaryPositionalEmbedding | None = None):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_head = d_model//num_heads
        d_k = self.d_head
        self.q_proj = Linear(d_model, num_heads*d_k)
        self.k_proj = Linear(d_model, num_heads*d_k)
        self.v_proj = Linear(d_model, num_heads*d_k)
        self.output_proj = Linear(num_heads*d_k, d_model)
        self.rope = rope
    def forward(self, x: Tensor, token_positions: Tensor = None):
        B, T, _ = x.shape
        h = self.num_heads
        d = self.d_head

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, T, h, d).transpose(1, 2)
        k = k.view(B, T, h, d).transpose(1, 2)
        v = v.view(B, T, h, d).transpose(1, 2)

        token_positions = torch.arange(T, device=x.device)
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(T, device=x.device)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        q = q.reshape(B * h, T, d)
        k = k.reshape(B * h, T, d)
        v = v.reshape(B * h, T, d)

        causal_mask = torch.tril(
            torch.ones(T, T, device=x.device, dtype=torch.bool)
        )
        causal_mask = causal_mask.unsqueeze(0)

        out = SCPAttention(q, k, v, mask=causal_mask)

        out = out.view(B, h, T, d)
        out = out.transpose(1, 2).contiguous()  
        out = out.view(B, T, h * d)

        out = self.output_proj(out)

        return out


