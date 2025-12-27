import torch.nn as nn
import torch
from torch import Tensor
import math
from .softmax import softmax
def SCPAttention(q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None):
    assert q.shape[-1] == k.shape[-1]
    assert k.shape[-2] == v.shape[-2]

    d_k = q.shape[-1]
    attn_obj = torch.matmul(q, k.transpose(-2, -1))
    attn_obj = attn_obj / math.sqrt(d_k)
    if mask is not None:
        attn_obj = attn_obj.masked_fill(mask==0, -float("inf"))
    attn_obj = softmax(attn_obj, dim=-1)
    result = torch.matmul(attn_obj, v)
    return result
