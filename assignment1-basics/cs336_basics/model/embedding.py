import torch.nn as nn
import torch
from jaxtyping import Float, Int
from torch import Tensor
class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device = None, dtype = None):
        super().__init__()
        self.num_embeddings = num_embeddings # vocab_size
        self.embedding_dim = embedding_dim # dimension of embedding vector
        self.device: torch.device | None = device
        self.dtype: torch.dtype | None = dtype
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, dtype=self.dtype, device=self.device))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=0.02)
    def forward(self, token_ids: Int[Tensor, " ..."]) -> torch.Tensor:
       # Given a list of token ids, we want to find the vector for a 
       # corresponding token by performing a lookup into a tensor of 
       # token ids with shape (num_embeddings, embedding_dim)
       return self.weight[token_ids]