
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class GatedAttentionMIL(nn.Module):

    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int | None = None) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = int(out_dim if out_dim is not None else in_dim)
        self.embed_dim = self.out_dim  
        self.attn_v = nn.Linear(in_dim, hidden_dim)   
        self.attn_u = nn.Linear(in_dim, hidden_dim)   
        self.attn_w = nn.Linear(hidden_dim, 1)
        self.project = nn.Linear(in_dim, self.out_dim)

    def forward(self, bag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if bag.dim() != 2:
            raise ValueError(f"expected a [N_tiles, in_dim] bag, got shape {tuple(bag.shape)}")
        gated = torch.tanh(self.attn_v(bag)) * torch.sigmoid(self.attn_u(bag))  
        scores = self.attn_w(gated).squeeze(-1)                                 
        attn = F.softmax(scores, dim=0)                                         
        bag_embedding = self.project(attn @ bag)                               
        return bag_embedding, attn
