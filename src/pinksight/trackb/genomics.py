
from __future__ import annotations

import torch
from torch import nn


class GenomicsEncoder(nn.Module):

    def __init__(self, in_dim: int, embed_dim: int = 128, hidden_dim: int = 256,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x)
