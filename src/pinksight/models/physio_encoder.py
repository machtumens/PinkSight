
from __future__ import annotations

import torch
from torch import nn

from .mri_encoder import EMBED_DIM

N_PHYSIO_FEATURES = 7  


class PhysioEncoder(nn.Module):

    def __init__(self, in_features: int = N_PHYSIO_FEATURES, hidden: int = 64,
                 out_dim: int = EMBED_DIM) -> None:
        super().__init__()
        self.in_features = in_features
        self.hidden = hidden
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
            nn.BatchNorm1d(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.shape[1] != self.in_features:
            raise ValueError(
                f"PhysioEncoder expects (N, {self.in_features}) input, got shape {tuple(x.shape)}"
            )
        return self.net(x)
