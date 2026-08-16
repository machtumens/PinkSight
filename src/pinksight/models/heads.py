
from __future__ import annotations

import torch
from torch import nn


class SubtypeClassifier(nn.Module):

    def __init__(self, encoder: nn.Module, embed_dim: int | None = None) -> None:
        super().__init__()
        self.encoder = encoder
        dim = embed_dim if embed_dim is not None else getattr(encoder, "embed_dim", 512)
        self.head = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))
