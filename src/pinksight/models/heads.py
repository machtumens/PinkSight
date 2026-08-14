"""P06: classification head(s) on top of an encoder embedding.

ponytail: a head is one `nn.Linear` — no MLP, no dropout stack until a real run shows it helps. The
wrapper is model-agnostic (any encoder exposing `.embed_dim` and `forward(x)->(N,embed_dim)`), which
is what lets the smoke test inject a 3-line synthetic encoder instead of the 3D-ResNet.
"""

from __future__ import annotations

import torch
from torch import nn


class SubtypeClassifier(nn.Module):
    """encoder -> Linear(embed_dim, 1) logits. forward: (N, C, D, H, W) -> (N, 1)."""

    def __init__(self, encoder: nn.Module, embed_dim: int | None = None) -> None:
        super().__init__()
        self.encoder = encoder
        dim = embed_dim if embed_dim is not None else getattr(encoder, "embed_dim", 512)
        self.head = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))
