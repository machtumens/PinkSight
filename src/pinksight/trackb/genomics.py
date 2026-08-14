"""Genomic MLP encoder (decisions.md [5.2]): a label-safe tabular genomic vector -> embedding.

Track B is HARD-GATED (LOCK-6, dormant until Track A clears G5) and NOT patient-matched to MRI; the
~84-patient Duke∩TCGA overlap MUST be de-duplicated before any "external" claim.

Label-safe: like Track A's [1.16] leak rule, ER/PR/HER2/Ki-67/Mol-Subtype/Oncotype and any direct
subtype-defining signal must NOT be inputs — the genomic vector is expression/CNA features only.
Enforcement lives at the data-assembly layer; this encoder is a plain MLP over whatever safe
vector it is handed.

ponytail: a 2-layer MLP. Exposes `.embed_dim` so heads.py-style heads attach unchanged.
"""

from __future__ import annotations

import torch
from torch import nn


class GenomicsEncoder(nn.Module):
    """tabular genomic vector [N, in_dim] -> embedding [N, embed_dim]."""

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
