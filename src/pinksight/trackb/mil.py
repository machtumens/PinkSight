"""Gated attention-MIL aggregator, CLAM-style (decisions.md [5.3]): a [N, D] tile-embedding bag ->
one bag embedding + per-tile attention weights (kept for XAI).

Track B is HARD-GATED (LOCK-6, dormant until Track A clears G5) and NOT patient-matched to MRI; the
~84-patient Duke∩TCGA overlap MUST be de-duplicated before any "external" claim.

ponytail: the minimal gated-attention block (Ilse et al. 2018 / CLAM). Small, torch, no bells.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class GatedAttentionMIL(nn.Module):
    """Gated attention pooling over a bag of tile embeddings.

    forward(bag) where bag is [N_tiles, in_dim] -> (bag_embedding [out_dim], attention [N_tiles]).
    Attention weights are softmax-normalised over the bag (sum to 1) and returned for XAI.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int | None = None) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = int(out_dim if out_dim is not None else in_dim)
        self.embed_dim = self.out_dim  # expose the head-agnostic contract (like heads.py encoders)
        self.attn_v = nn.Linear(in_dim, hidden_dim)   # tanh branch
        self.attn_u = nn.Linear(in_dim, hidden_dim)   # gate (sigmoid) branch
        self.attn_w = nn.Linear(hidden_dim, 1)
        self.project = nn.Linear(in_dim, self.out_dim)

    def forward(self, bag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if bag.dim() != 2:
            raise ValueError(f"expected a [N_tiles, in_dim] bag, got shape {tuple(bag.shape)}")
        gated = torch.tanh(self.attn_v(bag)) * torch.sigmoid(self.attn_u(bag))  # [N, hidden]
        scores = self.attn_w(gated).squeeze(-1)                                 # [N]
        attn = F.softmax(scores, dim=0)                                         # [N], sums to 1
        bag_embedding = self.project(attn @ bag)                               # [out_dim]
        return bag_embedding, attn
