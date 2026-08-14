"""Modality-dropout fusion (decisions.md [5.2] / [0.1]): attaches {WSI, genomics} (and optionally
the MRI/clinical bus) with GRACEFUL DEGRADATION when a modality is absent.

Track B is HARD-GATED (LOCK-6, dormant until Track A clears G5) and NOT patient-matched to MRI; the
~84-patient Duke∩TCGA overlap MUST be de-duplicated before any "external" claim.

Each modality is projected to a shared width; present modalities are mean-pooled; during training a
modality can be randomly dropped so the model learns to predict from any available subset. At
inference, a missing modality is simply omitted — the fused vector is still valid (this is the
mechanism [5.2]/[0.1] describe). Requires >=1 present modality.

ponytail: dict-in, projected, averaged. No gating network until a real run shows it helps.
"""

from __future__ import annotations

import torch
from torch import nn


class ModalityDropoutFusion(nn.Module):
    """Project each named modality embedding to `fused_dim`, then average the PRESENT ones.

    modality_dims: {name: in_dim}. forward(feats, drop=...) where feats is {name: [N, in_dim]} for
    the present modalities only. Absent modalities are just left out of the dict -> graceful
    degradation. `drop` (train-time) additionally removes named modalities at random-caller control.
    """

    def __init__(self, modality_dims: dict[str, int], fused_dim: int = 128) -> None:
        super().__init__()
        if not modality_dims:
            raise ValueError("need at least one modality")
        self.fused_dim = fused_dim
        self.embed_dim = fused_dim  # head-agnostic contract
        self.projections = nn.ModuleDict(
            {name: nn.Linear(dim, fused_dim) for name, dim in modality_dims.items()}
        )

    def forward(self, feats: dict[str, torch.Tensor],
                drop: set[str] | None = None) -> torch.Tensor:
        drop = drop or set()
        present = [name for name in feats if name in self.projections and name not in drop]
        if not present:
            raise ValueError("all modalities dropped/absent — cannot fuse an empty set")
        projected = []
        for name in present:
            x = feats[name]
            if x.dim() == 1:
                x = x.unsqueeze(0)
            projected.append(torch.relu(self.projections[name](x)))
        return torch.stack(projected, dim=0).mean(dim=0)  # [N, fused_dim]
