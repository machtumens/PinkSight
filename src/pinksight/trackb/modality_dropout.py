
from __future__ import annotations

import torch
from torch import nn


class ModalityDropoutFusion(nn.Module):

    def __init__(self, modality_dims: dict[str, int], fused_dim: int = 128) -> None:
        super().__init__()
        if not modality_dims:
            raise ValueError("need at least one modality")
        self.fused_dim = fused_dim
        self.embed_dim = fused_dim  
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
        return torch.stack(projected, dim=0).mean(dim=0)  
