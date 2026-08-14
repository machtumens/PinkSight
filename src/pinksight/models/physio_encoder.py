"""A1-5: parallel semi-quantitative DCE physiology encoder (7 kinetic features -> bus embedding).

A small MLP that projects a per-patient 7-dim semi-quantitative DCE feature vector (wash-in slope,
time-to-peak, washout rate, iAUC, curve class, mean peak enhancement, spatial heterogeneity — see
``scripts/extract_semiquant_features.py``) into the same embedding bus the MRI encoder feeds. This is
the **parallel-encoder** arm of the A1 ablation: the MRI-Eye stays independently ablatable, so the
kill-gate ΔAUC between the raw-MRI arm and this physiology arm is clean.

Architecture (plan A1-5): ``Linear(7,64) -> BatchNorm1d(64) -> ReLU -> Linear(64,out_dim) ->
BatchNorm1d(out_dim)``. ``out_dim`` defaults to ``MriEncoder.EMBED_DIM`` (512, confirmed A1-1 / E2)
so both encoders write the same bus for future G3 cross-attention fusion.

Ledger (LOCK-1): these are **semi-quantitative DCE enhancement** features. Never pharmacokinetic
(Ktrans/kep/ve) — full Tofts is infeasible at Duke's ~100-130 s/phase.
"""

from __future__ import annotations

import torch
from torch import nn

from .mri_encoder import EMBED_DIM

N_PHYSIO_FEATURES = 7  # the fixed semi-quantitative DCE feature-vector length (plan A1 table)


class PhysioEncoder(nn.Module):
    """MLP encoder: (N, 7) semi-quantitative DCE features -> (N, out_dim) bus embedding.

    out_dim defaults to the MRI encoder's EMBED_DIM (512) so the physiology arm and the imaging arm
    share the fusion bus. hidden defaults to 64 (plan A1-5).
    """

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
        """(N, in_features) -> (N, out_dim). Requires N>=2 in train mode (BatchNorm needs a batch)."""
        if x.dim() != 2 or x.shape[1] != self.in_features:
            raise ValueError(
                f"PhysioEncoder expects (N, {self.in_features}) input, got shape {tuple(x.shape)}"
            )
        return self.net(x)
