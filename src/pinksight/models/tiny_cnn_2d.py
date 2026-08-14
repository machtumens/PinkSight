"""[HEAD2-GRADE-2D-SLICE] Tiny from-scratch 2D CNN — the DeepRadGrade model class.

DeepRadGrade (Eur Radiol 2025, PMC13086883) explicitly abandoned pretrained/3D/ResNet backbones
("overfit on this exact Duke cohort") and won with a SMALL sequential 2D CNN trained FROM SCRATCH:
3x3 convs with channel-doubling up to 512 feature maps, then FC 256 -> 1 -> sigmoid. This module
replicates that model class. NOT pretrained. fp32 (the loop.py non-finite guard is always on).

Input (3, 64, 64) = 3 post-contrast phases as channels. Output (N, 1) logit (sigmoid applied in the
training loop's loss / eval, matching SubtypeClassifier's contract) so it slots into `train_model`
and `cross_val_slices` unchanged.
"""

from __future__ import annotations

import torch
from torch import nn


def _block(cin: int, cout: int) -> nn.Sequential:
    """conv3x3 -> BN -> ReLU -> conv3x3 -> BN -> ReLU -> maxpool2. Halves the spatial size, sets the
    channel width. BatchNorm keeps the from-scratch net trainable at small N (DeepRadGrade style)."""
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=3, padding=1),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, kernel_size=3, padding=1),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class TinyCnn2D(nn.Module):
    """From-scratch 2D CNN: 3x3 convs, channel-doubling 32->64->128->256->512, GAP, FC 256->1.

    64x64 -> (32,32) -> (16,16) -> (8,8) -> (4,4) -> (2,2) after five channel-doubling blocks, then
    global-average-pool to a 512-vector, FC 512->256->1. `forward` returns (N, 1) logits (no sigmoid;
    the loss/eval applies it) so it is a drop-in for the existing training loop."""

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        widths = (32, 64, 128, 256, 512)  # channel-doubling up to 512 feature maps (DeepRadGrade)
        cin = in_channels
        blocks = []
        for w in widths:
            blocks.append(_block(cin, w))
            cin = w
        self.features = nn.Sequential(*blocks)
        self.gap = nn.AdaptiveAvgPool2d(1)  # -> (N, 512, 1, 1)
        self.classifier = nn.Sequential(
            nn.Flatten(),          # (N, 512)
            nn.Linear(512, 256),   # FC 512 -> 256
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),       # light regularisation for the small from-scratch net
            nn.Linear(256, 1),     # FC 256 -> 1 (sigmoid applied downstream)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.gap(self.features(x)))


if __name__ == "__main__":  # runnable check: `python -m pinksight.models.tiny_cnn_2d`
    m = TinyCnn2D(in_channels=3)
    out = m(torch.randn(4, 3, 64, 64))
    assert out.shape == (4, 1), out.shape
    n_params = sum(p.numel() for p in m.parameters())
    print(f"OK: TinyCnn2D (3,64,64)->(N,1) logits; {n_params/1e6:.2f}M params (from scratch, 2D)")  # noqa: T201
