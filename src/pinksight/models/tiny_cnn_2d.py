
from __future__ import annotations

import torch
from torch import nn


def _block(cin: int, cout: int) -> nn.Sequential:
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

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        widths = (32, 64, 128, 256, 512)  
        cin = in_channels
        blocks = []
        for w in widths:
            blocks.append(_block(cin, w))
            cin = w
        self.features = nn.Sequential(*blocks)
        self.gap = nn.AdaptiveAvgPool2d(1)  
        self.classifier = nn.Sequential(
            nn.Flatten(),          
            nn.Linear(512, 256),   
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),       
            nn.Linear(256, 1),     
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.gap(self.features(x)))


if __name__ == "__main__":  
    m = TinyCnn2D(in_channels=3)
    out = m(torch.randn(4, 3, 64, 64))
    assert out.shape == (4, 1), out.shape
    n_params = sum(p.numel() for p in m.parameters())
    print(f"OK: TinyCnn2D (3,64,64)->(N,1) logits; {n_params/1e6:.2f}M params (from scratch, 2D)")  
