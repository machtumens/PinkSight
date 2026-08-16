
from __future__ import annotations

import torch
from torch import nn


class PcrHead(nn.Module):

    def __init__(self, encoder: nn.Module, embed_dim: int | None = None) -> None:
        super().__init__()
        self.encoder = encoder
        dim = embed_dim if embed_dim is not None else getattr(encoder, "embed_dim", 512)
        self.head = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


def build_pcr_model(
    in_channels: int = 3,
    depth: int = 18,
    medicalnet_weights: str | None = None,
    freeze_bn: bool = False,
) -> PcrHead:
    from pinksight.models.mri_encoder import MriEncoder

    encoder = MriEncoder(
        in_channels=in_channels,
        depth=depth,
        medicalnet_weights=medicalnet_weights,
        freeze_bn=freeze_bn,
    )
    return PcrHead(encoder, embed_dim=encoder.embed_dim)


def _selfcheck() -> int:
    torch.manual_seed(0)

    class _TinyEncoder(nn.Module):

        embed_dim = 8

        def __init__(self, in_channels: int) -> None:
            super().__init__()
            self.proj = nn.Linear(in_channels, self.embed_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            pooled = x.mean(dim=(2, 3, 4))  
            return self.proj(pooled)  

    model = PcrHead(_TinyEncoder(in_channels=3))
    x = torch.randn(4, 3, 8, 8, 8)
    out = model(x)
    assert out.shape == (4, 1), out.shape
    assert torch.isfinite(out).all(), "pCR head produced non-finite logits"

    loss = nn.functional.binary_cross_entropy_with_logits(out, (torch.rand(4, 1) > 0.5).float())
    loss.backward()
    assert model.head.weight.grad is not None and torch.isfinite(model.head.weight.grad).all(), (
        "pCR head weight received no finite gradient"
    )

    assert PcrHead(_TinyEncoder(3), embed_dim=8).head.out_features == 1

    print("pcr_head selfcheck OK — (N,3,D,H,W)->(N,1) logits, finite, differentiable head.")  
    return 0


if __name__ == "__main__":
    raise SystemExit(_selfcheck())
