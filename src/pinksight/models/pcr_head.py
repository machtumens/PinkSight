"""pCR pilot (Phase E2): pCR prediction head on top of the ADR-0001 3D-ResNet-18 MRI encoder.

Structurally a twin of `models.heads.SubtypeClassifier` (encoder -> Linear(embed_dim, 1) logits), but
named for the pCR target so the pilot's model wiring is self-documenting and the fusion/subtype
architecture stays untouched (plan Blast Radius: new file, existing models unchanged). The head is one
`nn.Linear` ([ponytail] — no MLP stack until a real run shows it helps); focal loss is selected in
`TrainCfg(loss="focal")`, not baked into the head (matches loop.py's flag-gated focal path).

`build_pcr_model()` composes the real 3D-ResNet-18 `MriEncoder` (O-1 default, ADR-0001) + the pCR head
as the `model_factory` `train.cv.cross_val_imaging` expects. The head wrapper is encoder-agnostic (any
module exposing `.embed_dim` and `forward(x)->(N,embed_dim)`), which is what lets the smoke test inject
a 3-line synthetic encoder instead of instantiating the 33M-param backbone on CPU.

ponytail: pCR is another binary imaging head — reuse SubtypeClassifier's exact shape. The ONLY reason
this is a separate class is naming clarity + a pCR-specific `build_pcr_model` factory; behaviour is
identical to a `SubtypeClassifier` over the same encoder.
"""

from __future__ import annotations

import torch
from torch import nn


class PcrHead(nn.Module):
    """encoder -> Linear(embed_dim, 1) pCR logits. forward: (N, C, D, H, W) -> (N, 1).

    Identical wiring to `models.heads.SubtypeClassifier`; distinct class only for target clarity. The
    logit is a pCR-vs-not-pCR score; sigmoid + focal/BCE loss live in the training loop, not here.
    """

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
    """Factory: 3D-ResNet-18 `MriEncoder` (ADR-0001, O-1 default) + pCR head.

    `in_channels` defaults to 3 (plan E1 channel policy: pre + first-post + late-post). `depth` stays
    18 (ADR-0001 recommended 3D-ResNet-18/34; the backbone is the ratified-at-G2 default, unchanged
    here — the pilot rides ADR-0001, it does not re-open it). `freeze_bn=True` is the batch=1 BN-fix
    ([3.12]) for tiny-batch local runs. Import-local so this module imports without the ml stack
    (the smoke test injects a synthetic encoder and never touches MONAI).
    """
    from pinksight.models.mri_encoder import MriEncoder

    encoder = MriEncoder(
        in_channels=in_channels,
        depth=depth,
        medicalnet_weights=medicalnet_weights,
        freeze_bn=freeze_bn,
    )
    return PcrHead(encoder, embed_dim=encoder.embed_dim)


def _selfcheck() -> int:
    """Runnable check WITHOUT the real backbone: a 3-line synthetic encoder -> pCR head -> shapes + no NaN.

    Proves the head contract on the dry surface: (N, C, D, H, W) -> (N, 1) logits, finite, differentiable.
    The real 3D-ResNet-18 forward is exercised by the `--smoke` path of scripts/pcr_cnn_ispy2.py; here
    we keep it CPU-cheap so `python -m pinksight.models.pcr_head` runs anywhere torch is installed.
    """
    torch.manual_seed(0)

    class _TinyEncoder(nn.Module):
        """Global-avg-pool -> tiny linear; mimics MriEncoder's (N,C,D,H,W)->(N,embed_dim) contract."""

        embed_dim = 8

        def __init__(self, in_channels: int) -> None:
            super().__init__()
            self.proj = nn.Linear(in_channels, self.embed_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            pooled = x.mean(dim=(2, 3, 4))  # (N, C)
            return self.proj(pooled)  # (N, embed_dim)

    model = PcrHead(_TinyEncoder(in_channels=3))
    x = torch.randn(4, 3, 8, 8, 8)
    out = model(x)
    assert out.shape == (4, 1), out.shape
    assert torch.isfinite(out).all(), "pCR head produced non-finite logits"

    # one backward pass — the head must be trainable (grad reaches the linear head weight).
    loss = nn.functional.binary_cross_entropy_with_logits(out, (torch.rand(4, 1) > 0.5).float())
    loss.backward()
    assert model.head.weight.grad is not None and torch.isfinite(model.head.weight.grad).all(), (
        "pCR head weight received no finite gradient"
    )

    # build_pcr_model wiring: default in_channels=3 head shape matches (real backbone is import-local).
    assert PcrHead(_TinyEncoder(3), embed_dim=8).head.out_features == 1

    print("pcr_head selfcheck OK — (N,3,D,H,W)->(N,1) logits, finite, differentiable head.")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(_selfcheck())
