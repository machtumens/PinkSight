"""P04 sub-step (2): DCE-MRI 3D-CNN encoder (phases-as-channels -> embedding).

Backbone is OPEN LOCK **O-1**. The locked combo "DenseNet121-3D + MedicalNet" ([3.1]+[3.2]) is
INVALID — MedicalNet ships pretrained weights only for 3D-ResNet, never DenseNet. The default here
is the ADR-0001-recommended **3D-ResNet-18 + MedicalNet**, but the backbone is **NOT ratified**: it
is decided at G2 against the unimodal-encoder number (LOCK-4). Do not treat this default as final.

Input is `preprocess_patient()`'s `(C, row, col, slice)` stack with the DCE phases as channels
([3.3] phases-as-channels). The encoder returns a pooled embedding only — the subtype/Ki-67 heads
are P06, out of scope here.

ponytail: the backbone is `monai.networks.nets.resnet{18,34}` (spatial_dims=3), not a hand-rolled
ResNet. MedicalNet first-conv channel adaptation (1 -> C) is deferred to the real training run
(Tier 2) — until weights are provisioned, `medicalnet_weights` is a graceful no-op.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import torch
from torch import nn

from monai.networks.nets import resnet18, resnet34

_BACKBONES = {18: resnet18, 34: resnet34}
EMBED_DIM = 512  # resnet18/34 pooled feature dim (BasicBlock expansion == 1)


class MriEncoder(nn.Module):
    """3D-ResNet DCE-MRI encoder (O-1 default, unratified). forward: (N, C, D, H, W) -> (N, 512)."""

    def __init__(
        self,
        in_channels: int = 4,
        depth: int = 18,
        medicalnet_weights: str | Path | None = None,
        freeze_bn: bool = False,
    ) -> None:
        super().__init__()
        if depth not in _BACKBONES:
            raise ValueError(
                f"depth must be one of {sorted(_BACKBONES)} (O-1: 3D-ResNet-18/34), got {depth}"
            )
        self.embed_dim = EMBED_DIM
        self._freeze_bn = freeze_bn
        # shortcut_type='A' + bias_downsample=False is the MedicalNet 3D-ResNet-18/34 convention,
        # kept so genuine MedicalNet transfer stays possible once weights land.
        self.backbone = _BACKBONES[depth](
            spatial_dims=3,
            n_input_channels=in_channels,
            shortcut_type="A",
            bias_downsample=False,
            feed_forward=False,  # return the pooled embedding, not class logits
        )
        if medicalnet_weights is not None:
            self._load_medicalnet(medicalnet_weights)
        if freeze_bn:
            self._apply_freeze_bn()

    def _apply_freeze_bn(self) -> None:
        """Put every BatchNorm3d in eval-mode + freeze its affine params (batch=1 BN fix [3.12]).

        With batch=1 the per-step BN statistics are a single voxel-mean and destabilise transfer;
        freezing keeps the running stats fixed and the affine (weight/bias) untrained. Paired with
        the ``train()`` override below so the epoch-start ``model.train()`` cannot silently thaw it.
        """
        for m in self.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):  # covers BatchNorm3d
                m.eval()
                if m.weight is not None:
                    m.weight.requires_grad_(False)
                if m.bias is not None:
                    m.bias.requires_grad_(False)

    def train(self, mode: bool = True) -> "MriEncoder":
        """Override so ``model.train()`` never un-freezes BN when ``freeze_bn`` is set.

        No-op vs ``nn.Module.train`` when ``self._freeze_bn`` is False (default) — behaviour is
        byte-identical for existing runs. When frozen, re-assert BN eval-mode after the base call.
        """
        super().train(mode)
        if self._freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Penultimate pooled feature vector, (N, C, D, H, W) -> (N, 512).

        Explicit alias for `forward`: with `feed_forward=False` the backbone already returns the
        pooled embedding BEFORE any classifier head, so this is exactly the frozen per-patient MRI
        embedding the fusion arm consumes (`mri_embed_s{SEED}.npz`, key `emb`). Named so the OOF
        embedding-export step (train/cv.py, flag-gated) reads the same vector whether it holds a raw
        `MriEncoder` or one wrapped in `models.heads.SubtypeClassifier` (via `.encoder`)."""
        return self.backbone(x)

    def _load_medicalnet(self, path: str | Path) -> None:
        """Load MedicalNet pretrained weights if present; graceful no-op otherwise (O-1 transfer)."""
        p = Path(path)
        if not p.exists():
            warnings.warn(
                f"MedicalNet weights not found at {p}; using random init (O-1 unresolved).",
                stacklevel=2,
            )
            return
        state = torch.load(p, map_location="cpu", weights_only=True)
        state = state.get("state_dict", state) if isinstance(state, dict) else state
        state = {k.removeprefix("module."): v for k, v in state.items()}
        # strict=False ignores missing/unexpected keys but still RAISES on a shape mismatch for a
        # key present in both — so the 1-ch pretrain conv1 (vs in_channels) + fc must be dropped
        # explicitly here. Full channel-inflation is deferred to the Tier-2 training run.
        model_sd = self.backbone.state_dict()
        state = {
            k: v for k, v in state.items()
            if k in model_sd and v.shape == model_sd[k].shape
        }
        missing, unexpected = self.backbone.load_state_dict(state, strict=False)
        warnings.warn(
            f"MedicalNet load: {len(missing)} missing / {len(unexpected)} unexpected keys.",
            stacklevel=2,
        )
