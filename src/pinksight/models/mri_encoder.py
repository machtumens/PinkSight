
from __future__ import annotations

import warnings
from pathlib import Path

import torch
from torch import nn

from monai.networks.nets import resnet18, resnet34

_BACKBONES = {18: resnet18, 34: resnet34}
EMBED_DIM = 512  


class MriEncoder(nn.Module):

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
        self.backbone = _BACKBONES[depth](
            spatial_dims=3,
            n_input_channels=in_channels,
            shortcut_type="A",
            bias_downsample=False,
            feed_forward=False,  
        )
        if medicalnet_weights is not None:
            self._load_medicalnet(medicalnet_weights)
        if freeze_bn:
            self._apply_freeze_bn()

    def _apply_freeze_bn(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):  
                m.eval()
                if m.weight is not None:
                    m.weight.requires_grad_(False)
                if m.bias is not None:
                    m.bias.requires_grad_(False)

    def train(self, mode: bool = True) -> "MriEncoder":
        super().train(mode)
        if self._freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def _load_medicalnet(self, path: str | Path) -> None:
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
