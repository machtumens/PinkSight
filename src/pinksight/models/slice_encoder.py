
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

_EMBED_DIM = {"resnet50_radimagenet": 2048, "dinov2_vits14": 384}
_BACKBONES = tuple(_EMBED_DIM)


class SliceEncoder(nn.Module):

    def __init__(
        self,
        backbone: str = "resnet50_radimagenet",
        weights_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        if backbone not in _BACKBONES:
            raise ValueError(f"backbone must be one of {list(_BACKBONES)}, got {backbone!r}")
        self.backbone_name = backbone
        self.embed_dim = _EMBED_DIM[backbone]

        if backbone == "resnet50_radimagenet":
            self.net = self._build_resnet50(weights_path)
        else:  
            self.net = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")

        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    @staticmethod
    def _build_resnet50(weights_path: str | Path | None) -> nn.Module:
        from torchvision.models import resnet50

        net = resnet50(weights=None)
        net.fc = nn.Identity()  
        if weights_path is not None:
            SliceEncoder._load_state_dict(net, Path(weights_path))
        return net

    @staticmethod
    def _load_state_dict(net: nn.Module, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(
                f"RadImageNet weights not found at {path}. Provide --weights pointing at "
                "data/pretrained/resnet50_radimagenet.pth (see the module docstring for the "
                "huggingface_hub download command), or use --backbone dinov2_vits14 for the fallback."
            )
        ck = torch.load(path, map_location="cpu", weights_only=False)
        state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if hasattr(state, "state_dict"):
            state = state.state_dict()

        def _strip(k: str) -> str:
            for pre in ("module.", "_orig_mod."):
                k = k.removeprefix(pre)
            return k

        cleaned = {_strip(k): v for k, v in state.items() if not _strip(k).startswith("fc.")}
        missing, unexpected = net.load_state_dict(cleaned, strict=False)
        non_fc_missing = [k for k in missing if not k.startswith("fc.")]
        if non_fc_missing:
            raise RuntimeError(
                f"RadImageNet load mapped incompletely: {len(non_fc_missing)} non-fc keys missing "
                f"(e.g. {non_fc_missing[:5]}). Checkpoint is not a torchvision resnet50 body."
            )

    def train(self, mode: bool = True) -> "SliceEncoder":
        super().train(mode)
        super().train(False)  
        return self

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"expected (N, 3, H, W), got {tuple(x.shape)}")
        if self.backbone_name == "dinov2_vits14":
            feats = self.net.forward_features(x)["x_norm_clstoken"]
        else:
            feats = self.net(x)
        return feats.reshape(feats.shape[0], -1)
