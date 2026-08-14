"""[G2-RADIMAGENET-FROZEN-2P5D] Frozen 2.5D slice encoder — pretrained ResNet50 -> pooled features.

Arm `r50_radimagenet_frozen_2p5d`. A FROZEN pretrained encoder consumed as a fixed feature
extractor: every parameter has `requires_grad=False`, the module lives in `eval()` permanently, and
`forward` is wrapped in `@torch.no_grad()` so a caller that forgets the context manager still cannot
accumulate gradients through the frozen activations (execute-agent instruction E4).

BACKBONE RESOLVED IN PHASE 0 (execute-agent instruction E1): `resnet50_radimagenet`.
  - Weights: RadImageNet ResNet50 (2D radiology, 1.35M imgs, 165 classes), a genuine torchvision
    `resnet50` state-dict. 318/318 non-fc body layers load into torchvision `resnet50`; only the
    165-class `fc` differs from ImageNet's 1000 and is stripped here (we keep the 2048-d pooled
    feature). EMBED_DIM = 2048.
  - DINOv2 ViT-S/14 fallback (EMBED_DIM = 384) was NOT needed — RadImageNet weights were obtained
    inside the 30-min Phase 0 cap.

Download command (weights are gitignored under data/pretrained/):
    from huggingface_hub import hf_hub_download
    hf_hub_download("convergedmachine/RadImagenet", "resnet50.pth",
                    local_dir="data/pretrained")   # then rename to resnet50_radimagenet.pth
The checkpoint is a torch.compile training state: top-level dict with a nested "model" key whose
tensor keys carry an `_orig_mod.` prefix. `_load_state_dict` below strips `_orig_mod.`/`module.` and
drops the fc so the 2048-d body loads cleanly into a torchvision resnet50.

The DINOv2 fallback path (`backbone="dinov2_vits14"`, torch.hub, 384-d) is kept live so the same
module and probe pipeline work if RadImageNet weights ever become unavailable.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

# Output dims per backbone. resnet50 pooled = 2048; DINOv2 ViT-S/14 CLS token = 384.
_EMBED_DIM = {"resnet50_radimagenet": 2048, "dinov2_vits14": 384}
_BACKBONES = tuple(_EMBED_DIM)


class SliceEncoder(nn.Module):
    """Frozen 2.5D slice encoder. forward: (N, 3, 224, 224) -> (N, EMBED_DIM), no_grad, eval-only.

    All parameters are frozen at construction and the module refuses to leave eval mode (the
    ``train()`` override mirrors ``MriEncoder.train()`` so an epoch-start ``model.train()`` cannot
    silently thaw it — though this arm never trains the encoder at all).
    """

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
        else:  # dinov2_vits14 — fallback (torch.hub); no local weights file needed
            self.net = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")

        # Freeze EVERY parameter (no gradient ever flows into the encoder).
        for p in self.parameters():
            p.requires_grad_(False)
        # Permanent eval mode — set now and re-asserted by the train() override below.
        self.eval()

    @staticmethod
    def _build_resnet50(weights_path: str | Path | None) -> nn.Module:
        """torchvision resnet50 with the classifier fc replaced by Identity -> returns 2048-d pooled.

        Loads RadImageNet weights from `weights_path` when given; a missing file raises so a run can
        never silently proceed on random weights (the whole arm is a *pretrained* feature-extractor
        bet — random init would be a scientifically meaningless NULL, not an honest one)."""
        from torchvision.models import resnet50

        net = resnet50(weights=None)
        net.fc = nn.Identity()  # strip the 1000/165-class head -> avgpool output (2048-d) is `forward`
        if weights_path is not None:
            SliceEncoder._load_state_dict(net, Path(weights_path))
        return net

    @staticmethod
    def _load_state_dict(net: nn.Module, path: Path) -> None:
        """Load a RadImageNet resnet50 checkpoint into a headless torchvision resnet50.

        Handles the torch.compile training-state shape: {"model": OrderedDict(_orig_mod.<layer>...)}.
        Strips `_orig_mod.`/`module.` prefixes, drops `fc.*` (165-class head, incompatible with the
        Identity head and not needed for pooled features), and asserts the body actually loaded so a
        silently-random encoder can never slip through."""
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
        # `fc.*` is the ONLY key we expect torchvision to report missing (we replaced it with
        # Identity, and we deliberately dropped the 165-class head). Anything else missing means the
        # port did not map onto resnet50 — fail loudly rather than run on a half-random encoder.
        non_fc_missing = [k for k in missing if not k.startswith("fc.")]
        if non_fc_missing:
            raise RuntimeError(
                f"RadImageNet load mapped incompletely: {len(non_fc_missing)} non-fc keys missing "
                f"(e.g. {non_fc_missing[:5]}). Checkpoint is not a torchvision resnet50 body."
            )

    def train(self, mode: bool = True) -> "SliceEncoder":
        """Override so ``model.train()`` never leaves eval mode — the encoder is permanently frozen.

        Mirrors ``MriEncoder.train()`` (freeze-BN guard). Regardless of `mode`, the module is forced
        back to eval so BatchNorm running stats and dropout stay in inference behaviour."""
        super().train(mode)
        super().train(False)  # force eval no matter what was requested
        return self

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N, 3, 224, 224) float32 -> (N, EMBED_DIM) pooled features. no_grad (E4), eval-only.

        DINOv2 exposes pooled CLS features via `forward_features(x)["x_norm_clstoken"]`; torchvision
        resnet50-with-Identity-fc returns the 2048-d pooled vector directly from `forward`."""
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"expected (N, 3, H, W), got {tuple(x.shape)}")
        if self.backbone_name == "dinov2_vits14":
            feats = self.net.forward_features(x)["x_norm_clstoken"]
        else:
            feats = self.net(x)
        return feats.reshape(feats.shape[0], -1)
