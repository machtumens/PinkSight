
from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

FOUNDATION_DIMS: dict[str, int] = {
    "uni": 1536,       
    "conch": 512,      
    "ctranspath": 768,  
}

_UNI_DIR = Path("data/pathology/uni")
DEFAULT_UNI_WEIGHTS = _UNI_DIR / "pytorch_model.bin"
DEFAULT_UNI_CONFIG = _UNI_DIR / "config.json"


@runtime_checkable
class TileEncoder(Protocol):

    name: str
    embed_dim: int

    def encode(self, tiles: np.ndarray) -> np.ndarray:
        ...


class SyntheticTileEncoder:

    def __init__(self, name: str = "uni", embed_dim: int | None = None, seed: int = 0) -> None:
        if name not in FOUNDATION_DIMS and embed_dim is None:
            raise ValueError(f"unknown foundation model {name!r}; pass embed_dim explicitly")
        self.name = name
        self.embed_dim = int(embed_dim if embed_dim is not None else FOUNDATION_DIMS[name])
        self._seed = seed

    def encode(self, tiles: np.ndarray) -> np.ndarray:
        tiles = np.asarray(tiles, dtype=np.float32)
        n = tiles.shape[0]
        flat = tiles.reshape(n, -1)
        d_in = flat.shape[1]
        rng = np.random.default_rng(self._seed)
        proj = rng.standard_normal((d_in, self.embed_dim), dtype=np.float64).astype(np.float32)
        proj /= np.sqrt(d_in)
        emb = flat @ proj
        norm = np.linalg.norm(emb, axis=1, keepdims=True)
        return (emb / np.clip(norm, 1e-8, None)).astype(np.float32)


class UNITileEncoder(TileEncoder):

    _TIMM_MODEL = "vit_giant_patch14_224"

    def __init__(
        self,
        name: str = "uni",
        weights_path: str | Path = DEFAULT_UNI_WEIGHTS,
        config_path: str | Path = DEFAULT_UNI_CONFIG,
        device: str = "cpu",
        batch_size: int = 128,
    ) -> None:
        self.name = name
        self.embed_dim = int(FOUNDATION_DIMS.get(name, 1536))
        self._weights_path = Path(weights_path)
        self._config_path = Path(config_path)
        self._device = device
        self._batch_size = int(batch_size)
        self._model = None  
        cfg = self._read_config()
        pc = cfg.get("pretrained_cfg", {})
        self._mean = tuple(float(v) for v in pc.get("mean", (0.485, 0.456, 0.406)))
        self._std = tuple(float(v) for v in pc.get("std", (0.229, 0.224, 0.225)))
        input_size = pc.get("input_size", [3, 224, 224])
        self._img_size = int(input_size[-1])
        self._interp = str(pc.get("interpolation", "bilinear"))
        num_features = cfg.get("num_features")
        if num_features is not None and int(num_features) != self.embed_dim:
            raise ValueError(
                f"UNI dim drift: config num_features={num_features} != "
                f"FOUNDATION_DIMS['{name}']={self.embed_dim}; reconcile [BOUND-F7] before encoding."
            )

    def _read_config(self) -> dict:
        import json

        if not self._config_path.exists():
            return {}
        return json.loads(self._config_path.read_text(encoding="utf-8"))

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        import timm
        import torch

        if not self._weights_path.exists():
            raise FileNotFoundError(
                f"UNI2-h weights not found: {self._weights_path} — the frozen checkpoint must be on "
                "disk (data/pathology/uni/pytorch_model.bin)."
            )
        timm_kwargs = dict(
            img_size=self._img_size,
            patch_size=14,
            depth=24,
            num_heads=24,
            init_values=1e-5,
            embed_dim=self.embed_dim,
            mlp_ratio=2.66667 * 2,
            num_classes=0,
            no_embed_class=True,
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=torch.nn.SiLU,
            reg_tokens=8,
            dynamic_img_size=True,
        )
        model = timm.create_model(self._TIMM_MODEL, pretrained=False, **timm_kwargs)
        state = torch.load(self._weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)  
        model.eval()
        model.to(self._device)
        self._model = model
        return model

    def encode(self, tiles: np.ndarray) -> np.ndarray:
        import torch

        arr = np.asarray(tiles)
        if arr.ndim == 3:  
            arr = arr[None, ...]
        if arr.ndim != 4:
            raise ValueError(f"expected tiles [N, H, W, C], got shape {arr.shape}")
        if arr.shape[-1] not in (1, 3):
            raise ValueError(
                f"expected channels-last RGB tiles [N, H, W, C]; got last dim {arr.shape[-1]}"
            )

        model = self._ensure_model()
        x = torch.from_numpy(np.ascontiguousarray(arr)).to(torch.float32)
        x = x.permute(0, 3, 1, 2)  
        if x.shape[1] == 1:  
            x = x.repeat(1, 3, 1, 1)
        if float(x.max()) > 1.5:  
            x = x / 255.0
        if x.shape[-2:] != (self._img_size, self._img_size):
            resize_kw: dict = {"size": (self._img_size, self._img_size), "mode": self._interp}
            if self._interp in ("bilinear", "bicubic"):
                resize_kw.update(align_corners=False, antialias=True)
            x = torch.nn.functional.interpolate(x, **resize_kw)
        mean = torch.tensor(self._mean, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(self._std, dtype=torch.float32).view(1, 3, 1, 1)
        x = (x - mean) / std

        outs = []
        with torch.no_grad():
            for i in range(0, x.shape[0], self._batch_size):
                chunk = x[i : i + self._batch_size].to(self._device)
                emb = model(chunk)
                outs.append(emb.detach().to("cpu", torch.float32))
        return torch.cat(outs, dim=0).numpy()


def load_tile_encoder(name: str = "uni", synthetic: bool = True, **kw) -> TileEncoder:
    if synthetic:
        return SyntheticTileEncoder(name=name, **kw)
    from pinksight.trackb.gate import assert_gate_open

    assert_gate_open()  
    if name == "uni":
        return UNITileEncoder(name=name, **kw)
    raise NotImplementedError(
        f"real frozen-weight loading is only vendored for 'uni' (UNI2-h); {name!r} not yet wired"
    )
