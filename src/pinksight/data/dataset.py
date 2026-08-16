
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from monai.transforms import (
    Compose,
    RandFlip,
    RandRotate,
    RandScaleIntensity,
    RandShiftIntensity,
    Resize,
)
from torch.utils.data import Dataset

from .lesion_crop import BOX_MARGIN_MM_DEFAULT, CROP_SIZE_DEFAULT, box_crop, lesion_crop

CHANNEL_POLICIES = ("first_post", "pre_post", "fixed4", "subtraction", "kinetic")
CROP_MODES = ("none", "lesion", "box")
MASK_SUBDIR = "processed_masks"  

_LOGGED_PHASE_SHORTFALL: set[str] = set()  


def _znorm(x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32)
    for c in range(x.shape[0]):
        m, s = float(x[c].mean()), float(x[c].std())
        x[c] = (x[c] - m) / (s + 1e-6)
    return x


def _aug_pipeline() -> Compose:
    return Compose([
        RandFlip(prob=0.5, spatial_axis=0),
        RandFlip(prob=0.5, spatial_axis=1),
        RandFlip(prob=0.5, spatial_axis=2),
        RandRotate(range_x=0.26, range_y=0.26, range_z=0.26, prob=0.3, mode="bilinear"),  
        RandScaleIntensity(factors=0.1, prob=0.5),
        RandShiftIntensity(offsets=0.1, prob=0.5),
    ])


def select_channels(vol: np.ndarray, policy: str) -> np.ndarray:
    c = vol.shape[0]
    if policy == "first_post":
        return vol[1:2] if c > 1 else vol[0:1]  
    if policy == "pre_post":
        return vol[[0, 1 if c > 1 else 0]]  
    if policy == "fixed4":
        if c >= 4:
            return vol[:4]
        pad = np.repeat(vol[-1:], 4 - c, axis=0)  
        return np.concatenate([vol, pad], axis=0)
    if policy == "subtraction":
        if c < 2:  
            _warn_phase_shortfall("subtraction", c, need=2)
        pre = vol[0]
        first_post = vol[1] if c > 1 else vol[0]
        second_post = vol[2] if c > 2 else first_post  
        return np.stack([first_post - pre, second_post - pre], axis=0)
    if policy == "kinetic":
        if c < 3:  
            _warn_phase_shortfall("kinetic", c, need=3)
        pre = vol[0]
        first_post = vol[1] if c > 1 else vol[0]
        second_post = vol[2] if c > 2 else first_post
        return np.stack([pre, first_post, first_post - pre, second_post - pre], axis=0)
    raise ValueError(f"channels must be one of {CHANNEL_POLICIES}, got {policy!r}")


def _warn_phase_shortfall(policy: str, c: int, need: int) -> None:
    if policy in _LOGGED_PHASE_SHORTFALL:
        return
    _LOGGED_PHASE_SHORTFALL.add(policy)
    import warnings
    warnings.warn(
        f"channels={policy!r} wants >= {need} phases but a volume has only {c}; padding the missing "
        "enhancement-difference channel with (first_post - pre). Disclosed once per policy.",
        stacklevel=3,
    )


def n_channels(policy: str) -> int:
    return {"first_post": 1, "pre_post": 2, "fixed4": 4, "subtraction": 2, "kinetic": 4}[policy]


class NpyVolumeDataset(Dataset):

    def __init__(
        self,
        items: list[tuple[str, int]],
        proc_dir: Path = Path("data/processed"),
        channels: str = "first_post",
        spatial_size: tuple[int, int, int] = (96, 96, 96),
        augment: bool = False,
        crop_mode: str = "none",
        crop_size: int = CROP_SIZE_DEFAULT,
        mask_dir: Path | None = None,
        box_margin_mm: int = BOX_MARGIN_MM_DEFAULT,
    ) -> None:
        if channels not in CHANNEL_POLICIES:
            raise ValueError(f"channels must be one of {CHANNEL_POLICIES}, got {channels!r}")
        if crop_mode not in CROP_MODES:
            raise ValueError(f"crop_mode must be one of {CROP_MODES}, got {crop_mode!r}")
        self.items = list(items)
        self.proc_dir = Path(proc_dir)
        self.channels = channels
        self._resize = Resize(spatial_size=spatial_size, mode="trilinear", align_corners=False)
        self._aug = _aug_pipeline() if augment else None  
        self.crop_mode = crop_mode
        self.crop_size = int(crop_size)
        self.box_margin_mm = int(box_margin_mm)  
        self.mask_dir = Path(mask_dir) if mask_dir is not None else self.proc_dir.parent / MASK_SUBDIR

    def __len__(self) -> int:
        return len(self.items)

    def _load_predicted_mask(self, pid: str, spatial: tuple[int, int, int]) -> np.ndarray | None:
        mp = self.mask_dir / f"{pid}.npy"
        if not mp.exists():
            return None
        m = np.load(mp)
        if m.ndim == 4 and m.shape[0] == 1:  
            m = m[0]
        if m.ndim != 3 or m.shape != tuple(spatial):
            if not getattr(self, "_warned_mask_grid", False):
                import warnings
                warnings.warn(
                    f"predicted mask for {pid} has shape {m.shape} != crop grid {tuple(spatial)}; "
                    "treating as unreliable -> [1.6] Duke-box fallback (not a crash). "
                    "(further such masks in this dataset fall back silently).",
                    stacklevel=2,
                )
                self._warned_mask_grid = True
            return None
        return m

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        pid, label = self.items[i]
        vol = np.load(self.proc_dir / f"{pid}.npy")  
        vol = select_channels(vol, self.channels)
        if self.crop_mode in ("lesion", "box"):
            spatial = (vol.shape[1], vol.shape[2], vol.shape[3])
            mask = self._load_predicted_mask(pid, spatial)
            vol_f = np.ascontiguousarray(vol, dtype=np.float32)
            if self.crop_mode == "box":
                x = box_crop(vol_f, mask, box_margin_mm=self.box_margin_mm,
                             out_size=self.crop_size)  
            else:
                x = lesion_crop(vol_f, mask, out_size=self.crop_size)  
        else:
            x = self._resize(np.ascontiguousarray(vol, dtype=np.float32))  
        x = _znorm(np.asarray(x, dtype=np.float32))                    
        if self._aug is not None:
            x = np.asarray(self._aug(x), dtype=np.float32)            
        x = torch.as_tensor(x, dtype=torch.float32)
        y = torch.tensor(float(label), dtype=torch.float32)
        return x, y, pid
