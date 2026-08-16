
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

MASK_SUBDIR = "processed_masks"  
SLICE_HW = 64  
TRAIN_SLICES_PER_PATIENT = 8  
POST_CONTRAST_CHANNELS = (1, 2, 3)  

_LOGGED_MASK_FALLBACK: set[str] = set()  


def _minmax_4phase(sl4: np.ndarray) -> np.ndarray:
    lo = float(sl4.min())
    hi = float(sl4.max())
    scaled = (sl4 - lo) / (hi - lo + 1e-6)
    return np.ascontiguousarray(scaled[list(POST_CONTRAST_CHANNELS)], dtype=np.float32)  


def _resize_hw(x: np.ndarray, hw: int = SLICE_HW) -> np.ndarray:
    from monai.transforms import Resize

    resize = Resize(spatial_size=(hw, hw), mode="bilinear", align_corners=False)
    return np.asarray(resize(np.ascontiguousarray(x, dtype=np.float32)), dtype=np.float32)


def _load_mask(pid: str, mask_dir: Path, spatial: tuple[int, int, int]) -> np.ndarray | None:
    mp = mask_dir / f"{pid}.npy"
    if not mp.exists():
        return None
    m = np.load(mp)
    if m.ndim == 4 and m.shape[0] == 1:
        m = m[0]
    if m.ndim != 3 or m.shape != tuple(spatial):
        return None
    return m.astype(bool)


def _center_slice(pid: str, mask: np.ndarray | None, n_slices: int) -> int:
    from ..models.h0_localizer import largest_cc

    if mask is not None:
        m = largest_cc(mask)  
        per_slice = m.reshape(-1, m.shape[-1]).sum(axis=0)
        if per_slice.sum() > 0:
            slice_axis = np.arange(m.shape[-1])
            centroid = float((slice_axis * per_slice).sum() / per_slice.sum())
            return int(round(centroid))
    if pid not in _LOGGED_MASK_FALLBACK:
        _LOGGED_MASK_FALLBACK.add(pid)
        import warnings

        warnings.warn(
            f"no usable predicted mask for {pid} -> [1.6] fallback: tumour-centre slice = crop "
            "mid-slice (LOCK-2: never a ground-truth box). Disclosed once per pid.",
            stacklevel=2,
        )
    return n_slices // 2


def patient_slice_plan(pid: str, mask: np.ndarray | None, n_slices: int) -> tuple[list[int], int]:
    center = _center_slice(pid, mask, n_slices)
    half = TRAIN_SLICES_PER_PATIENT // 2
    lo = max(0, center - half)
    hi = min(n_slices, lo + TRAIN_SLICES_PER_PATIENT)
    lo = max(0, hi - TRAIN_SLICES_PER_PATIENT)  
    train_idx = list(range(lo, hi))
    test_idx = min(center + 1, n_slices - 1)  
    return train_idx, test_idx


class SliceGradeDataset(Dataset):

    def __init__(
        self,
        items: list[tuple[str, int]],
        proc_dir: Path = Path("data/processed"),
        mask_dir: Path | None = None,
        split: str = "train",
        augment: bool = False,
    ) -> None:
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")
        self.proc_dir = Path(proc_dir)
        self.mask_dir = Path(mask_dir) if mask_dir is not None else self.proc_dir.parent / MASK_SUBDIR
        self.split = split
        self.augment = augment and split == "train"  
        self.samples: list[tuple[str, int, int, str]] = []
        for pid, label in items:
            vol_shape = self._vol_shape(pid)
            n_slices = vol_shape[-1]
            mask = _load_mask(pid, self.mask_dir, (vol_shape[1], vol_shape[2], vol_shape[3]))
            train_idx, test_idx = patient_slice_plan(pid, mask, n_slices)
            if split == "test":
                self.samples.append((pid, label, test_idx, "none"))
            else:
                self.samples.extend(self._augmented_train_samples(pid, label, train_idx))

    def _vol_shape(self, pid: str) -> tuple[int, int, int, int]:
        arr = np.load(self.proc_dir / f"{pid}.npy", mmap_mode="r")
        return tuple(arr.shape)  

    def _augmented_train_samples(
        self, pid: str, label: int, train_idx: list[int]
    ) -> list[tuple[str, int, int, str]]:
        if not self.augment:
            return [(pid, label, s, "none") for s in train_idx]
        if not train_idx:
            return []
        out: list[tuple[str, int, int, str]] = []
        plan = ["h"] * 4 + ["v"] * 4 + ["none"] * 5  
        for k, op in enumerate(plan):
            s = train_idx[k % len(train_idx)]  
            out.append((pid, label, s, op))
        return out

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        pid, label, s, flip = self.samples[i]
        vol = np.load(self.proc_dir / f"{pid}.npy")  
        c = vol.shape[0]
        phase_idx = list(range(min(4, c)))
        sl4 = vol[phase_idx, :, :, s]  
        if sl4.shape[0] < 4:  
            pad = np.repeat(sl4[-1:], 4 - sl4.shape[0], axis=0)
            sl4 = np.concatenate([sl4, pad], axis=0)
        x = _minmax_4phase(np.ascontiguousarray(sl4, dtype=np.float32))  
        x = _resize_hw(x, SLICE_HW)  
        if flip == "h":
            x = np.ascontiguousarray(x[:, :, ::-1])
        elif flip == "v":
            x = np.ascontiguousarray(x[:, ::-1, :])
        xt = torch.as_tensor(x, dtype=torch.float32)
        yt = torch.tensor(float(label), dtype=torch.float32)
        return xt, yt, pid
