"""[HEAD2-GRADE-2D-SLICE] 2D-per-slice dataset — the DeepRadGrade recipe (Eur Radiol 2025, PMC13086883).

Every prior imaging->grade arm fed a 3D encoder (MedicalNet 3D-ResNet-18) a whole lesion CUBE and
nulled. DeepRadGrade explicitly abandoned 3D/ResNet ("overfit on this exact Duke cohort") and won
(test AUC 0.82) with a tiny from-scratch 2D CNN reading ONE axial slice at a time. This module
produces that 2D-per-slice representation and is the ONLY axis this probe changes.

Pipeline (pre-reg `reports/G2_imaging/HEAD2-GRADE-2D-SLICE_PREREG_13-07-26.md`):
  * cached volume `data/processed/{pid}.npy` is (C, row, col, slice) float32 — axis 3 == axial slice.
  * the tumour CENTER slice is the slice index with the most H0-PREDICTED tumour voxels (LOCK-2:
    predicted masks only, `data/processed_masks/{pid}.npy`; [1.6] fallback = the crop's mid-slice when
    a mask is absent/degenerate/ill-fitting, so we never read a ground-truth box at test).
  * channels = FIRST 3 POST-CONTRAST phases -> (3, row, col); min-max normalised over the 4-phase
    (pre + first-3-post) intensity range; in-plane resized to (3, 64, 64).
  * TRAIN: up to 8 slices/patient around the predicted centre. TEST/OOF: the SINGLE axial slice
    immediately SUPERIOR to the tumour centre (their protocol) -> exactly ONE prediction per patient.

NEW leakage guard (this recipe introduces it): slices are inputs but the split is by PATIENT. A
patient's slices must NEVER appear in both train and val of any fold. The dataset tags every emitted
sample with its `pid` so the CV harness can assert slice-level patient-disjointness.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

MASK_SUBDIR = "processed_masks"  # data/processed_masks/{pid}.npy — H0 PREDICTED mask on the crop grid
SLICE_HW = 64  # DeepRadGrade in-plane 64x64
TRAIN_SLICES_PER_PATIENT = 8  # "train ~8/patient around the tumour centroid"
POST_CONTRAST_CHANNELS = (1, 2, 3)  # first 3 post-contrast phases (vol[0] is pre-contrast)

_LOGGED_MASK_FALLBACK: set[str] = set()  # dedup: warn ONCE per pid when the [1.6] fallback fires


def _minmax_4phase(sl4: np.ndarray) -> np.ndarray:
    """Min-max normalise a (4, H, W) pre+3post slice stack over the whole 4-phase range, then keep the
    3 post-contrast channels. Pre-reg: "min-max norm over the 4-phase (pre+3post) range." One shared
    (min,max) across all 4 phases preserves inter-phase enhancement contrast (a per-channel norm would
    erase it). eps-guarded; deterministic (applied to every fold) so it is preprocessing, not a knob."""
    lo = float(sl4.min())
    hi = float(sl4.max())
    scaled = (sl4 - lo) / (hi - lo + 1e-6)
    return np.ascontiguousarray(scaled[list(POST_CONTRAST_CHANNELS)], dtype=np.float32)  # (3, H, W)


def _resize_hw(x: np.ndarray, hw: int = SLICE_HW) -> np.ndarray:
    """Resize a (3, H, W) slice to (3, hw, hw) with MONAI's Resize (same transform the 3D path uses)."""
    from monai.transforms import Resize

    resize = Resize(spatial_size=(hw, hw), mode="bilinear", align_corners=False)
    return np.asarray(resize(np.ascontiguousarray(x, dtype=np.float32)), dtype=np.float32)


def _load_mask(pid: str, mask_dir: Path, spatial: tuple[int, int, int]) -> np.ndarray | None:
    """H0 PREDICTED mask (row, col, slice) for `pid` on the crop grid, or None ([1.6] fallback).

    LOCK-2: reads ONLY data/processed_masks/{pid}.npy — no ground-truth path exists here. A missing /
    wrong-grid mask returns None so the caller falls back to the crop mid-slice (never a GT box)."""
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
    """Index (along the slice axis) of the tumour CENTRE slice = the voxel-count-weighted CENTROID of
    the predicted tumour along the slice axis (robust to ties, matches the pre-reg 'tumour centroid'
    wording). [1.6] fallback: mid-volume slice when the mask is absent/empty/degenerate."""
    from ..models.h0_localizer import largest_cc

    if mask is not None:
        m = largest_cc(mask)  # drop nnU-Net FP specks; keep the dominant blob (same as lesion_crop)
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
    """(train_slice_indices, test_slice_index) for one patient given its slice count.

    center = tumour-centre slice (most predicted voxels). TRAIN = up to 8 slices centred on it
    (clamped into range). TEST/OOF = the slice immediately SUPERIOR to the centre (center+1, clamped)
    per the DeepRadGrade protocol -> exactly one held-out prediction per patient.
    """
    center = _center_slice(pid, mask, n_slices)
    half = TRAIN_SLICES_PER_PATIENT // 2
    lo = max(0, center - half)
    hi = min(n_slices, lo + TRAIN_SLICES_PER_PATIENT)
    lo = max(0, hi - TRAIN_SLICES_PER_PATIENT)  # re-clamp low end when near the top edge
    train_idx = list(range(lo, hi))
    test_idx = min(center + 1, n_slices - 1)  # immediately superior; clamp at the top slice
    return train_idx, test_idx


class SliceGradeDataset(Dataset):
    """(pid, label) list -> flattened per-slice samples. Each __getitem__ yields (x:(3,64,64), y, pid).

    `split="train"`: up to 8 slices/patient around the predicted tumour centre, optionally flip-augmented
    (DeepRadGrade: per slice-set 4 h-flip + 4 v-flip + 5 unchanged). `split="test"`: exactly ONE slice
    per patient (the supra-central slice) -> one OOF prediction per patient.

    `pid` rides on every sample so the CV harness can assert no patient's slices cross a fold boundary
    (the NEW slice-level patient-disjoint guard this recipe requires).
    """

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
        self.augment = augment and split == "train"  # augmentation is train-only, never test/OOF
        # Build the flat sample table up front: (pid, label, slice_index, flip_op).
        # flip_op in {"none","h","v"}; augmentation expands a train slice-set to 4h+4v+5none *per set*
        # by tagging slices with flip ops (DeepRadGrade's per-set 4/4/5). Deterministic -> reproducible.
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
        # np.load(mmap) reads only the header -> shape without pulling the whole volume into RAM.
        arr = np.load(self.proc_dir / f"{pid}.npy", mmap_mode="r")
        return tuple(arr.shape)  # (C, row, col, slice)

    def _augmented_train_samples(
        self, pid: str, label: int, train_idx: list[int]
    ) -> list[tuple[str, int, int, str]]:
        """Per DeepRadGrade: per patient's slice-set, 4 h-flip + 4 v-flip + 5 unchanged. With no augment,
        every train slice is emitted once unchanged. The flip ops are assigned deterministically by
        cycling through the slice set so the expansion is reproducible (no per-item RNG)."""
        if not self.augment:
            return [(pid, label, s, "none") for s in train_idx]
        if not train_idx:
            return []
        out: list[tuple[str, int, int, str]] = []
        plan = ["h"] * 4 + ["v"] * 4 + ["none"] * 5  # 13 ops per patient slice-set (DeepRadGrade 4/4/5)
        for k, op in enumerate(plan):
            s = train_idx[k % len(train_idx)]  # cycle the (<=8) slices to fill 13 augmented samples
            out.append((pid, label, s, op))
        return out

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        pid, label, s, flip = self.samples[i]
        vol = np.load(self.proc_dir / f"{pid}.npy")  # (C, row, col, slice) float32
        # take 4 phases (pre + first 3 post) at this axial slice -> (4, row, col)
        c = vol.shape[0]
        phase_idx = list(range(min(4, c)))
        sl4 = vol[phase_idx, :, :, s]  # (<=4, row, col)
        if sl4.shape[0] < 4:  # pad missing post-contrast phases by repeating the last (rare; <4 phases)
            pad = np.repeat(sl4[-1:], 4 - sl4.shape[0], axis=0)
            sl4 = np.concatenate([sl4, pad], axis=0)
        x = _minmax_4phase(np.ascontiguousarray(sl4, dtype=np.float32))  # (3, row, col), first-3-post
        x = _resize_hw(x, SLICE_HW)  # (3, 64, 64)
        if flip == "h":
            x = np.ascontiguousarray(x[:, :, ::-1])
        elif flip == "v":
            x = np.ascontiguousarray(x[:, ::-1, :])
        xt = torch.as_tensor(x, dtype=torch.float32)
        yt = torch.tensor(float(label), dtype=torch.float32)
        return xt, yt, pid
