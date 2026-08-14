"""P06: Dataset over cached P03 ROI volumes (`data/processed/{pid}.npy`) for deep-net training.

Each on-disk volume is `(C, row, col, slice)` float32 with VARIABLE C (pre-contrast + a variable
number of post-contrast phases) and variable spatial size (already lesion-ROI cropped + 7mm rim,
1mm-iso). For batching we (1) pick a deterministic channel subset and (2) resize to a fixed grid.

ponytail: no mask handling — the volume is already the ROI crop (`preprocess.py`), the CNN eats the
whole crop. Spatial resample is `monai.transforms.Resize` (installed), not a hand-rolled interpolate.
Channel idiom mirrors the radiomics baseline (`scripts/g1_radiomics_baseline.py:67`) so first-post is
byte-for-byte the same phase the G1 number used.
"""

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
# "none"=loose Duke box; "lesion"=tight predicted-mask box +7mm rim (S29 NULL); "box"=radiologist
# bounding box + a LARGE peritumoral margin ([HEAD2-GRADE-PIVOT] DeepRadGrade analog, the untested lever)
CROP_MODES = ("none", "lesion", "box")
MASK_SUBDIR = "processed_masks"  # data/processed_masks/{pid}.npy — H0 PREDICTED mask on the crop grid

_LOGGED_PHASE_SHORTFALL: set[str] = set()  # dedup guard so the diff-channel fallback warns ONCE, not per-item


def _znorm(x: np.ndarray) -> np.ndarray:
    """Per-channel z-score (eps-guarded). Deterministic — applied to EVERY fold (train/val/test), so
    it's preprocessing, not a tunable knob. MedicalNet transfer expects normalized input; raw MRI
    intensities (~1-80 here) train poorly un-normalized. ponytail: per-channel mean/std, no clip."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    for c in range(x.shape[0]):
        m, s = float(x[c].mean()), float(x[c].std())
        x[c] = (x[c] - m) / (s + 1e-6)
    return x


def _aug_pipeline() -> Compose:
    """Train-ONLY 3D augmentation (pre-registered G2 recipe): random flips on all axes, small
    rotations, gentle intensity scale/shift. Regularizes the weak small-N imaging encoder. Reproducible
    via monai.utils.set_determinism(seed) in cross_val_imaging. ponytail: standard MONAI rand ops."""
    return Compose([
        RandFlip(prob=0.5, spatial_axis=0),
        RandFlip(prob=0.5, spatial_axis=1),
        RandFlip(prob=0.5, spatial_axis=2),
        RandRotate(range_x=0.26, range_y=0.26, range_z=0.26, prob=0.3, mode="bilinear"),  # ~15deg
        RandScaleIntensity(factors=0.1, prob=0.5),
        RandShiftIntensity(offsets=0.1, prob=0.5),
    ])


def select_channels(vol: np.ndarray, policy: str) -> np.ndarray:
    """Deterministic channel subset so variable-C volumes batch. vol is (C, D, H, W)."""
    c = vol.shape[0]
    if policy == "first_post":
        return vol[1:2] if c > 1 else vol[0:1]  # first post-contrast (matches g1 ROI choice)
    if policy == "pre_post":
        return vol[[0, 1 if c > 1 else 0]]  # pre + first-post (enhancement contrast)
    if policy == "fixed4":
        if c >= 4:
            return vol[:4]
        pad = np.repeat(vol[-1:], 4 - c, axis=0)  # pad by repeating last phase
        return np.concatenate([vol, pad], axis=0)
    # [G2-SUBTRACTION-REOPEN] enhancement-difference (post - pre) channels — the standard DCE
    # representation for enhancement PATTERN (a snapshot, NOT growth-rate kinetics). Computed on the
    # RAW volume here; _znorm (in NpyVolumeDataset.__getitem__) runs AFTER, so the DIFFERENCE image is
    # normalized, not the raw intensities. pre=vol[0], first_post=vol[1], second_post=vol[2].
    if policy == "subtraction":
        if c < 2:  # no post-contrast phase at all — degenerate all-zero diff (never expected in-cohort)
            _warn_phase_shortfall("subtraction", c, need=2)
        pre = vol[0]
        first_post = vol[1] if c > 1 else vol[0]
        second_post = vol[2] if c > 2 else first_post  # <3 phases: fall back to (first_post - pre)
        return np.stack([first_post - pre, second_post - pre], axis=0)
    if policy == "kinetic":
        if c < 3:  # missing a second post-contrast phase — pad the 2nd diff with the 1st diff
            _warn_phase_shortfall("kinetic", c, need=3)
        pre = vol[0]
        first_post = vol[1] if c > 1 else vol[0]
        second_post = vol[2] if c > 2 else first_post
        return np.stack([pre, first_post, first_post - pre, second_post - pre], axis=0)
    raise ValueError(f"channels must be one of {CHANNEL_POLICIES}, got {policy!r}")


def _warn_phase_shortfall(policy: str, c: int, need: int) -> None:
    """One-shot logged fallback when a volume has fewer phases than the diff-channel policy wants.
    Deduped per policy so it discloses the degradation once rather than flooding the per-item loop."""
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
    """How many channels `select_channels` yields — so the encoder's in_channels matches."""
    return {"first_post": 1, "pre_post": 2, "fixed4": 4, "subtraction": 2, "kinetic": 4}[policy]


class NpyVolumeDataset(Dataset):
    """(patient_id, label) -> (x:(C,*spatial) float32, y scalar float32, pid str)."""

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
        self._aug = _aug_pipeline() if augment else None  # train folds only; never val/test
        # [G2-LESION-CROP] geometry lever: crop_mode="none" (default) is byte-identical to the loose-box
        # baseline (channel-subset -> resize -> znorm). crop_mode="lesion" derives a TIGHT box from the
        # H0 PREDICTED mask (data/processed_masks/{pid}.npy) -> resample to a fixed cube. LOCK-2: the
        # box source is the PREDICTED mask only; a missing/degenerate mask hits the [1.6] Duke-box
        # fallback (whole cached crop), never a ground-truth mask, never an empty crop.
        self.crop_mode = crop_mode
        self.crop_size = int(crop_size)
        self.box_margin_mm = int(box_margin_mm)  # [HEAD2-GRADE-PIVOT] radiologist-box margin (box mode only)
        self.mask_dir = Path(mask_dir) if mask_dir is not None else self.proc_dir.parent / MASK_SUBDIR

    def __len__(self) -> int:
        return len(self.items)

    def _load_predicted_mask(self, pid: str, spatial: tuple[int, int, int]) -> np.ndarray | None:
        """Load the H0 PREDICTED mask for `pid` on the crop grid, or None ([1.6] fallback trigger).

        LOCK-2: this ONLY ever reads a predicted-mask file (data/processed_masks/{pid}.npy) — there is
        no code path here that reads a ground-truth segmentation. Returns None (-> [1.6] Duke-box
        fallback in `lesion_crop`) when the mask is:
          * absent (no file) — a run without provisioned H0 masks still trains oracle-free; or
          * UNRELIABLE — a mask whose grid doesn't match this patient's cached crop. Per the pre-reg
            [1.6] "a poor predicted mask" mitigation, an ill-fitting mask is treated as unreliable and
            falls back to the whole cached crop rather than CRASHING the run (1/748 Duke masks, e.g.
            Breast_MRI_042, don't share the crop grid). `derive_lesion_box` keeps its strict raise for
            direct callers; the dataset degrades gracefully. A (1, r, c, s) mask is squeezed to (r,c,s).
        """
        mp = self.mask_dir / f"{pid}.npy"
        if not mp.exists():
            return None
        m = np.load(mp)
        if m.ndim == 4 and m.shape[0] == 1:  # tolerate a leading singleton channel
            m = m[0]
        if m.ndim != 3 or m.shape != tuple(spatial):
            # unreliable mask (wrong grid) -> [1.6] fallback, not a crash. Disclosed once per dataset.
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
        vol = np.load(self.proc_dir / f"{pid}.npy")  # (C, D, H, W) float32
        vol = select_channels(vol, self.channels)
        if self.crop_mode in ("lesion", "box"):
            # PREDICTED-mask box -> fixed cube (already resized inside lesion_crop/box_crop). LOCK-2 +
            # [1.6] fallback live there; the mask is single-channel over the shared spatial grid so
            # channel-subset order is irrelevant. `spatial` = the cached crop's (r,c,s), used to reject
            # an ill-fitting mask into the [1.6] fallback instead of crashing. The two modes share the
            # code path and differ ONLY in the margin: "lesion" uses the tight 7mm rim (S29 NULL);
            # "box" uses the LARGE radiologist margin (`box_margin_mm`, [HEAD2-GRADE-PIVOT] untested lever).
            spatial = (vol.shape[1], vol.shape[2], vol.shape[3])
            mask = self._load_predicted_mask(pid, spatial)
            vol_f = np.ascontiguousarray(vol, dtype=np.float32)
            if self.crop_mode == "box":
                x = box_crop(vol_f, mask, box_margin_mm=self.box_margin_mm,
                             out_size=self.crop_size)  # (C, crop_size^3)
            else:
                x = lesion_crop(vol_f, mask, out_size=self.crop_size)  # (C, crop_size^3)
        else:
            x = self._resize(np.ascontiguousarray(vol, dtype=np.float32))  # (C, *spatial)
        x = _znorm(np.asarray(x, dtype=np.float32))                    # always (preprocessing)
        if self._aug is not None:
            x = np.asarray(self._aug(x), dtype=np.float32)            # train-only regularizer
        x = torch.as_tensor(x, dtype=torch.float32)
        y = torch.tensor(float(label), dtype=torch.float32)
        return x, y, pid
