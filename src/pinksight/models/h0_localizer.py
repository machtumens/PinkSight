"""P04 sub-step (1): H0 localiser output -> encoder ROI (predicted mask -> ROI + peritumoral rim).

The H0 tumour localiser (MAMA-MIA nnU-Net, forward register O-4) emits a per-voxel tumour mask;
this module turns that mask into the lesion ROI the imaging encoder reads. Two jobs:

  * post-processing (pure numpy/scipy, runs now): dilate the mask into a 5-10mm peritumoral rim
    (LOCK-3) and reduce it to a `Box`, so an H0 mask plugs straight into `preprocess.crop()`;
  * inference (gated): run the nnU-Net predictor. The MAMA-MIA weights are NOT provisioned yet
    (O-4 + data blocker) and `nnunetv2` is not installed, so `predict_mask` raises a clear error
    rather than fabricating a mask.

LOCK-2: at test time the ROI is built from the PREDICTED mask, never a ground-truth mask. The
ground-truth box path stays in `annotation_boxes` and may only seed a separately-labelled "oracle
upper bound", never the headline.

Grid: the encoder ROI lives on P03's 1mm-isotropic grid, so 1 voxel == 1 mm and a rim of `rim_mm`
millimetres is `rim_mm` dilation iterations. Arrays are (row, col, slice)-ordered to match `Box`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage

from ..data.annotation_boxes import Box

RIM_MM_DEFAULT = 7  # mid of LOCK-3's 5-10mm peritumoral band; == voxels at 1mm iso


class WeightsNotProvisioned(RuntimeError):
    """H0 inference requested but the MAMA-MIA nnU-Net weights are absent (O-4 unresolved)."""


def dilate_rim(mask: np.ndarray, rim_mm: int = RIM_MM_DEFAULT) -> np.ndarray:
    """Grow a boolean tumour mask outward by `rim_mm` to include the peritumoral rim (LOCK-3).

    On the 1mm-iso grid one dilation iteration == 1 mm. Returns the dilated mask (lesion + rim).

    ponytail: face-connected iterations give a diamond, not a Euclidean ball — fine for an
    approximate peritumoral band; swap in a ball structuring element if rim shape ever matters.
    """
    if rim_mm < 0:
        raise ValueError(f"rim_mm must be >= 0, got {rim_mm}")
    m = np.asarray(mask).astype(bool)
    if rim_mm == 0:
        return m
    return ndimage.binary_dilation(m, iterations=rim_mm)


def mask_to_box(mask: np.ndarray, patient: str = "") -> Box:
    """Tight half-open `Box` around a boolean mask's nonzero extent (row, col, slice order).

    Lets a predicted H0 mask feed `preprocess.crop()` exactly like a Duke annotation box.
    """
    m = np.asarray(mask).astype(bool)
    if m.ndim != 3:
        raise ValueError(f"mask must be 3D (row, col, slice), got shape {m.shape}")
    if not m.any():
        raise ValueError("empty mask: no tumour voxels to bound")

    def extent(axes: tuple[int, int]) -> tuple[int, int]:
        idx = np.flatnonzero(m.any(axis=axes))
        return int(idx[0]), int(idx[-1] + 1)  # half-open, matches Box convention

    return Box(patient, extent((1, 2)), extent((0, 2)), extent((0, 1)))


def largest_cc(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component of a boolean mask.

    The MAMA-MIA nnU-Net sprays scattered false-positive specks across the breast (a
    raw mask's bounding box spans most of the volume); the true lesion is the dominant
    blob. Standard nnU-Net post-processing — reduces the mask to that blob so the ROI box
    is lesion-sized, not whole-volume. Empty mask returns unchanged.
    """
    from scipy import ndimage

    m = np.asarray(mask).astype(bool)
    lab, n = ndimage.label(m)
    if n <= 1:
        return m
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)


def predicted_roi_box(
    mask: np.ndarray, rim_mm: int = RIM_MM_DEFAULT, patient: str = "", largest_only: bool = True
) -> Box:
    """Predicted mask -> ROI box including the peritumoral rim — what the encoder pipeline calls.

    `largest_only` (default True) keeps just the dominant tumour blob before boxing — without
    it the nnU-Net's scattered FP specks blow the box up to the whole volume.
    """
    m = largest_cc(mask) if largest_only else mask
    return mask_to_box(dilate_rim(m, rim_mm), patient)


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice similarity coefficient between two boolean masks (H0 DSC-band metric)."""
    a = np.asarray(pred).astype(bool)
    b = np.asarray(gt).astype(bool)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    denom = int(a.sum() + b.sum())
    if denom == 0:
        return 1.0  # both empty -> identical by convention
    return 2.0 * int(np.logical_and(a, b).sum()) / denom


# MAMA-MIA nnU-Net is single-channel ("T1"), 1mm-iso, ZScore-normalised internally
# (dataset.json/plans.json). Our P03 grid is 1mm-iso, so we predict on a (row,col,slice)
# array with assumed unit spacing — isotropic spacing makes axis order irrelevant to
# nnU-Net's internal resample, and the mask returns on the same grid.
_PREDICTORS: dict[str, object] = {}  # ponytail: cache one predictor per weights_dir; init+load is slow


def _get_predictor(weights_dir: str | Path):
    """Lazily build + cache an nnUNetPredictor for a MAMA-MIA results folder (single fold)."""
    key = str(Path(weights_dir).resolve())
    if key not in _PREDICTORS:
        import torch
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # ponytail: use_mirroring=False — 8x test-time mirroring augmentation costs ~33s/patient
        # for ~5s without it, and we only need a lesion ROI box (largest_cc + rim), not a
        # pixel-perfect boundary. Turn it back on if mask QC ever needs the extra fidelity.
        p = nnUNetPredictor(perform_everything_on_device=(device == "cuda"), use_mirroring=False,
                            device=torch.device(device), allow_tqdm=False, verbose=False)
        # ponytail: fold_0 only — 1 net, not a 5-fold ensemble. Ensemble (use_folds=(0,1,2,3,4))
        # only if mask QC is poor; 5x the inference cost for a baseline ROI is not worth it.
        p.initialize_from_trained_model_folder(key, use_folds=(0,), checkpoint_name="checkpoint_final.pth")
        _PREDICTORS[key] = p
    return _PREDICTORS[key]


def predict_mask(
    volume: np.ndarray, weights_dir: str | Path | None = None, spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> np.ndarray:
    """Run the H0 nnU-Net localiser to get a PREDICTED boolean tumour mask (LOCK-2 test-time source).

    `volume` is a single DCE phase as a (row, col, slice) array on the 1mm-iso P03 grid (feed the
    first post-contrast phase, pre-Nyul). Returns a boolean mask on the same grid.

    GATED on an explicit `weights_dir`: with no weights it raises rather than fabricating a mask —
    never substitute a ground-truth box at test time (LOCK-2). Point `weights_dir` at the nnU-Net
    results folder holding `plans.json` + `fold_*/checkpoint_final.pth`.
    """
    if weights_dir is None:
        raise WeightsNotProvisioned(
            "predict_mask needs an explicit weights_dir (MAMA-MIA nnU-Net results folder with "
            "plans.json + fold_*/checkpoint_final.pth). Refusing to fabricate a mask (LOCK-2)."
        )
    arr = np.ascontiguousarray(np.asarray(volume, dtype=np.float32))
    if arr.ndim != 3:
        raise ValueError(f"volume must be 3D (row, col, slice), got shape {arr.shape}")
    predictor = _get_predictor(weights_dir)
    seg = predictor.predict_single_npy_array(arr[None], {"spacing": list(spacing)})  # (1,r,c,s) in -> (r,c,s) out
    return np.asarray(seg).astype(bool)
