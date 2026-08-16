
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage

from ..data.annotation_boxes import Box

RIM_MM_DEFAULT = 7  


class WeightsNotProvisioned(RuntimeError):
    pass


def dilate_rim(mask: np.ndarray, rim_mm: int = RIM_MM_DEFAULT) -> np.ndarray:
    if rim_mm < 0:
        raise ValueError(f"rim_mm must be >= 0, got {rim_mm}")
    m = np.asarray(mask).astype(bool)
    if rim_mm == 0:
        return m
    return ndimage.binary_dilation(m, iterations=rim_mm)


def mask_to_box(mask: np.ndarray, patient: str = "") -> Box:
    m = np.asarray(mask).astype(bool)
    if m.ndim != 3:
        raise ValueError(f"mask must be 3D (row, col, slice), got shape {m.shape}")
    if not m.any():
        raise ValueError("empty mask: no tumour voxels to bound")

    def extent(axes: tuple[int, int]) -> tuple[int, int]:
        idx = np.flatnonzero(m.any(axis=axes))
        return int(idx[0]), int(idx[-1] + 1)  

    return Box(patient, extent((1, 2)), extent((0, 2)), extent((0, 1)))


def largest_cc(mask: np.ndarray) -> np.ndarray:
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
    m = largest_cc(mask) if largest_only else mask
    return mask_to_box(dilate_rim(m, rim_mm), patient)


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    a = np.asarray(pred).astype(bool)
    b = np.asarray(gt).astype(bool)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    denom = int(a.sum() + b.sum())
    if denom == 0:
        return 1.0  
    return 2.0 * int(np.logical_and(a, b).sum()) / denom


_PREDICTORS: dict[str, object] = {}  


def _get_predictor(weights_dir: str | Path):
    key = str(Path(weights_dir).resolve())
    if key not in _PREDICTORS:
        import torch
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        p = nnUNetPredictor(perform_everything_on_device=(device == "cuda"), use_mirroring=False,
                            device=torch.device(device), allow_tqdm=False, verbose=False)
        p.initialize_from_trained_model_folder(key, use_folds=(0,), checkpoint_name="checkpoint_final.pth")
        _PREDICTORS[key] = p
    return _PREDICTORS[key]


def predict_mask(
    volume: np.ndarray, weights_dir: str | Path | None = None, spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> np.ndarray:
    if weights_dir is None:
        raise WeightsNotProvisioned(
            "predict_mask needs an explicit weights_dir (MAMA-MIA nnU-Net results folder with "
            "plans.json + fold_*/checkpoint_final.pth). Refusing to fabricate a mask (LOCK-2)."
        )
    arr = np.ascontiguousarray(np.asarray(volume, dtype=np.float32))
    if arr.ndim != 3:
        raise ValueError(f"volume must be 3D (row, col, slice), got shape {arr.shape}")
    predictor = _get_predictor(weights_dir)
    seg = predictor.predict_single_npy_array(arr[None], {"spacing": list(spacing)})  
    return np.asarray(seg).astype(bool)
