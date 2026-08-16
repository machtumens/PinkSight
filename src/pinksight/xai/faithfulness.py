
from __future__ import annotations

import numpy as np


def _binarize(saliency, q=0.9) -> np.ndarray:
    s = np.asarray(saliency, float)
    return s >= np.quantile(s, q)


def resample_mask_to(reference_mask, target_shape) -> np.ndarray:
    ref = np.asarray(reference_mask)
    target_shape = tuple(int(s) for s in target_shape)
    if ref.ndim != len(target_shape):
        raise ValueError(
            f"reference mask ndim {ref.ndim} != target ndim {len(target_shape)} "
            f"(mask shape {ref.shape}, target {target_shape}) — axis order mismatch, cannot resample."
        )
    if ref.shape == target_shape:
        return ref
    idx = np.ix_(*[
        np.clip((np.arange(t) * s // t), 0, s - 1)
        for s, t in zip(ref.shape, target_shape)
    ])
    out = ref[idx]
    if out.shape != target_shape:  
        raise ValueError(f"resample produced {out.shape}, expected {target_shape}")
    return out


def iou(saliency, reference_mask, q=0.9) -> float:
    pred = _binarize(saliency, q)
    ref = np.asarray(reference_mask).astype(bool)
    union = np.logical_or(pred, ref).sum()
    return float(np.logical_and(pred, ref).sum()) / float(union) if union else 0.0


def pointing_game(saliency, reference_mask) -> bool:
    peak = np.unravel_index(np.argmax(np.asarray(saliency)), np.asarray(saliency).shape)
    return bool(np.asarray(reference_mask).astype(bool)[peak])


def box_hit(saliency, box) -> bool:
    z, y, x = np.unravel_index(np.argmax(np.asarray(saliency)), np.asarray(saliency).shape)
    z0, z1, y0, y1, x0, x1 = box
    return bool(z0 <= z < z1 and y0 <= y < y1 and x0 <= x < x1)


_NPY_CROP_MARGIN_VOX = 7


def box_to_cam_mask(box, native_shape, cam_shape=(96, 96, 96)) -> np.ndarray:
    native_shape = tuple(int(s) for s in native_shape)
    if len(native_shape) != 3:
        raise ValueError(f"native_shape must be 3D (D,H,W), got {native_shape}")
    if getattr(box, "row", None) is None:  
        raise TypeError("box must be an annotation_boxes.Box with row/col/slice ranges")

    m = int(_NPY_CROP_MARGIN_VOX)
    mask_native = np.zeros(native_shape, dtype=bool)
    sl = []
    for dim in native_shape:
        lo, hi = (m, dim - m) if dim > 2 * m else (0, dim)
        sl.append(slice(lo, hi))
    mask_native[sl[0], sl[1], sl[2]] = True
    return resample_mask_to(mask_native, cam_shape).astype(bool)


def randomization_test(saliency_orig, saliency_random, q=0.9, drop_thresh=0.5) -> dict:
    roi = _binarize(saliency_orig, q)
    o = np.abs(np.asarray(saliency_orig, float))
    r = np.abs(np.asarray(saliency_random, float))
    m0 = float(o[roi].sum()) / float(o.sum()) if o.sum() else 0.0
    m1 = float(r[roi].sum()) / float(r.sum()) if r.sum() else 0.0
    drop = 1.0 - m1 / m0 if m0 else 0.0
    return {"roi_mass_orig": round(m0, 4), "roi_mass_random": round(m1, 4),
            "rel_drop": round(float(drop), 4), "passed": bool(drop > drop_thresh)}
