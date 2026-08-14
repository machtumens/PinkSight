"""P10 faithfulness gauntlet: localisation vs an INDEPENDENT reference + model-randomization.

The reference mask MUST be independent of what the model conditioned on. Scoring saliency against
its own H0 / GT crop is CIRCULAR (Rule 6): high IoU is then guaranteed and proves nothing. These
functions consume saliency tensors + a reference mask (model-agnostic). Localisation always ships
paired with the randomization sanity test. Shows "characterisation / localisation".
"""

from __future__ import annotations

import numpy as np


def _binarize(saliency, q=0.9) -> np.ndarray:
    """Top-(1-q) quantile of the saliency as the predicted salient region."""
    s = np.asarray(saliency, float)
    return s >= np.quantile(s, q)


def resample_mask_to(reference_mask, target_shape) -> np.ndarray:
    """Nearest-neighbour resample a 3D reference mask to `target_shape` (D,H,W) — TICKET-015.

    The CAM is upsampled to the model INPUT dims, but a tumor-mask `.npy` may still be at native
    segmentation resolution/grid. `iou()`/`pointing_game()` do a voxelwise `logical_and`, which
    requires identical shape AND axis order. We nearest-neighbour resample (label-preserving — no
    interpolation across class boundaries) so the comparison is voxel-aligned. Fails loud on any
    residual ndim/axis mismatch rather than silently scoring against a mis-gridded mask.
    """
    ref = np.asarray(reference_mask)
    target_shape = tuple(int(s) for s in target_shape)
    if ref.ndim != len(target_shape):
        raise ValueError(
            f"reference mask ndim {ref.ndim} != target ndim {len(target_shape)} "
            f"(mask shape {ref.shape}, target {target_shape}) — axis order mismatch, cannot resample."
        )
    if ref.shape == target_shape:
        return ref
    # Nearest-neighbour index map per axis: map each target voxel back to its source voxel.
    idx = np.ix_(*[
        np.clip((np.arange(t) * s // t), 0, s - 1)
        for s, t in zip(ref.shape, target_shape)
    ])
    out = ref[idx]
    if out.shape != target_shape:  # defensive — should be unreachable
        raise ValueError(f"resample produced {out.shape}, expected {target_shape}")
    return out


def iou(saliency, reference_mask, q=0.9) -> float:
    """IoU of the top-quantile saliency vs an INDEPENDENT reference mask. Target ≥ 0.30."""
    pred = _binarize(saliency, q)
    ref = np.asarray(reference_mask).astype(bool)
    union = np.logical_or(pred, ref).sum()
    return float(np.logical_and(pred, ref).sum()) / float(union) if union else 0.0


def pointing_game(saliency, reference_mask) -> bool:
    """Hit if the peak saliency voxel lands inside the reference region. Target ≥ 0.70 hit-rate."""
    peak = np.unravel_index(np.argmax(np.asarray(saliency)), np.asarray(saliency).shape)
    return bool(np.asarray(reference_mask).astype(bool)[peak])


def box_hit(saliency, box) -> bool:
    """Fallback when Duke ships only boxes (G0 finding): peak voxel inside (z0,z1,y0,y1,x0,x1)."""
    z, y, x = np.unravel_index(np.argmax(np.asarray(saliency)), np.asarray(saliency).shape)
    z0, z1, y0, y1, x0, x1 = box
    return bool(z0 <= z < z1 and y0 <= y < y1 and x0 <= x < x1)


# The Duke ROI `.npy` on disk is ALREADY the lesion box + this many voxels of peritumoral rim per
# face (preprocess.MARGIN_MM = 7 at 1mm-iso; lesion_crop.py header). So on a crop_mode="none" cube,
# the lesion box occupies the crop INTERIOR with a `_NPY_CROP_MARGIN_VOX` border on each face.
_NPY_CROP_MARGIN_VOX = 7


def box_to_cam_mask(box, native_shape, cam_shape=(96, 96, 96)) -> np.ndarray:
    """Project a Duke lesion box into the model's CAM grid (G5 Leg-3, execute-agent E7c).

    The reference mask MUST live in the SAME coordinate frame the CAM does, or the IoU/pointing
    score is silently invalid (E7c: "#1 risk"). Recovering that frame:

    The trained encoder ate a `crop_mode="none"` input — `data/processed/{pid}.npy`
    (already the Duke box + 7mm peritumoral rim, 1mm-iso; see `preprocess.preprocess_patient`)
    -> `select_channels(first_post)` -> `monai Resize(spatial_size=cam_shape, mode="trilinear")`
    -> z-norm. There is NO further crop. Therefore the WHOLE native `.npy` volume maps 1:1 onto the
    CAM cube, and the lesion box itself sits in the crop INTERIOR with a `_NPY_CROP_MARGIN_VOX`
    border on every face (minus any face the original box clamped to the volume edge — unknowable
    from the cropped `.npy` alone, so the honest, edge-safe reconstruction keeps the full margin).

    Args:
        box: `annotation_boxes.Box` — half-open (row, col, slice) ranges in NATIVE (pre-crop) image
            coords. Only used here to confirm the type contract; the box's crop-interior position is
            derived from `native_shape` + the fixed 7mm rim, NOT from the raw native coordinates
            (those live in the full-MRI frame, not the `.npy` crop frame — see module note).
        native_shape: the `.npy` crop's spatial shape `(D, H, W)` == `vol_npy.shape[1:]` — required to
            scale the interior box to `cam_shape` (E7c mandates this parameter).
        cam_shape: the CAM grid (the encoder input size). Default 96^3 (the trained config).

    Returns:
        A boolean `cam_shape` array — True inside the projected lesion box. Leakage-safe: the box
        annotation is never a training input; this mask is only a scoring reference.
    """
    native_shape = tuple(int(s) for s in native_shape)
    if len(native_shape) != 3:
        raise ValueError(f"native_shape must be 3D (D,H,W), got {native_shape}")
    if getattr(box, "row", None) is None:  # type contract guard — must be an annotation Box
        raise TypeError("box must be an annotation_boxes.Box with row/col/slice ranges")

    m = int(_NPY_CROP_MARGIN_VOX)
    mask_native = np.zeros(native_shape, dtype=bool)
    sl = []
    for dim in native_shape:
        # Interior = [margin : dim-margin]. When the crop is thinner than 2*margin the lesion fills
        # the whole axis (a small/edge-clamped lesion) — never an empty box.
        lo, hi = (m, dim - m) if dim > 2 * m else (0, dim)
        sl.append(slice(lo, hi))
    mask_native[sl[0], sl[1], sl[2]] = True
    return resample_mask_to(mask_native, cam_shape).astype(bool)


def randomization_test(saliency_orig, saliency_random, q=0.9, drop_thresh=0.5) -> dict:
    """Sanity (Adebayo): saliency mass in the original ROI must drop > 50% under weight randomization.

    A faithful map concentrates in its ROI; a randomized-weight map should not. We measure the
    fraction of |saliency| falling inside the ORIGINAL top-quantile ROI, before vs after.

    # ponytail: rel_drop here comes from a SINGLE weight re-init seed, so it is noisy — a single
    # unlucky draw can push it spuriously negative. That noise is tolerable while the TICKET-001
    # gate is only "reported, not gated" (no trained encoder yet). When TICKET-001 is flipped to a
    # hard gate, upgrade this to average rel_drop over several re-init seeds; deferred until then to
    # keep the primitive minimal (see TICKET-024 LOW).
    """
    roi = _binarize(saliency_orig, q)
    o = np.abs(np.asarray(saliency_orig, float))
    r = np.abs(np.asarray(saliency_random, float))
    m0 = float(o[roi].sum()) / float(o.sum()) if o.sum() else 0.0
    m1 = float(r[roi].sum()) / float(r.sum()) if r.sum() else 0.0
    drop = 1.0 - m1 / m0 if m0 else 0.0
    return {"roi_mass_orig": round(m0, 4), "roi_mass_random": round(m1, 4),
            "rel_drop": round(float(drop), 4), "passed": bool(drop > drop_thresh)}
