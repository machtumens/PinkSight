"""[G2-LESION-CROP] Tight mask-derived lesion crop -> fixed encoder resolution (the geometry lever).

Pre-registered in reports/G2_imaging/G2-LESION-CROP_PREREG_10-07-26.md. The prior imaging arms fed the
encoder the LOOSE Duke-box crop (`preprocess.preprocess_patient`): the Duke `Annotation_Boxes.xlsx` box
+ 7mm margin at native 1mm-iso scale. That leaves the tumour under-filling a large ragged crop at
inconsistent apparent scale (baseline `r18_mn` pooled-OOF 0.518, at chance). This module derives a
TIGHT box from the H0 PREDICTED mask (`h0_localizer`), then resamples to a FIXED cube (96^3) so the
lesion fills the receptive field at a consistent scale across patients.

LOCK-2 (baked in, non-negotiable): the crop box is derived from the H0 nnU-Net PREDICTED mask, NEVER a
ground-truth mask, at train AND test. This module has exactly two box sources — (a) a predicted mask
passed in by the caller, or (b) the [1.6] Duke-box fallback = the ALREADY-Duke-box-cropped cached
volume's own extent (oracle-free; no ground-truth segmentation is ever read here). A ground-truth mask
is structurally unreachable from this code path.

[1.6] Duke-box fallback (non-negotiable): if the predicted mask is None / empty / degenerate, the box
falls back to the full cached crop (which is itself the Duke box + 7mm, produced upstream) so a poor H0
prediction yields a valid whole-crop-at-96^3, NEVER a silent empty crop. `mask_to_box` raises on empty,
so the fallback fires deliberately, not by accident.

Grid: cached volumes are (C, row, col, slice) float32 on P03's 1mm-iso grid (so 1 voxel == 1 mm and a
predicted mask over the same crop shares the grid). Output is (C, out_size, out_size, out_size) float32.
"""

from __future__ import annotations

import numpy as np

from ..models.h0_localizer import RIM_MM_DEFAULT, dilate_rim, largest_cc, mask_to_box
from .annotation_boxes import Box, crop

CROP_SIZE_DEFAULT = 96  # fixed encoder cube edge ([3.4] "fixed moderate crop ~96^3-128^3"; G0-stat floor)

# [HEAD2-GRADE-PIVOT box-crop] radiologist-style BOX margin (mm == voxels on the 1mm-iso grid). This
# is the ONE deliberate lever vs the S29 tight-lesion re-smoke (`crop_mode=lesion`, +7mm rim → NULL).
# DeepRadGrade (Eur Radiol 2025, same Duke cohort, test AUC 0.82) cropped a radiologist BOUNDING BOX
# with a PERITUMORAL MARGIN — a larger box-shaped context region, NOT a tight mask hug. [1.9]/[3.4]
# lock "lesion crop + peritumoral margin, margin size is a hyperparameter"; LOCK-3's 5-10mm band is the
# TIGHT-rim setting (dilate_rim), whereas this box margin is that explicit tunable. 20mm ≈ doubles a
# typical ~20-30mm lesion extent → tumour + surrounding parenchyma, the radiologist-box analog.
BOX_MARGIN_MM_DEFAULT = 20


def _full_box(spatial: tuple[int, int, int]) -> Box:
    """[1.6] fallback box = the whole cached crop's extent (already the Duke box + 7mm upstream)."""
    r, c, s = spatial
    return Box("_dukebox_fallback", (0, r), (0, c), (0, s))


def derive_lesion_box(
    mask: np.ndarray | None, spatial: tuple[int, int, int], rim_mm: int = RIM_MM_DEFAULT
) -> tuple[Box, bool]:
    """Predicted mask -> tight ROI box (+peritumoral rim/margin), or the [1.6] Duke-box fallback.

    Returns (box, used_fallback). The box is derived from the PREDICTED `mask` only (LOCK-2); when the
    mask is None / all-zero / raises on boxing, the [1.6] fallback box spanning the full cached crop is
    returned instead so the caller never produces an empty crop. `mask` must match `spatial` (the cached
    crop grid); a mismatched mask is a caller bug and raises.

    `rim_mm` is the isotropic dilation applied to the mask before boxing — the SAME knob for the tight
    lesion crop (7mm rim) and the radiologist BOX crop (a larger `BOX_MARGIN_MM_DEFAULT`-style margin).
    The dilation math is identical (1 iteration == 1mm on the iso grid); only the magnitude differs, so
    "tight lesion" and "box" are one code path parameterised by margin size ([1.9] margin hyperparameter).
    """
    if mask is not None:
        m = np.asarray(mask).astype(bool)
        if m.ndim != 3:
            raise ValueError(f"mask must be 3D (row, col, slice), got shape {m.shape}")
        if m.shape != tuple(spatial):
            raise ValueError(
                f"predicted mask shape {m.shape} != cached crop spatial {tuple(spatial)} — the mask "
                "must live on the same 1mm-iso crop grid as the volume (LOCK-2 predicted-mask source)."
            )
        m = largest_cc(m)  # drop nnU-Net FP specks; keep the dominant lesion blob
        if m.any():
            return mask_to_box(dilate_rim(m, rim_mm)), False  # predicted box + rim/margin (LOCK-2)
    # [1.6] Duke-box fallback: empty/None/degenerate predicted mask -> whole cached crop, never empty.
    return _full_box(spatial), True


def resample_cube(vol: np.ndarray, out_size: int = CROP_SIZE_DEFAULT) -> np.ndarray:
    """Trilinear-resample a (C, D, H, W) float32 crop to a fixed (C, out_size^3) cube.

    Uses `monai.transforms.Resize` (same transform + settings NpyVolumeDataset already uses to hit a
    fixed grid) so the fixed-scale normalisation is byte-identical to the existing resize path — the
    ONLY difference from the loose-box baseline is that the box fed in here is tight, not ragged.
    """
    from monai.transforms import Resize

    x = np.ascontiguousarray(vol, dtype=np.float32)
    resize = Resize(spatial_size=(out_size, out_size, out_size), mode="trilinear", align_corners=False)
    return np.asarray(resize(x), dtype=np.float32)


def lesion_crop(
    vol: np.ndarray,
    mask: np.ndarray | None = None,
    rim_mm: int = RIM_MM_DEFAULT,
    out_size: int = CROP_SIZE_DEFAULT,
) -> np.ndarray:
    """Cached ROI volume + PREDICTED mask -> tight lesion crop resampled to a fixed cube (the arm's input).

    `vol` is (C, row, col, slice) float32 (the cached Duke-box crop). `mask` is the H0 PREDICTED tumour
    mask on the SAME grid, or None. Pipeline: derive tight box from the predicted mask (LOCK-2) ->
    `crop` -> trilinear-resample to (C, out_size, out_size, out_size). Empty/None/degenerate mask ->
    [1.6] Duke-box fallback (whole cached crop), so the output is ALWAYS a valid non-empty fixed cube.
    """
    v = np.asarray(vol, dtype=np.float32)
    if v.ndim != 4:
        raise ValueError(f"vol must be 4D (C, row, col, slice), got shape {v.shape}")
    spatial = (v.shape[1], v.shape[2], v.shape[3])
    box, _ = derive_lesion_box(mask, spatial, rim_mm)
    # crop() operates on (row, col, slice); apply per-channel so all phases share the one lesion box.
    cropped = np.stack([crop(v[c], box) for c in range(v.shape[0])], axis=0).astype(np.float32)
    return resample_cube(cropped, out_size)


def box_crop(
    vol: np.ndarray,
    mask: np.ndarray | None = None,
    box_margin_mm: int = BOX_MARGIN_MM_DEFAULT,
    out_size: int = CROP_SIZE_DEFAULT,
) -> np.ndarray:
    """Radiologist-style BOX crop: predicted-mask box + a LARGE peritumoral margin -> fixed cube.

    Identical machinery to `lesion_crop` (LOCK-2 predicted-mask source + [1.6] Duke-box fallback +
    trilinear resample to a fixed cube); the ONLY difference is the margin magnitude — a generous
    `box_margin_mm` (default 20mm) instead of the tight 7mm lesion rim. This is the DeepRadGrade
    radiologist-box analog (the S29 tight-lesion re-smoke used 7mm and NULLED; the box margin is the
    genuinely untested, ADR-0005-faithful lever). Kept as a named entry point so callers and configs
    read `box_crop`/`box_margin_mm` rather than an opaque large `rim_mm`.
    """
    return lesion_crop(vol, mask, rim_mm=box_margin_mm, out_size=out_size)
