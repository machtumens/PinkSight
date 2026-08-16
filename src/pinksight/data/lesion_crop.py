
from __future__ import annotations

import numpy as np

from ..models.h0_localizer import RIM_MM_DEFAULT, dilate_rim, largest_cc, mask_to_box
from .annotation_boxes import Box, crop

CROP_SIZE_DEFAULT = 96  

BOX_MARGIN_MM_DEFAULT = 20


def _full_box(spatial: tuple[int, int, int]) -> Box:
    r, c, s = spatial
    return Box("_dukebox_fallback", (0, r), (0, c), (0, s))


def derive_lesion_box(
    mask: np.ndarray | None, spatial: tuple[int, int, int], rim_mm: int = RIM_MM_DEFAULT
) -> tuple[Box, bool]:
    if mask is not None:
        m = np.asarray(mask).astype(bool)
        if m.ndim != 3:
            raise ValueError(f"mask must be 3D (row, col, slice), got shape {m.shape}")
        if m.shape != tuple(spatial):
            raise ValueError(
                f"predicted mask shape {m.shape} != cached crop spatial {tuple(spatial)} — the mask "
                "must live on the same 1mm-iso crop grid as the volume (LOCK-2 predicted-mask source)."
            )
        m = largest_cc(m)  
        if m.any():
            return mask_to_box(dilate_rim(m, rim_mm)), False  
    return _full_box(spatial), True


def resample_cube(vol: np.ndarray, out_size: int = CROP_SIZE_DEFAULT) -> np.ndarray:
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
    v = np.asarray(vol, dtype=np.float32)
    if v.ndim != 4:
        raise ValueError(f"vol must be 4D (C, row, col, slice), got shape {v.shape}")
    spatial = (v.shape[1], v.shape[2], v.shape[3])
    box, _ = derive_lesion_box(mask, spatial, rim_mm)
    cropped = np.stack([crop(v[c], box) for c in range(v.shape[0])], axis=0).astype(np.float32)
    return resample_cube(cropped, out_size)


def box_crop(
    vol: np.ndarray,
    mask: np.ndarray | None = None,
    box_margin_mm: int = BOX_MARGIN_MM_DEFAULT,
    out_size: int = CROP_SIZE_DEFAULT,
) -> np.ndarray:
    return lesion_crop(vol, mask, rim_mm=box_margin_mm, out_size=out_size)
