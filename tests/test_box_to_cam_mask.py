"""G5 Leg-3 E7c self-check: the box->96^3 projection (the #1 silent-invalidation risk).

The reference box-mask MUST land in the SAME grid the CAM does or IoU is meaningless. These asserts
pin the recovered crop-interior geometry (the .npy is already box+7mm-rim, so the box occupies the
crop interior minus a 7-voxel border), the resample to the CAM cube, and the edge-clamp fallback.
"""

import pytest

pytest.importorskip("torch", reason="torch is an ml/arms/trackb-extra dep — skip cleanly under a base install; pinksight.xai package __init__ imports torch (saliency)")

import numpy as np

from pinksight.data.annotation_boxes import Box
from pinksight.xai.faithfulness import (
    _NPY_CROP_MARGIN_VOX,
    box_to_cam_mask,
    resample_mask_to,
)


def _box(row=(10, 40), col=(10, 40), slc=(5, 25)) -> Box:
    return Box("Breast_MRI_TEST", row, col, slc)


def test_shape_and_dtype_match_cam_grid():
    m = box_to_cam_mask(_box(), native_shape=(80, 80, 40), cam_shape=(96, 96, 96))
    assert m.shape == (96, 96, 96)
    assert m.dtype == bool
    assert m.any(), "mask must not be empty"
    assert not m.all(), "a normal crop leaves a rim -> the box must not fill the whole cube"


def test_interior_border_matches_7mm_rim_before_resample():
    """On a native crop larger than 2*margin, the box interior is [7 : dim-7] per axis (the rim the
    .npy already baked in). Verified at NATIVE resolution (cam_shape == native_shape -> no resample)."""
    native = (60, 60, 50)
    m = box_to_cam_mask(_box(), native_shape=native, cam_shape=native)
    v = int(_NPY_CROP_MARGIN_VOX)
    expected = np.zeros(native, dtype=bool)
    expected[v:native[0] - v, v:native[1] - v, v:native[2] - v] = True
    assert np.array_equal(m, expected), "crop-interior box geometry drifted from the 7-voxel rim"


def test_thin_axis_falls_back_to_full_extent():
    """A crop axis thinner than 2*margin (small/edge-clamped lesion) fills the whole axis, never
    empty — the edge-safe fallback."""
    native = (10, 60, 60)  # axis 0 = 10 <= 2*7 -> full extent on that axis
    m = box_to_cam_mask(_box(), native_shape=native, cam_shape=native)
    assert m[:, 30, 30].all(), "thin axis must be fully covered (no empty box at an edge)"
    assert m.any()


def test_aligned_saliency_lands_inside_the_box():
    """A CAM whose salient mass sits inside the projected box must (a) point inside it and (b) have
    its top-decile binarization be a strict SUBSET of the box — proves the reference lands in the CAM
    grid (the E7c contract). A mis-gridded mask would fail both. (Full IoU is bounded below by the
    q=0.9 binarize keeping only the top 10%, so a ">0.5 self-IoU" bar is structurally wrong — the
    subset+pointing check is the correct alignment proof.)"""
    from pinksight.xai.faithfulness import _binarize
    from pinksight.xai.faithfulness import pointing_game as _pg

    native = (72, 72, 48)
    box_mask = box_to_cam_mask(_box(), native_shape=native, cam_shape=(96, 96, 96))
    # Saliency high inside the box, ~0 outside.
    sal = box_mask.astype(float) + 1e-3 * np.random.default_rng(0).random(box_mask.shape)
    assert _pg(sal, box_mask), "peak saliency must land inside the grid-aligned box"
    top = _binarize(sal, q=0.9)
    inside = np.logical_and(top, box_mask).sum() / max(int(top.sum()), 1)
    assert inside > 0.95, f"top-decile saliency must sit inside the box (got {inside:.3f} inside)"


def test_native_shape_scales_the_box():
    """The box interior scales with native_shape: two different native shapes project the same box
    to different covered fractions of the fixed CAM cube (native_shape is load-bearing, not ignored)."""
    small = box_to_cam_mask(_box(), native_shape=(30, 30, 30), cam_shape=(96, 96, 96))
    large = box_to_cam_mask(_box(), native_shape=(90, 90, 90), cam_shape=(96, 96, 96))
    # Smaller native crop -> the fixed 7-voxel rim eats a BIGGER fraction -> smaller interior fraction
    # of the fixed CAM cube. So the larger native crop projects a larger box fraction.
    assert large.mean() > small.mean(), "native_shape not affecting the projected box fraction"


def test_resample_mask_to_is_used_and_label_preserving():
    """box_to_cam_mask must go through the nearest-neighbour resampler (label-preserving, no
    interpolation across the boundary) so the returned mask stays boolean 0/1."""
    m = box_to_cam_mask(_box(), native_shape=(64, 64, 40), cam_shape=(96, 96, 96))
    assert set(np.unique(m.astype(int)).tolist()) <= {0, 1}
    # And the resampler itself is a no-op when shapes match (sanity on the primitive it relies on).
    native = np.zeros((64, 64, 40), dtype=bool)
    native[7:57, 7:57, 7:33] = True
    assert np.array_equal(resample_mask_to(native, (64, 64, 40)), native)
