"""H0 hand-off checks (Tier 0, scipy-only — always runs): rim dilation, mask->Box, DSC, gating.

Verifies the genuinely-new P04 piece: a predicted tumour mask becomes the encoder ROI and plugs
straight into `preprocess.crop()`. No torch/data/weights needed.
"""

import pytest

pytest.importorskip("scipy", reason="scipy is an ml/arms/trackb-extra dep — skip cleanly under a base install; pinksight.models.h0_localizer imports scipy")

import numpy as np

from pinksight.data.annotation_boxes import Box, crop
from pinksight.models.h0_localizer import (
    WeightsNotProvisioned,
    dice,
    dilate_rim,
    mask_to_box,
    predict_mask,
    predicted_roi_box,
)


def _point_mask(shape=(21, 21, 21), center=(10, 10, 10)) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    m[center] = True
    return m


def test_dilate_rim_grows_by_rim_mm_on_iso_grid():
    # 1 voxel == 1 mm: a point dilated by 5 mm spans [center-5, center+5] on each axis.
    box = mask_to_box(dilate_rim(_point_mask(), rim_mm=5))
    assert box.row == (5, 16) and box.col == (5, 16) and box.slice == (5, 16)


def test_dilate_rim_zero_is_identity_and_negative_rejected():
    m = _point_mask()
    assert np.array_equal(dilate_rim(m, rim_mm=0), m)
    with pytest.raises(ValueError):
        dilate_rim(m, rim_mm=-1)


def test_mask_to_box_is_tight_and_half_open():
    m = np.zeros((10, 10, 10), dtype=bool)
    m[4:7, 5:9, 2:3] = True
    box = mask_to_box(m, patient="p")
    assert box == Box("p", (4, 7), (5, 9), (2, 3))


def test_mask_to_box_rejects_empty_and_non_3d():
    with pytest.raises(ValueError):
        mask_to_box(np.zeros((5, 5, 5), dtype=bool))
    with pytest.raises(ValueError):
        mask_to_box(np.zeros((5, 5), dtype=bool))


def test_predicted_roi_box_feeds_preprocess_crop():
    # The integration contract: H0 mask -> ROI box -> crop() of a (row, col, slice) volume.
    vol = np.arange(20 * 20 * 20).reshape(20, 20, 20)
    mask = np.zeros((20, 20, 20), dtype=bool)
    mask[8:12, 9:11, 7:13] = True
    box = predicted_roi_box(mask, rim_mm=2, patient="p")
    assert crop(vol, box).shape == (
        8,
        6,
        10,
    )  # lesion extent + 2mm rim each side, clipped in-bounds


def test_dice_overlap_cases():
    a = np.zeros((4, 4, 4), dtype=bool)
    b = np.zeros((4, 4, 4), dtype=bool)
    a[0:2] = True
    b[1:3] = True
    assert dice(a, a) == 1.0  # perfect
    assert dice(a, np.zeros_like(a)) == 0.0  # disjoint (one empty)
    assert dice(np.zeros_like(a), np.zeros_like(a)) == 1.0  # both empty -> identical
    assert dice(a, b) == pytest.approx(0.5)  # half overlap
    with pytest.raises(ValueError):
        dice(a, np.zeros((4, 4), dtype=bool))


def test_predict_mask_is_gated_not_fabricated():
    # LOCK-2: no fabricated/oracle masks at test time until real weights are wired.
    with pytest.raises(WeightsNotProvisioned):
        predict_mask(np.zeros((4, 4, 4), dtype=np.float32))
