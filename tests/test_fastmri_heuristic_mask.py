
from __future__ import annotations

import pytest

pytest.importorskip("scipy", reason="scipy is an ml/arms/trackb-extra dep — skip cleanly under a base install; imports the scipy-backed eval stack")

import inspect

import numpy as np

from pinksight.data.fastmri_heuristic_mask import enhancement_mask
from pinksight.data.lesion_crop import derive_lesion_box

_SPATIAL = (24, 20, 28)


def _blob_pair(spatial=_SPATIAL, blob=((10, 15), (8, 12), (12, 18)), lift=50.0):
    rng = np.random.default_rng(0)
    pre = rng.random(spatial).astype(np.float32)
    post = pre.copy()
    (r0, r1), (c0, c1), (s0, s1) = blob
    post[r0:r1, c0:c1, s0:s1] += lift  
    return pre, post


def test_enhancement_mask_basic_shape_and_dtype():
    pre = np.zeros(_SPATIAL, dtype=np.float32)
    post = np.arange(np.prod(_SPATIAL), dtype=np.float32).reshape(_SPATIAL)  
    mask = enhancement_mask(pre, post)
    assert mask.shape == _SPATIAL
    assert mask.dtype == bool
    frac = float(mask.mean())
    assert 0.0 < frac < 0.5


def test_enhancement_mask_degenerate_input_returns_valid_mask():
    z = np.zeros(_SPATIAL, dtype=np.float32)
    m_zero = enhancement_mask(z, z)
    assert m_zero.shape == _SPATIAL and m_zero.dtype == bool
    assert not m_zero.any()  

    const = np.full(_SPATIAL, 7.0, dtype=np.float32)
    m_const = enhancement_mask(const, const)
    assert m_const.dtype == bool and not m_const.any()


def test_signature_has_no_label_parameter():
    params = set(inspect.signature(enhancement_mask).parameters)
    forbidden = {"label", "y", "target", "malignant", "class_id"}
    leaked = params & forbidden
    assert not leaked, f"label-dependent parameter(s) leaked into the mask signature: {sorted(leaked)}"


def test_identical_images_yield_identical_masks_regardless_of_label():
    pre, post = _blob_pair()
    m1 = enhancement_mask(pre, post)
    m2 = enhancement_mask(pre.copy(), post.copy())  
    assert np.array_equal(m1, m2)
    assert m1.dtype == bool


def test_does_not_mutate_inputs():
    pre, post = _blob_pair()
    pre_before, post_before = pre.copy(), post.copy()
    _ = enhancement_mask(pre, post)
    assert np.array_equal(pre, pre_before)
    assert np.array_equal(post, post_before)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        enhancement_mask(np.zeros((4, 4, 4), np.float32), np.zeros((4, 4, 5), np.float32))


def test_invalid_method_raises():
    with pytest.raises(ValueError):
        enhancement_mask(np.zeros(_SPATIAL, np.float32), np.zeros(_SPATIAL, np.float32), method="otsu")


def test_percentile_out_of_range_raises():
    with pytest.raises(ValueError):
        enhancement_mask(np.zeros(_SPATIAL, np.float32), np.zeros(_SPATIAL, np.float32), percentile=150.0)


def test_enhancing_blob_is_selected_and_background_is_not():
    pre, post = _blob_pair()
    mask = enhancement_mask(pre, post)
    (r0, r1), (c0, c1), (s0, s1) = ((10, 15), (8, 12), (12, 18))
    blob = mask[r0:r1, c0:c1, s0:s1]
    assert blob.mean() > 0.9  
    assert float(mask.sum()) < 2.0 * float(blob.sum())


def test_mask_feeds_derive_lesion_box_non_fallback():
    pre, post = _blob_pair()
    mask = enhancement_mask(pre, post)
    _box, used_fallback = derive_lesion_box(mask, _SPATIAL, rim_mm=7)
    assert used_fallback is False  


def test_degenerate_mask_triggers_fallback_downstream():
    z = np.zeros(_SPATIAL, dtype=np.float32)
    mask = enhancement_mask(z, z)  
    box, used_fallback = derive_lesion_box(mask, _SPATIAL, rim_mm=7)
    assert used_fallback is True
    assert box.row == (0, _SPATIAL[0])  
