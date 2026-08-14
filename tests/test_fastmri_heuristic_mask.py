"""Unit tests for the label-blind heuristic enhancement mask (lesion-crop plan, Section 2 + 8).

Covers the plan's Verification Evidence rows for ``enhancement_mask``:
  * shape/dtype on a synthetic ramp;
  * degenerate (all-equal / all-zero) input returns a valid (empty) mask, never a crash — proving the
    ``[1.6]`` Duke-box fallback will fire cleanly downstream;
  * STRUCTURAL label-blindness (``inspect.signature`` — no label parameter, Section 2 rule 3);
  * determinism (byte-identical mask on byte-identical inputs, Section 2 rule 3);
  * the mask actually localizes enhancement and feeds the tested ``derive_lesion_box`` machinery
    (non-fallback for a real blob, fallback for a degenerate mask).

    PYTHONPATH=src .venv/bin/python -m pytest tests/test_fastmri_heuristic_mask.py -q
"""

from __future__ import annotations

import pytest

pytest.importorskip("scipy", reason="scipy is an ml/arms/trackb-extra dep — skip cleanly under a base install; imports the scipy-backed eval stack")

import inspect

import numpy as np

from pinksight.data.fastmri_heuristic_mask import enhancement_mask
from pinksight.data.lesion_crop import derive_lesion_box

_SPATIAL = (24, 20, 28)


def _blob_pair(spatial=_SPATIAL, blob=((10, 15), (8, 12), (12, 18)), lift=50.0):
    """A (pre, post) pair whose ONLY enhancing region is a rectangular blob (post = pre + lift there)."""
    rng = np.random.default_rng(0)
    pre = rng.random(spatial).astype(np.float32)
    post = pre.copy()
    (r0, r1), (c0, c1), (s0, s1) = blob
    post[r0:r1, c0:c1, s0:s1] += lift  # a clearly-enhancing lesion-like region
    return pre, post


# --- shape / dtype -------------------------------------------------------------------------------

def test_enhancement_mask_basic_shape_and_dtype():
    """A synthetic ramp -> boolean mask matching the input spatial shape (plan Section 8)."""
    pre = np.zeros(_SPATIAL, dtype=np.float32)
    post = np.arange(np.prod(_SPATIAL), dtype=np.float32).reshape(_SPATIAL)  # monotone ramp
    mask = enhancement_mask(pre, post)
    assert mask.shape == _SPATIAL
    assert mask.dtype == bool
    # percentile=90 keeps the top ~10% of the ramp -> some but not all voxels selected.
    frac = float(mask.mean())
    assert 0.0 < frac < 0.5


# --- degenerate input (proves the [1.6] fallback path is reachable, not a crash) ------------------

def test_enhancement_mask_degenerate_input_returns_valid_mask():
    """All-zero and all-equal (pre == post) pairs must return a well-formed EMPTY mask, never raise."""
    z = np.zeros(_SPATIAL, dtype=np.float32)
    m_zero = enhancement_mask(z, z)
    assert m_zero.shape == _SPATIAL and m_zero.dtype == bool
    assert not m_zero.any()  # strict '>' on a constant enhancement image -> empty mask

    const = np.full(_SPATIAL, 7.0, dtype=np.float32)
    m_const = enhancement_mask(const, const)
    assert m_const.dtype == bool and not m_const.any()


# --- Section 2 rule 3: structural label-blindness --------------------------------------------------

def test_signature_has_no_label_parameter():
    """No label-dependent parameter may exist on enhancement_mask (Section 2 rule 3)."""
    params = set(inspect.signature(enhancement_mask).parameters)
    forbidden = {"label", "y", "target", "malignant", "class_id"}
    leaked = params & forbidden
    assert not leaked, f"label-dependent parameter(s) leaked into the mask signature: {sorted(leaked)}"


def test_identical_images_yield_identical_masks_regardless_of_label():
    """Determinism: byte-identical (pre, post) -> byte-identical mask (no hidden state/label branch)."""
    pre, post = _blob_pair()
    m1 = enhancement_mask(pre, post)
    m2 = enhancement_mask(pre.copy(), post.copy())  # byte-identical, independent buffers
    assert np.array_equal(m1, m2)
    assert m1.dtype == bool


def test_does_not_mutate_inputs():
    """The mask must never mutate pre/post (a mutation would break the determinism guarantee)."""
    pre, post = _blob_pair()
    pre_before, post_before = pre.copy(), post.copy()
    _ = enhancement_mask(pre, post)
    assert np.array_equal(pre, pre_before)
    assert np.array_equal(post, post_before)


# --- boundary validation ---------------------------------------------------------------------------

def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        enhancement_mask(np.zeros((4, 4, 4), np.float32), np.zeros((4, 4, 5), np.float32))


def test_invalid_method_raises():
    with pytest.raises(ValueError):
        enhancement_mask(np.zeros(_SPATIAL, np.float32), np.zeros(_SPATIAL, np.float32), method="otsu")


def test_percentile_out_of_range_raises():
    with pytest.raises(ValueError):
        enhancement_mask(np.zeros(_SPATIAL, np.float32), np.zeros(_SPATIAL, np.float32), percentile=150.0)


# --- localizes enhancement + feeds the tested derive_lesion_box machinery --------------------------

def test_enhancing_blob_is_selected_and_background_is_not():
    """The mask must fire inside a clearly-enhancing blob and stay (mostly) off the flat background."""
    pre, post = _blob_pair()
    mask = enhancement_mask(pre, post)
    (r0, r1), (c0, c1), (s0, s1) = ((10, 15), (8, 12), (12, 18))
    blob = mask[r0:r1, c0:c1, s0:s1]
    assert blob.mean() > 0.9  # the enhancing region is almost entirely selected
    # background selection is small: nearly all True voxels come from the blob.
    assert float(mask.sum()) < 2.0 * float(blob.sum())


def test_mask_feeds_derive_lesion_box_non_fallback():
    """A real enhancing blob -> non-empty mask -> derive_lesion_box uses it (NOT the [1.6] fallback)."""
    pre, post = _blob_pair()
    mask = enhancement_mask(pre, post)
    _box, used_fallback = derive_lesion_box(mask, _SPATIAL, rim_mm=7)
    assert used_fallback is False  # the heuristic mask actually drove the box, Rung 0 mechanism proven


def test_degenerate_mask_triggers_fallback_downstream():
    """A degenerate (empty) mask -> derive_lesion_box hits the [1.6] Duke-box fallback, never empty."""
    z = np.zeros(_SPATIAL, dtype=np.float32)
    mask = enhancement_mask(z, z)  # empty
    box, used_fallback = derive_lesion_box(mask, _SPATIAL, rim_mm=7)
    assert used_fallback is True
    assert box.row == (0, _SPATIAL[0])  # whole-crop fallback extent
