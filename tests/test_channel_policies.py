"""Channel-policy unit tests (Tier 1, skip-if-no-torch): subtraction/kinetic shapes + edge cases.

Proves the [G2-SUBTRACTION-REOPEN] channel policies added to select_channels()/n_channels():
  * subtraction (2ch: first_post-pre, second_post-pre) -> (2, D, H, W) for c>=2
  * kinetic (4ch: pre, first_post, first_post-pre, second_post-pre) -> (4, D, H, W) for c>=3
  * edge cases: c=2 (kinetic pads the missing second_post-pre), c=3, c=4+
  * existing policies (first_post/pre_post/fixed4) remain byte-identical
  * n_channels() agrees with the actual select_channels() output channel count

All shape tests run on CPU with synthetic arrays — no GPU, no preprocessed cohort. Values are also
checked: the subtraction channel must equal (post - pre) BEFORE z-norm (select_channels is applied
to the raw volume; _znorm runs after in NpyVolumeDataset.__getitem__).
"""

from __future__ import annotations

import numpy as np
import pytest

# select_channels/n_channels are pure numpy (no torch), but keep the import guard consistent with
# the rest of the suite so a torch-less runner skips uniformly rather than half-collecting.
pytest.importorskip("monai")

from pinksight.data.dataset import (
    CHANNEL_POLICIES,
    n_channels,
    select_channels,
)


def _vol(c: int, d: int = 6, h: int = 5, w: int = 4) -> np.ndarray:
    """Deterministic (c, d, h, w) float32 volume; each channel is a distinct constant-plus-ramp so
    subtraction differences are exactly predictable (channel k has base value 10*k)."""
    rng = np.random.default_rng(0)
    ramp = rng.standard_normal((d, h, w)).astype(np.float32)
    return np.stack([10.0 * k + ramp for k in range(c)], axis=0).astype(np.float32)


# --- registry wiring (the four coupled targets must all know the new policies) ---

def test_channel_policies_tuple_contains_new_policies():
    assert "subtraction" in CHANNEL_POLICIES
    assert "kinetic" in CHANNEL_POLICIES
    # existing policies preserved (no accidental drop)
    assert "first_post" in CHANNEL_POLICIES
    assert "pre_post" in CHANNEL_POLICIES
    assert "fixed4" in CHANNEL_POLICIES


def test_n_channels_new_policies():
    assert n_channels("subtraction") == 2
    assert n_channels("kinetic") == 4
    # existing entries unchanged
    assert n_channels("first_post") == 1
    assert n_channels("pre_post") == 2
    assert n_channels("fixed4") == 4


# --- subtraction (2ch) ---

def test_subtraction_shape_c3():
    x = select_channels(_vol(3), "subtraction")
    assert x.shape == (2, 6, 5, 4)


def test_subtraction_shape_c2_falls_back():
    """With only 2 phases (pre + first_post), the second_post-pre channel has no second_post to use;
    it must fall back to (first_post - pre) so the output is still 2ch (never a crash / ragged shape)."""
    x = select_channels(_vol(2), "subtraction")
    assert x.shape == (2, 6, 5, 4)


def test_subtraction_shape_c5():
    x = select_channels(_vol(5), "subtraction")
    assert x.shape == (2, 6, 5, 4)  # only first_post/second_post used, extra phases ignored


def test_subtraction_values_are_post_minus_pre():
    """Channel 0 == first_post - pre; channel 1 == second_post - pre — computed on the RAW volume
    (before z-norm). This is the enhancement-difference invariant the plan's Data Flow requires."""
    vol = _vol(3)
    x = select_channels(vol, "subtraction")
    np.testing.assert_allclose(x[0], vol[1] - vol[0], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(x[1], vol[2] - vol[0], rtol=1e-6, atol=1e-6)


def test_subtraction_c2_second_channel_equals_first():
    vol = _vol(2)
    x = select_channels(vol, "subtraction")
    np.testing.assert_allclose(x[0], vol[1] - vol[0], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(x[1], vol[1] - vol[0], rtol=1e-6, atol=1e-6)  # fallback


# --- kinetic (4ch) ---

def test_kinetic_shape_c4():
    x = select_channels(_vol(4), "kinetic")
    assert x.shape == (4, 6, 5, 4)


def test_kinetic_shape_c3():
    x = select_channels(_vol(3), "kinetic")
    assert x.shape == (4, 6, 5, 4)


def test_kinetic_shape_c2_pads():
    """Edge: c=2 (pre + first_post only). kinetic still yields 4ch — the missing second_post-pre
    channel pads with (first_post - pre) rather than crashing (plan Phase 1 guard)."""
    x = select_channels(_vol(2), "kinetic")
    assert x.shape == (4, 6, 5, 4)


def test_kinetic_values_are_raw_plus_subtraction():
    """kinetic = [pre, first_post, (first_post - pre), (second_post - pre)] on the RAW volume."""
    vol = _vol(3)
    x = select_channels(vol, "kinetic")
    np.testing.assert_allclose(x[0], vol[0], rtol=1e-6, atol=1e-6)              # pre
    np.testing.assert_allclose(x[1], vol[1], rtol=1e-6, atol=1e-6)              # first_post
    np.testing.assert_allclose(x[2], vol[1] - vol[0], rtol=1e-6, atol=1e-6)     # first_post - pre
    np.testing.assert_allclose(x[3], vol[2] - vol[0], rtol=1e-6, atol=1e-6)     # second_post - pre


def test_kinetic_c2_pads_last_channel_with_first_diff():
    vol = _vol(2)
    x = select_channels(vol, "kinetic")
    np.testing.assert_allclose(x[0], vol[0], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(x[1], vol[1], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(x[2], vol[1] - vol[0], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(x[3], vol[1] - vol[0], rtol=1e-6, atol=1e-6)  # pad == first diff


# --- existing policies remain byte-identical after the additions ---

def test_existing_policies_unchanged():
    vol = _vol(4)
    fp = select_channels(vol, "first_post")
    assert fp.shape == (1, 6, 5, 4)
    np.testing.assert_allclose(fp[0], vol[1], rtol=1e-6, atol=1e-6)

    pp = select_channels(vol, "pre_post")
    assert pp.shape == (2, 6, 5, 4)
    np.testing.assert_allclose(pp[0], vol[0], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(pp[1], vol[1], rtol=1e-6, atol=1e-6)

    f4 = select_channels(vol, "fixed4")
    assert f4.shape == (4, 6, 5, 4)


def test_unknown_policy_still_raises():
    with pytest.raises(ValueError):
        select_channels(_vol(3), "nonsense")


# --- NpyVolumeDataset end-to-end: new policies drive the correct in_channels through __getitem__ ---

def test_dataset_yields_subtraction_channels(tmp_path):
    from pinksight.data.dataset import NpyVolumeDataset

    np.save(tmp_path / "FAKE_000.npy", _vol(3, 8, 8, 8))
    x, _, _ = NpyVolumeDataset([("FAKE_000", 0)], tmp_path, channels="subtraction",
                               spatial_size=(8, 8, 8))[0]
    assert x.shape == (2, 8, 8, 8)


def test_dataset_yields_kinetic_channels(tmp_path):
    from pinksight.data.dataset import NpyVolumeDataset

    np.save(tmp_path / "FAKE_000.npy", _vol(3, 8, 8, 8))
    x, _, _ = NpyVolumeDataset([("FAKE_000", 0)], tmp_path, channels="kinetic",
                               spatial_size=(8, 8, 8))[0]
    assert x.shape == (4, 8, 8, 8)
