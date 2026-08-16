
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("monai")

from pinksight.data.dataset import (
    CHANNEL_POLICIES,
    n_channels,
    select_channels,
)


def _vol(c: int, d: int = 6, h: int = 5, w: int = 4) -> np.ndarray:
    rng = np.random.default_rng(0)
    ramp = rng.standard_normal((d, h, w)).astype(np.float32)
    return np.stack([10.0 * k + ramp for k in range(c)], axis=0).astype(np.float32)


def test_channel_policies_tuple_contains_new_policies():
    assert "subtraction" in CHANNEL_POLICIES
    assert "kinetic" in CHANNEL_POLICIES
    assert "first_post" in CHANNEL_POLICIES
    assert "pre_post" in CHANNEL_POLICIES
    assert "fixed4" in CHANNEL_POLICIES


def test_n_channels_new_policies():
    assert n_channels("subtraction") == 2
    assert n_channels("kinetic") == 4
    assert n_channels("first_post") == 1
    assert n_channels("pre_post") == 2
    assert n_channels("fixed4") == 4


def test_subtraction_shape_c3():
    x = select_channels(_vol(3), "subtraction")
    assert x.shape == (2, 6, 5, 4)


def test_subtraction_shape_c2_falls_back():
    x = select_channels(_vol(2), "subtraction")
    assert x.shape == (2, 6, 5, 4)


def test_subtraction_shape_c5():
    x = select_channels(_vol(5), "subtraction")
    assert x.shape == (2, 6, 5, 4)  


def test_subtraction_values_are_post_minus_pre():
    vol = _vol(3)
    x = select_channels(vol, "subtraction")
    np.testing.assert_allclose(x[0], vol[1] - vol[0], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(x[1], vol[2] - vol[0], rtol=1e-6, atol=1e-6)


def test_subtraction_c2_second_channel_equals_first():
    vol = _vol(2)
    x = select_channels(vol, "subtraction")
    np.testing.assert_allclose(x[0], vol[1] - vol[0], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(x[1], vol[1] - vol[0], rtol=1e-6, atol=1e-6)  


def test_kinetic_shape_c4():
    x = select_channels(_vol(4), "kinetic")
    assert x.shape == (4, 6, 5, 4)


def test_kinetic_shape_c3():
    x = select_channels(_vol(3), "kinetic")
    assert x.shape == (4, 6, 5, 4)


def test_kinetic_shape_c2_pads():
    x = select_channels(_vol(2), "kinetic")
    assert x.shape == (4, 6, 5, 4)


def test_kinetic_values_are_raw_plus_subtraction():
    vol = _vol(3)
    x = select_channels(vol, "kinetic")
    np.testing.assert_allclose(x[0], vol[0], rtol=1e-6, atol=1e-6)              
    np.testing.assert_allclose(x[1], vol[1], rtol=1e-6, atol=1e-6)              
    np.testing.assert_allclose(x[2], vol[1] - vol[0], rtol=1e-6, atol=1e-6)     
    np.testing.assert_allclose(x[3], vol[2] - vol[0], rtol=1e-6, atol=1e-6)     


def test_kinetic_c2_pads_last_channel_with_first_diff():
    vol = _vol(2)
    x = select_channels(vol, "kinetic")
    np.testing.assert_allclose(x[0], vol[0], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(x[1], vol[1], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(x[2], vol[1] - vol[0], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(x[3], vol[1] - vol[0], rtol=1e-6, atol=1e-6)  


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
