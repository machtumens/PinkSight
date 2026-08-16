
from __future__ import annotations

import pytest

pytest.importorskip("torch", reason="torch is an ml/arms/trackb-extra dep — skip cleanly under a base install; lesion-crop tests use torch ops")

from pathlib import Path

import numpy as np
import torch

from pinksight.data.lesion_crop import (
    CROP_SIZE_DEFAULT,
    derive_lesion_box,
    lesion_crop,
)

PROC = Path("data/processed")


def _vol(c: int = 2, spatial=(40, 30, 50)) -> np.ndarray:
    r, cc, s = spatial
    base = np.arange(r * cc * s, dtype=np.float32).reshape(r, cc, s)
    return np.stack([base + 1000.0 * ch for ch in range(c)], axis=0)


def _mask_at(spatial, rr, cr, sr) -> np.ndarray:
    m = np.zeros(spatial, dtype=bool)
    m[rr[0]:rr[1], cr[0]:cr[1], sr[0]:sr[1]] = True
    return m


def test_derive_box_from_predicted_mask_is_tight_plus_rim():
    spatial = (40, 30, 50)
    mask = _mask_at(spatial, (15, 20), (12, 16), (25, 31))  
    box, used_fallback = derive_lesion_box(mask, spatial, rim_mm=7)
    assert used_fallback is False  
    assert box.row == (max(15 - 7, 0), min(20 + 7, 40))
    assert box.col == (max(12 - 7, 0), min(16 + 7, 30))
    assert box.slice == (max(25 - 7, 0), min(31 + 7, 50))


def test_empty_mask_triggers_dukebox_fallback_not_empty():
    spatial = (40, 30, 50)
    box, used_fallback = derive_lesion_box(np.zeros(spatial, dtype=bool), spatial, rim_mm=7)
    assert used_fallback is True  
    assert box.row == (0, 40) and box.col == (0, 30) and box.slice == (0, 50)  


def test_none_mask_triggers_dukebox_fallback():
    spatial = (40, 30, 50)
    box, used_fallback = derive_lesion_box(None, spatial, rim_mm=7)
    assert used_fallback is True and box.row == (0, 40)  


def test_mask_grid_mismatch_rejected():
    with pytest.raises(ValueError):  
        derive_lesion_box(np.zeros((10, 10, 10), dtype=bool), (40, 30, 50))


def test_lesion_crop_output_is_exactly_96_cube():
    x = lesion_crop(_vol(c=2, spatial=(40, 30, 50)), _mask_at((40, 30, 50), (15, 20), (12, 16), (25, 31)))
    assert x.shape == (2, CROP_SIZE_DEFAULT, CROP_SIZE_DEFAULT, CROP_SIZE_DEFAULT)  
    assert x.dtype == np.float32 and np.isfinite(x).all()


def test_lesion_crop_honours_non_default_size():
    x = lesion_crop(_vol(c=1, spatial=(40, 30, 50)), None, out_size=64)  
    assert x.shape == (1, 64, 64, 64)


def test_lesion_crop_derives_from_predicted_mask_not_whole_volume():
    spatial = (40, 30, 50)
    vol = _vol(c=1, spatial=spatial)
    tight = lesion_crop(vol, _mask_at(spatial, (2, 5), (2, 5), (2, 5)), rim_mm=0)  
    whole = lesion_crop(vol, None)  
    assert tight.shape == whole.shape == (1, 96, 96, 96)
    assert not np.allclose(tight, whole)  


def test_lesion_crop_empty_mask_is_nonempty_cube():
    x = lesion_crop(_vol(c=2, spatial=(40, 30, 50)), np.zeros((40, 30, 50), dtype=bool))
    assert x.shape == (2, 96, 96, 96) and x.size > 0  


def test_lesion_crop_rejects_non_4d():
    with pytest.raises(ValueError):
        lesion_crop(np.zeros((40, 30, 50), dtype=np.float32))  


def test_dataset_grid_mismatched_mask_falls_back_not_crash(tmp_path):
    from pinksight.data.dataset import NpyVolumeDataset

    proc = tmp_path / "processed"
    proc.mkdir()
    vol = _vol(c=2, spatial=(40, 30, 50))
    np.save(proc / "P1.npy", vol)
    masks = tmp_path / "processed_masks"
    masks.mkdir()
    np.save(masks / "P1.npy", np.ones((20, 20, 20), dtype=bool))  

    ds = NpyVolumeDataset([("P1", 0)], proc, channels="pre_post", crop_mode="lesion",
                          crop_size=96, mask_dir=masks)
    x, _, _ = ds[0]  
    assert x.shape == (2, 96, 96, 96) and torch.isfinite(x).all()  


def test_dataset_absent_mask_uses_fallback(tmp_path):
    from pinksight.data.dataset import NpyVolumeDataset

    proc = tmp_path / "processed"
    proc.mkdir()
    np.save(proc / "P1.npy", _vol(c=1, spatial=(40, 30, 50)))
    ds = NpyVolumeDataset([("P1", 0)], proc, channels="first_post", crop_mode="lesion",
                          crop_size=96, mask_dir=tmp_path / "no_such_dir")
    x, _, _ = ds[0]  
    assert x.shape == (1, 96, 96, 96)


def test_dataset_matching_mask_drives_crop(tmp_path):
    import torch as _t

    from pinksight.data.dataset import NpyVolumeDataset

    proc = tmp_path / "processed"
    proc.mkdir()
    spatial = (40, 30, 50)
    np.save(proc / "P1.npy", _vol(c=1, spatial=spatial))
    masks = tmp_path / "processed_masks"
    masks.mkdir()
    np.save(masks / "P1.npy", _mask_at(spatial, (3, 7), (3, 7), (3, 7)))  

    ds_pred = NpyVolumeDataset([("P1", 0)], proc, channels="first_post", crop_mode="lesion",
                               crop_size=96, mask_dir=masks)
    ds_fb = NpyVolumeDataset([("P1", 0)], proc, channels="first_post", crop_mode="lesion",
                             crop_size=96, mask_dir=tmp_path / "nope")
    assert not _t.allclose(ds_pred[0][0], ds_fb[0][0])  


def test_dataset_crop_none_is_unchanged(tmp_path):
    from pinksight.data.dataset import NpyVolumeDataset

    proc = tmp_path / "processed"
    proc.mkdir()
    np.save(proc / "P1.npy", _vol(c=2, spatial=(40, 30, 50)))
    ds_default = NpyVolumeDataset([("P1", 0)], proc, channels="pre_post", spatial_size=(96, 96, 96))
    ds_explicit_none = NpyVolumeDataset([("P1", 0)], proc, channels="pre_post",
                                        spatial_size=(96, 96, 96), crop_mode="none")
    x0, _, _ = ds_default[0]
    x1, _, _ = ds_explicit_none[0]
    assert x0.shape == (2, 96, 96, 96)
    assert torch.allclose(x0, x1)  


def test_lesion_crop_on_real_cached_volume():
    npys = sorted(PROC.glob("*.npy"))
    if not npys:
        pytest.skip("no cached data/processed/*.npy — synthetic box-math tests cover the logic")
    vol = np.load(npys[0])  
    assert vol.ndim == 4, f"cached volume unexpected shape {vol.shape}"
    spatial = (vol.shape[1], vol.shape[2], vol.shape[3])
    r, c, s = spatial
    mask = _mask_at(spatial, (r // 3, 2 * r // 3), (c // 3, 2 * c // 3), (s // 3, 2 * s // 3))
    x_pred = lesion_crop(vol, mask)
    assert x_pred.shape == (vol.shape[0], 96, 96, 96) and np.isfinite(x_pred).all()
    assert float((x_pred != 0).mean()) > 0.0  
    x_fb = lesion_crop(vol, None)
    assert x_fb.shape == (vol.shape[0], 96, 96, 96) and float((x_fb != 0).mean()) > 0.0


def _run_as_script() -> int:
    spatial = (40, 30, 50)
    x = lesion_crop(_vol(2, spatial), _mask_at(spatial, (15, 20), (12, 16), (25, 31)))
    assert x.shape == (2, 96, 96, 96), x.shape
    box, fb = derive_lesion_box(_mask_at(spatial, (15, 20), (12, 16), (25, 31)), spatial, rim_mm=7)
    assert fb is False and box.row == (8, 27), box
    box2, fb2 = derive_lesion_box(np.zeros(spatial, dtype=bool), spatial)
    assert fb2 is True and box2.row == (0, 40), box2
    xe = lesion_crop(_vol(1, spatial), np.zeros(spatial, dtype=bool))
    assert xe.shape == (1, 96, 96, 96) and xe.size > 0
    npys = sorted(PROC.glob("*.npy"))
    if npys:
        vol = np.load(npys[0])
        r, c, s = vol.shape[1:]
        m = _mask_at((r, c, s), (r // 3, 2 * r // 3), (c // 3, 2 * c // 3), (s // 3, 2 * s // 3))
        xr = lesion_crop(vol, m)
        assert xr.shape == (vol.shape[0], 96, 96, 96) and float((xr != 0).mean()) > 0.0
        print(f"OK real: {npys[0].name} {tuple(vol.shape)} -> {xr.shape} nonzero {float((xr!=0).mean()):.3f}")
    else:
        print("OK synthetic (no cached cohort on disk)")
    print("LESION-CROP SELF-CHECK PASS: 96^3 fixed cube | predicted-mask box | [1.6] fallback non-empty")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(_run_as_script())
