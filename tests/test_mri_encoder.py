"""MRI encoder CPU smoke test (Tier 1, skip-if-no-torch): forward shape + cached-volume integration.

Proves the O-1 3D-ResNet-18 encoder builds and runs on CPU and accepts P03's (4, D, H, W) stack.
No GPU (LOCK-5: GPU only for headline runs), no training, no number beyond the shape contract.
"""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("monai")

from pinksight.models.mri_encoder import EMBED_DIM, MriEncoder

_CACHED = Path("data/processed/Breast_MRI_042.npy")


@pytest.mark.parametrize("depth", [18, 34])
def test_forward_shape_cpu(depth):
    enc = MriEncoder(in_channels=4, depth=depth).eval()
    x = torch.randn(1, 4, 32, 32, 32)
    with torch.no_grad():
        out = enc(x)
    assert out.shape == (1, EMBED_DIM)


def test_rejects_unsupported_depth():
    with pytest.raises(ValueError):
        MriEncoder(depth=50)  # O-1 scopes the choice to 3D-ResNet-18/34


def test_medicalnet_missing_weights_is_graceful_noop():
    with pytest.warns(UserWarning):
        MriEncoder(in_channels=4, medicalnet_weights="does_not_exist.pth")  # random init, no crash


def test_medicalnet_loads_matching_keys_drops_conv1(tmp_path):
    """The 1->4 channel adaptation works mechanically (fastmri-medicalnet-backbone plan, Section 6).

    Build a synthetic 'pretrained' checkpoint from a 1-channel MriEncoder's own backbone.state_dict(),
    load it into a 4-channel MriEncoder, and prove the real _load_medicalnet loader:
      (a) DROPS conv1.weight (1-ch checkpoint vs 4-ch model -> shape mismatch -> stays 4-ch random init),
          surfaced as exactly '1 missing / 0 unexpected' (the value Section 2 decision 4 pre-registers), and
      (b) LOADS a genuine backbone key byte-equal (loader is NOT a silent no-op / not dropping everything).
    CPU-only: state_dict ops, no forward, no GPU (LOCK-5).
    """
    src = MriEncoder(in_channels=1, depth=18)          # the synthetic 'pretrained' source (1-channel)
    ckpt = src.backbone.state_dict()
    ckpt_path = tmp_path / "fake_medicalnet_1ch.pth"
    torch.save(ckpt, ckpt_path)
    assert ckpt["conv1.weight"].shape[1] == 1          # source first conv is 1-channel

    # real loader runs inside __init__; on a present file it warns exactly the missing/unexpected counts.
    with pytest.warns(UserWarning, match=r"1 missing / 0 unexpected"):
        dst = MriEncoder(in_channels=4, depth=18, medicalnet_weights=ckpt_path)  # 4-channel target

    dst_sd = dst.backbone.state_dict()
    # (a) conv1.weight was DROPPED: the 4-channel model kept its own (64, 4, k, k, k) init, not the
    #     checkpoint's 1-channel conv — i.e. conv1.weight is in the load's `missing` set.
    assert dst_sd["conv1.weight"].shape[1] == 4
    # (b) a genuine backbone key loaded byte-equal — pick one dynamically so this is MONAI-version robust.
    loaded_key = next(
        k for k, v in ckpt.items()
        if k != "conv1.weight" and k in dst_sd and v.shape == dst_sd[k].shape and v.numel() > 1
    )
    assert torch.equal(dst_sd[loaded_key], ckpt[loaded_key]), f"{loaded_key} was not loaded verbatim"


@pytest.mark.skipif(not _CACHED.exists(), reason="cached P03 volume not present (data/ gitignored)")
def test_accepts_cached_p03_volume():
    vol = np.load(_CACHED)  # (4, 73, 46, 62) from preprocess_patient
    assert vol.shape[0] == 4
    x = torch.from_numpy(vol).unsqueeze(0).float()
    enc = MriEncoder(in_channels=vol.shape[0]).eval()
    with torch.no_grad():
        out = enc(x)
    assert out.shape == (1, EMBED_DIM)
