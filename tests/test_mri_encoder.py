
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
        MriEncoder(depth=50)  


def test_medicalnet_missing_weights_is_graceful_noop():
    with pytest.warns(UserWarning):
        MriEncoder(in_channels=4, medicalnet_weights="does_not_exist.pth")  


def test_medicalnet_loads_matching_keys_drops_conv1(tmp_path):
    src = MriEncoder(in_channels=1, depth=18)          
    ckpt = src.backbone.state_dict()
    ckpt_path = tmp_path / "fake_medicalnet_1ch.pth"
    torch.save(ckpt, ckpt_path)
    assert ckpt["conv1.weight"].shape[1] == 1          

    with pytest.warns(UserWarning, match=r"1 missing / 0 unexpected"):
        dst = MriEncoder(in_channels=4, depth=18, medicalnet_weights=ckpt_path)  

    dst_sd = dst.backbone.state_dict()
    assert dst_sd["conv1.weight"].shape[1] == 4
    loaded_key = next(
        k for k, v in ckpt.items()
        if k != "conv1.weight" and k in dst_sd and v.shape == dst_sd[k].shape and v.numel() > 1
    )
    assert torch.equal(dst_sd[loaded_key], ckpt[loaded_key]), f"{loaded_key} was not loaded verbatim"


@pytest.mark.skipif(not _CACHED.exists(), reason="cached P03 volume not present (data/ gitignored)")
def test_accepts_cached_p03_volume():
    vol = np.load(_CACHED)  
    assert vol.shape[0] == 4
    x = torch.from_numpy(vol).unsqueeze(0).float()
    enc = MriEncoder(in_channels=vol.shape[0]).eval()
    with torch.no_grad():
        out = enc(x)
    assert out.shape == (1, EMBED_DIM)
