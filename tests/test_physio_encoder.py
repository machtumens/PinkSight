
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pinksight.models import PhysioEncoder
from pinksight.models.mri_encoder import EMBED_DIM
from pinksight.models.physio_encoder import N_PHYSIO_FEATURES


def test_forward_shape_matches_bus_dim():
    enc = PhysioEncoder().train()
    x = torch.randn(4, N_PHYSIO_FEATURES)
    out = enc(x)
    assert out.shape == (4, EMBED_DIM)
    assert enc.out_dim == EMBED_DIM == 512


def test_custom_out_dim_respected():
    enc = PhysioEncoder(out_dim=256).train()
    out = enc(torch.randn(8, N_PHYSIO_FEATURES))
    assert out.shape == (8, 256)


def test_batchnorm_tracks_running_stats_and_eval_is_stable():
    enc = PhysioEncoder()
    bn0 = enc.net[1]  
    assert bn0.running_mean is not None
    before = bn0.running_mean.clone()
    enc.train()
    enc(torch.randn(16, N_PHYSIO_FEATURES) * 3.0 + 1.0)
    assert not torch.allclose(bn0.running_mean, before), "BatchNorm running stats did not update in train()"

    enc.eval()
    with torch.no_grad():
        out = enc(torch.randn(1, N_PHYSIO_FEATURES))
    assert out.shape == (1, EMBED_DIM)
    assert torch.isfinite(out).all()


def test_no_nan_on_all_zero_input():
    enc = PhysioEncoder()
    enc.train()
    enc(torch.randn(16, N_PHYSIO_FEATURES))
    enc.eval()
    with torch.no_grad():
        out = enc(torch.zeros(4, N_PHYSIO_FEATURES))
    assert out.shape == (4, EMBED_DIM)
    assert torch.isfinite(out).all(), "all-zero input produced non-finite output"


def test_rejects_wrong_feature_count():
    enc = PhysioEncoder().eval()
    with pytest.raises(ValueError):
        enc(torch.randn(4, 5))  
