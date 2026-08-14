"""A1-10 unit test: PhysioEncoder forward-pass shape + BatchNorm + numerical soundness.

Proves the parallel physiology encoder (7 semi-quantitative DCE features -> 512-d bus embedding) is
architecturally correct in isolation — no data, no training, CPU-only (LOCK-5). Skips cleanly if
torch is absent.

Cases (plan A1-10):
  (a) forward([4, 7]) -> [4, out_dim]
  (b) BatchNorm tracks running stats and is stable in eval mode
  (c) no NaN in output for all-zero input
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pinksight.models import PhysioEncoder
from pinksight.models.mri_encoder import EMBED_DIM
from pinksight.models.physio_encoder import N_PHYSIO_FEATURES


def test_forward_shape_matches_bus_dim():
    """(4, 7) -> (4, EMBED_DIM) — the physio arm writes the same bus as the MRI encoder (E2=512)."""
    enc = PhysioEncoder().train()
    x = torch.randn(4, N_PHYSIO_FEATURES)
    out = enc(x)
    assert out.shape == (4, EMBED_DIM)
    assert enc.out_dim == EMBED_DIM == 512


def test_custom_out_dim_respected():
    """A non-default out_dim (e.g. the plan's placeholder 256) is honoured."""
    enc = PhysioEncoder(out_dim=256).train()
    out = enc(torch.randn(8, N_PHYSIO_FEATURES))
    assert out.shape == (8, 256)


def test_batchnorm_tracks_running_stats_and_eval_is_stable():
    """BatchNorm running stats update during train() and eval() runs on a single sample without error."""
    enc = PhysioEncoder()
    # running mean starts at 0; after a train forward on non-trivial data it must move.
    bn0 = enc.net[1]  # first BatchNorm1d(hidden)
    assert bn0.running_mean is not None
    before = bn0.running_mean.clone()
    enc.train()
    enc(torch.randn(16, N_PHYSIO_FEATURES) * 3.0 + 1.0)
    assert not torch.allclose(bn0.running_mean, before), "BatchNorm running stats did not update in train()"

    # eval mode uses the tracked stats -> a single-sample forward must NOT raise (BN would in train).
    enc.eval()
    with torch.no_grad():
        out = enc(torch.randn(1, N_PHYSIO_FEATURES))
    assert out.shape == (1, EMBED_DIM)
    assert torch.isfinite(out).all()


def test_no_nan_on_all_zero_input():
    """All-zero input in eval mode must yield a finite (no NaN/inf) embedding."""
    enc = PhysioEncoder()
    # prime the running stats once so eval-mode BN has a defined variance, then test zeros.
    enc.train()
    enc(torch.randn(16, N_PHYSIO_FEATURES))
    enc.eval()
    with torch.no_grad():
        out = enc(torch.zeros(4, N_PHYSIO_FEATURES))
    assert out.shape == (4, EMBED_DIM)
    assert torch.isfinite(out).all(), "all-zero input produced non-finite output"


def test_rejects_wrong_feature_count():
    """A wrong feature-vector width is a hard error, not a silent broadcast."""
    enc = PhysioEncoder().eval()
    with pytest.raises(ValueError):
        enc(torch.randn(4, 5))  # 5 != 7
