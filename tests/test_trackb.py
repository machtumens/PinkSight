
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fixtures import trackb_synthetic as fx

from pinksight.trackb import gate as trackb_gate
from pinksight.trackb.gate import (
    TRACKB_GATE_OPEN,
    TrackBGateClosedError,
    assert_gate_open,
)
from pinksight.trackb.genomics import GenomicsEncoder
from pinksight.trackb.head import TrackBSubtypeHead
from pinksight.trackb.mil import GatedAttentionMIL
from pinksight.trackb.modality_dropout import ModalityDropoutFusion
from pinksight.trackb.tiles import (
    FOUNDATION_DIMS,
    SyntheticTileEncoder,
    load_tile_encoder,
)


def test_gate_is_open_after_ratification():
    assert TRACKB_GATE_OPEN is True


def test_assert_gate_open_passes_when_open():
    assert_gate_open()


def test_assert_gate_open_raises_when_closed(monkeypatch):
    monkeypatch.setattr(trackb_gate, "TRACKB_GATE_OPEN", False)
    with pytest.raises(TrackBGateClosedError, match="HARD-GATED"):
        assert_gate_open()


def test_real_tile_encoder_loads_uni2h_when_gate_open():
    from pinksight.trackb.tiles import UNITileEncoder

    enc = load_tile_encoder("uni", synthetic=False)
    assert isinstance(enc, UNITileEncoder)
    assert enc.name == "uni"
    assert enc.embed_dim == FOUNDATION_DIMS["uni"] == 1536


def test_real_tile_encoder_unwired_name_still_not_implemented():
    with pytest.raises(NotImplementedError):
        load_tile_encoder("conch", synthetic=False)


def test_real_tile_encoder_refuses_to_load_when_gate_closed(monkeypatch):
    monkeypatch.setattr(trackb_gate, "TRACKB_GATE_OPEN", False)
    with pytest.raises(TrackBGateClosedError):
        load_tile_encoder("uni", synthetic=False)


def test_synthetic_tile_encoder_shapes_and_determinism():
    enc = SyntheticTileEncoder("uni")
    assert enc.embed_dim == FOUNDATION_DIMS["uni"]
    bag = fx.bag_fixture(seed=0, n_tiles=16)
    emb1 = enc.encode(bag)
    emb2 = enc.encode(bag)
    assert emb1.shape == (16, enc.embed_dim)
    assert np.allclose(emb1, emb2), "frozen encoder must be deterministic"


def test_named_encoders_selectable():
    for name in ("uni", "conch", "ctranspath"):
        enc = load_tile_encoder(name, synthetic=True)
        assert enc.name == name and enc.embed_dim == FOUNDATION_DIMS[name]


def test_mil_attention_sums_to_one_and_bag_shape():
    mil = GatedAttentionMIL(in_dim=1024, out_dim=128)
    bag = torch.tensor(fx.bag_fixture(seed=1, n_tiles=32), dtype=torch.float32)
    emb, attn = mil(bag)
    assert emb.shape == (128,)
    assert attn.shape == (32,)
    assert torch.isclose(attn.sum(), torch.tensor(1.0), atol=1e-5), "attention must sum to 1"
    assert torch.all(attn >= 0)


def test_mil_gradient_flows_overfit_single_bag():
    torch.manual_seed(0)
    mil = GatedAttentionMIL(in_dim=64, out_dim=32)
    head = TrackBSubtypeHead(mil, embed_dim=32)
    bag = torch.randn(20, 64)
    target = torch.tensor([[1.0]])
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    losses = []
    for step in range(50):
        opt.zero_grad()
        logits, _attn = head(bag)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        loss.backward()
        if step == 0:
            assert mil.attn_w.weight.grad is not None
            assert mil.attn_w.weight.grad.abs().sum() > 0
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0], f"loss did not drop: {losses[0]:.3f} -> {losses[-1]:.3f}"


def test_genomics_encoder_shape():
    enc = GenomicsEncoder(in_dim=fx.GENO_DIM, embed_dim=128)
    data = fx.genomics_fixture(seed=0)
    x = torch.tensor(data["X"], dtype=torch.float32)
    out = enc(x)
    assert out.shape == (fx.N_PATIENTS, 128)
    assert enc.embed_dim == 128


def _feats():
    return {
        "wsi": torch.randn(4, 128),
        "genomics": torch.randn(4, 64),
        "mri_clinical": torch.randn(4, 256),  
    }


def test_modality_dropout_all_present():
    fusion = ModalityDropoutFusion({"wsi": 128, "genomics": 64, "mri_clinical": 256}, fused_dim=96)
    out = fusion(_feats())
    assert out.shape == (4, 96)


def test_modality_dropout_graceful_degradation():
    fusion = ModalityDropoutFusion({"wsi": 128, "genomics": 64, "mri_clinical": 256}, fused_dim=96)
    head = TrackBSubtypeHead(fusion, embed_dim=96)

    full = head(_feats())
    assert full.shape == (4, 1)

    dropped = head(_feats(), drop={"mri_clinical"})
    assert dropped.shape == (4, 1) and torch.isfinite(dropped).all()

    wsi_only = head({"wsi": torch.randn(4, 128)})
    assert wsi_only.shape == (4, 1) and torch.isfinite(wsi_only).all()


def test_modality_dropout_all_absent_raises():
    fusion = ModalityDropoutFusion({"wsi": 128, "genomics": 64}, fused_dim=96)
    with pytest.raises(ValueError, match="empty set"):
        fusion({"wsi": torch.randn(2, 128)}, drop={"wsi"})


def test_cohort_fixture_feeds_build_metrics_json(tmp_path):
    from pinksight.eval import build_metrics_json

    cohort = fx.cohort_fixture()
    ki67 = {"y_true": np.random.default_rng(0).uniform(0, 60, 120),
            "pred": np.random.default_rng(1).uniform(0, 60, 120), "thresh": 14.0}
    doc = build_metrics_json(cohort, ki67, tmp_path, figures=False)
    top = doc["ablation_ladder"]["cross_attn"]["auroc"]["value"]
    assert 0.5 <= top <= 1.0, "PoC subtype AUROC out of range"
    assert (tmp_path / "metrics.json").exists()
