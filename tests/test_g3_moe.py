
import pytest

pytest.importorskip("torch", reason="torch is an ml/arms/trackb-extra dep — skip cleanly under a base install; G3 MoE tests build torch tensors")

import torch

from pinksight.models.fusion import BiologyGatedMoE


def test_moe_shape_and_gate():
    moe = BiologyGatedMoE(fused_dim=128, n_experts=2)
    rep = torch.randn(4, 128)
    strata = torch.tensor([0, 1, 0, 1])
    out = moe(rep, strata)
    assert out["subtype_logit"].shape == (4, 1)
    assert out["expert_weights"].shape == (4, 2)
    row_sums = out["expert_weights"].sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(4), atol=1e-5), "expert weights must sum to 1 per patient"


def test_moe_hard_gating_limit():
    moe = BiologyGatedMoE(fused_dim=128, n_experts=2, strata_init_method="hr_status", gate_min=0.8)
    rep = torch.randn(6, 128)
    strata = torch.tensor([0, 0, 0, 1, 1, 1])  
    out = moe(rep, strata)
    w = out["expert_weights"]
    assigned_weight = w[torch.arange(6), strata]
    assert (assigned_weight >= 0.8 - 1e-4).all(), f"gate_min not enforced: {assigned_weight.tolist()}"


def test_moe_train_val_gap_is_reported():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from train_g3_moe import make_gap_entry  

    entry = make_gap_entry(train_auroc_per_fold=[0.80, 0.82], val_auroc_per_fold=[0.71, 0.73])
    assert "value" in entry
    gap = entry["value"]
    assert isinstance(gap, float)
    assert 0.0 <= gap <= 1.0, f"train_val_gap {gap} out of [0,1]"
    assert abs(gap - 0.09) < 1e-9, f"gap formula wrong: {gap}"


def test_moe_no_learned_router():
    with pytest.raises(ValueError, match="learned router banned|not allowed"):
        BiologyGatedMoE(fused_dim=128, n_experts=2, strata_init_method="learned")
    BiologyGatedMoE(fused_dim=128, n_experts=2, strata_init_method="hr_status")
    BiologyGatedMoE(fused_dim=128, n_experts=2, strata_init_method="grade_band")


def test_moe_strata_not_in_forbidden_features():
    moe = BiologyGatedMoE(fused_dim=128, n_experts=2)
    rep = torch.randn(4, 128)
    good = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    assert good.dtype in (torch.long, torch.int, torch.int32, torch.int64)
    moe(rep, good)
    bad = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float32)
    with pytest.raises(ValueError, match="integer routing"):
        moe(rep, bad)
