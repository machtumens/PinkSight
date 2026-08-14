"""G3 #7 gate assertions for BiologyGatedMoE (deterministic strata gating, overfit guard, no router).

Model-agnostic CPU tests. Prove the gating is deterministic on KNOWN biology and cannot become a
free learned router at small N. The MoE-beats-baseline AUROC gate is a separate GPU run.
Leakage: strata_labels are INTEGER routing indices, never ER/PR float features (LOCK-2).
"""

import pytest

pytest.importorskip("torch", reason="torch is an ml/arms/trackb-extra dep — skip cleanly under a base install; G3 MoE tests build torch tensors")

import torch

from pinksight.models.fusion import BiologyGatedMoE


def test_moe_shape_and_gate():
    """Output shapes correct; expert weights sum to 1 per patient (a proper soft gate)."""
    moe = BiologyGatedMoE(fused_dim=128, n_experts=2)
    rep = torch.randn(4, 128)
    strata = torch.tensor([0, 1, 0, 1])
    out = moe(rep, strata)
    assert out["subtype_logit"].shape == (4, 1)
    assert out["expert_weights"].shape == (4, 2)
    row_sums = out["expert_weights"].sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(4), atol=1e-5), "expert weights must sum to 1 per patient"


def test_moe_hard_gating_limit():
    """With the gate_min clamp, the pre-assigned expert dominates each patient's mixture.

    The soft gate is initialised near-zero, so at init the gate is ~uniform; the gate_min=0.8 clamp
    must push each patient's assigned expert weight to at least gate_min (the biological strata
    assignment is honoured, not a free router)."""
    moe = BiologyGatedMoE(fused_dim=128, n_experts=2, strata_init_method="hr_status", gate_min=0.8)
    rep = torch.randn(6, 128)
    strata = torch.tensor([0, 0, 0, 1, 1, 1])  # HR-pos=0, HR-neg=1
    out = moe(rep, strata)
    w = out["expert_weights"]
    # each patient's assigned expert must carry >= gate_min of the mixture mass.
    assigned_weight = w[torch.arange(6), strata]
    assert (assigned_weight >= 0.8 - 1e-4).all(), f"gate_min not enforced: {assigned_weight.tolist()}"


def test_moe_train_val_gap_is_reported():
    """The training loop must emit a train_val_gap float in [0,1] in its output JSON.

    Asserts the CONTRACT (key present + float in range) via train_g3_moe.make_gap_entry — the single
    source of truth for the E-4 formula mean(train_auroc) - mean(val_auroc). GPU-scale statistical
    validity of the gap is a known-gap (accepted concern #1 in the validate-contract); this proves
    only the key-presence + range contract."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from train_g3_moe import make_gap_entry  # scripts/ on sys.path, not a package

    entry = make_gap_entry(train_auroc_per_fold=[0.80, 0.82], val_auroc_per_fold=[0.71, 0.73])
    assert "value" in entry
    gap = entry["value"]
    assert isinstance(gap, float)
    assert 0.0 <= gap <= 1.0, f"train_val_gap {gap} out of [0,1]"
    # E-4 formula: mean(train) - mean(val) = 0.81 - 0.72 = 0.09
    assert abs(gap - 0.09) < 1e-9, f"gap formula wrong: {gap}"


def test_moe_no_learned_router():
    """strata_init_method must be a KNOWN biological stratum — a free/unconstrained router is banned."""
    with pytest.raises(ValueError, match="learned router banned|not allowed"):
        BiologyGatedMoE(fused_dim=128, n_experts=2, strata_init_method="learned")
    # allowed methods construct fine.
    BiologyGatedMoE(fused_dim=128, n_experts=2, strata_init_method="hr_status")
    BiologyGatedMoE(fused_dim=128, n_experts=2, strata_init_method="grade_band")


def test_moe_strata_not_in_forbidden_features():
    """strata_labels are integer routing tensors (dtype long/int), NOT clinical feature float values.

    LOCK-2: routing HR-status as an integer index is safe; passing an ER/PR FLOAT value as the
    strata tensor must be rejected so a forbidden feature can never sneak in as a routing signal."""
    moe = BiologyGatedMoE(fused_dim=128, n_experts=2)
    rep = torch.randn(4, 128)
    # integer strata: accepted.
    good = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    assert good.dtype in (torch.long, torch.int, torch.int32, torch.int64)
    moe(rep, good)
    # float strata (e.g. ER/PR values): rejected.
    bad = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float32)
    with pytest.raises(ValueError, match="integer routing"):
        moe(rep, bad)
