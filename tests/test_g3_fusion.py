"""G3 #4 gate assertions for HierarchicalStagedFusion (shape, gradient flow, leakage, Kendall).

Model-agnostic CPU tests on random tensors — no trained model, no GPU. These prove the architecture
is wired correctly (the H-G3-A AUROC gate is a separate GPU run). Claim-ledger: subtype + grade
characterisation at diagnosis only.
"""

import pytest

pytest.importorskip("torch", reason="torch is an ml/arms/trackb-extra dep — skip cleanly under a base install; G3 fusion tests build torch tensors")

import torch

from pinksight.models.fusion import HierarchicalStagedFusion


def test_hierarchical_fusion_forward_shape():
    """Output dict matches the FusionModel contract: subtype/grade logits (N,1), fused (N,fused_dim)."""
    m = HierarchicalStagedFusion({"mri": 512, "clinical": 128}, fused_dim=128)
    out = m({"mri": torch.randn(4, 512), "clinical": torch.randn(4, 128)})
    assert out["subtype_logit"].shape == (4, 1)
    assert out["grade_logit"].shape == (4, 1)
    assert out["fused"].shape == (4, 128)
    assert m.embed_dim == 128


def test_hierarchical_fusion_clinical_strong():
    """With MRI dropped, clinical-only forward still returns valid logits (graceful degradation).

    This is the structural firewall from H6: clinical enters LATE and STRONG, so the model must
    produce a sane subtype call from clinical alone (clinical is the sole significant modality)."""
    m = HierarchicalStagedFusion({"mri": 512, "clinical": 128}, fused_dim=128)
    out = m({"mri": torch.randn(3, 512), "clinical": torch.randn(3, 128)}, drop={"mri"})
    assert out["subtype_logit"].shape == (3, 1)
    assert torch.isfinite(out["subtype_logit"]).all()
    assert torch.isfinite(out["fused"]).all()


def test_hierarchical_fusion_gradient_flow():
    """Gradients reach BOTH the stage-1 (imaging) and stage-3 (clinical) params — no dead stage."""
    m = HierarchicalStagedFusion({"mri": 512, "clinical": 128}, fused_dim=128)
    m.train()
    feats = {"mri": torch.randn(6, 512), "clinical": torch.randn(6, 128)}
    y = torch.randint(0, 2, (6,)).float()
    out = m(feats, drop=set())  # no random dropout so both stages are exercised
    total, _ = m.joint_loss(out, y)
    total.backward()
    # Stage 1 imaging projection must receive gradient.
    stage1_grad = m.stage1_proj[0].weight.grad
    assert stage1_grad is not None and stage1_grad.abs().sum() > 0, "stage-1 (imaging) is a dead stage"
    # Stage 3 clinical projection must receive gradient.
    clin_grad = m.clinical_proj[0].weight.grad
    assert clin_grad is not None and clin_grad.abs().sum() > 0, "stage-3 (clinical) is a dead stage"


def test_hierarchical_fusion_no_forbidden_features_in_input():
    """LOCK-2 leakage guard fires if a FORBIDDEN feature is passed as a fusion modality."""

    m = HierarchicalStagedFusion({"mri": 512, "clinical": 128}, fused_dim=128)
    bad = {"mri": torch.randn(2, 512), "clinical": torch.randn(2, 128), "ER": torch.randn(2, 4)}
    with pytest.raises(ValueError, match="LEAKAGE"):
        m(bad)


def test_hierarchical_fusion_kendall_n_tasks():
    """KendallUncertaintyWeighting(n_tasks=2) combines subtype+grade without overflow/NaN."""
    m = HierarchicalStagedFusion({"mri": 512, "clinical": 128}, fused_dim=128)
    m.train()
    feats = {"mri": torch.randn(5, 512), "clinical": torch.randn(5, 128)}
    y_sub = torch.randint(0, 2, (5,)).float()
    y_grade = torch.randint(0, 2, (5,)).float()
    out = m(feats, drop=set())
    total, weights = m.joint_loss(out, y_sub, y_grade)
    assert torch.isfinite(total), "Kendall total is NaN/inf"
    assert weights.numel() == 2, "expected 2 task weights (subtype + grade)"
    assert torch.isfinite(weights).all()
