"""G3 #3 validity gate for counterfactual grade XAI (HiResCAM mode, flip, claim-ledger, grade-only).

CPU-fixture tests — no trained model, no GPU. Prove the XAI primitives are wired correctly and the
claim-ledger wording guard + grade-head-only constraint are enforced. Spatial IoU/pointing on real
MRI is a separate GPU gate (Step 3.6). Claim-ledger: grade characterisation, never growth-rate.
"""

import pytest

pytest.importorskip("torch", reason="torch is an ml/arms/trackb-extra dep — skip cleanly under a base install; counterfactual XAI tests use torch")

import numpy as np
import torch
from fixtures.synthetic import tiny_cnn_fixture, tiny_cnn_grade_fixture

from pinksight.xai.saliency import counterfactual_grade_map, grad_cam_3d


def test_hirescam_mode_changes_cam():
    """mode='hirescam' produces a DIFFERENT (element-wise) cam than mode='grad_cam' on the same input."""
    model, layer, vol = tiny_cnn_fixture()
    cam_gc = grad_cam_3d(model, vol, layer, mode="grad_cam")
    cam_hr = grad_cam_3d(model, vol, layer, mode="hirescam")
    assert cam_gc.shape == cam_hr.shape == tuple(vol.shape[2:])
    # The two methods differ (GAP vs per-position); they must not be numerically identical.
    assert not np.allclose(cam_gc, cam_hr), "HiResCAM must differ from Grad-CAM"


def test_hirescam_backward_compat():
    """Calling grad_cam_3d(model, vol, layer) with no mode arg still works (default grad_cam)."""
    model, layer, vol = tiny_cnn_fixture()
    cam = grad_cam_3d(model, vol, layer)  # no mode kwarg — backward compatible
    assert cam.shape == tuple(vol.shape[2:])
    assert 0.0 <= cam.min() and cam.max() <= 1.0
    # invalid mode is rejected.
    with pytest.raises(ValueError, match="mode must be"):
        grad_cam_3d(model, vol, layer, mode="bogus")


def test_counterfactual_grade_flip_achieved():
    """The counterfactual perturbation flips the grade output for the synthetic model (>=1 of a few)."""
    fx = tiny_cnn_grade_fixture()
    flips = []
    rng = np.random.default_rng(0)
    for i in range(3):
        v = fx["volume"] + torch.from_numpy(rng.normal(0, 0.01, fx["volume"].shape)).float()
        res = counterfactual_grade_map(fx["model"], v, fx["grade_head"], fx["target_layer"],
                                       target_flip=i % 2, n_steps=150, lr=0.05)
        flips.append(res["flip_achieved"])
    assert sum(flips) >= 1, f"expected >=1/3 flips, got {sum(flips)}"


def test_counterfactual_claim_ledger_docstring():
    """The docstring MUST contain the claim-ledger guard string 'NOT a growth-rate' (LOCK-1)."""
    doc = counterfactual_grade_map.__doc__ or ""
    assert "NOT a growth-rate" in doc, "claim-ledger wording guard missing from docstring"
    # The exact plan-mandated phrasing.
    assert "NOT a growth-rate / kinetics claim" in doc


def test_counterfactual_not_on_subtype_head():
    """counterfactual_grade_map raises if handed a subtype head (grade-head-only, E-6)."""
    fx = tiny_cnn_grade_fixture()
    subtype_head = fx["subtype_head"]  # nn.Linear with debug_name='subtype_head'
    with pytest.raises(ValueError, match="grade-head only"):
        counterfactual_grade_map(fx["model"], fx["volume"], subtype_head, fx["target_layer"],
                                 n_steps=5)


def test_counterfactual_spearman_is_computed():
    """The output dict exposes a spearman_rho key (spatial agreement metric)."""
    fx = tiny_cnn_grade_fixture()
    res = counterfactual_grade_map(fx["model"], fx["volume"], fx["grade_head"], fx["target_layer"],
                                   n_steps=20)
    assert "spearman_rho" in res
    assert isinstance(res["spearman_rho"], float)
    for key in ("cam_original", "cam_counterfactual", "flip_achieved", "delta_logit"):
        assert key in res, f"missing output key {key}"
