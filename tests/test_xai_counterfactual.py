
import pytest

pytest.importorskip("torch", reason="torch is an ml/arms/trackb-extra dep — skip cleanly under a base install; counterfactual XAI tests use torch")

import numpy as np
import torch
from fixtures.synthetic import tiny_cnn_fixture, tiny_cnn_grade_fixture

from pinksight.xai.saliency import counterfactual_grade_map, grad_cam_3d


def test_hirescam_mode_changes_cam():
    model, layer, vol = tiny_cnn_fixture()
    cam_gc = grad_cam_3d(model, vol, layer, mode="grad_cam")
    cam_hr = grad_cam_3d(model, vol, layer, mode="hirescam")
    assert cam_gc.shape == cam_hr.shape == tuple(vol.shape[2:])
    assert not np.allclose(cam_gc, cam_hr), "HiResCAM must differ from Grad-CAM"


def test_hirescam_backward_compat():
    model, layer, vol = tiny_cnn_fixture()
    cam = grad_cam_3d(model, vol, layer)  
    assert cam.shape == tuple(vol.shape[2:])
    assert 0.0 <= cam.min() and cam.max() <= 1.0
    with pytest.raises(ValueError, match="mode must be"):
        grad_cam_3d(model, vol, layer, mode="bogus")


def test_counterfactual_grade_flip_achieved():
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
    doc = counterfactual_grade_map.__doc__ or ""
    assert "NOT a growth-rate" in doc, "claim-ledger wording guard missing from docstring"
    assert "NOT a growth-rate / kinetics claim" in doc


def test_counterfactual_not_on_subtype_head():
    fx = tiny_cnn_grade_fixture()
    subtype_head = fx["subtype_head"]  
    with pytest.raises(ValueError, match="grade-head only"):
        counterfactual_grade_map(fx["model"], fx["volume"], subtype_head, fx["target_layer"],
                                 n_steps=5)


def test_counterfactual_spearman_is_computed():
    fx = tiny_cnn_grade_fixture()
    res = counterfactual_grade_map(fx["model"], fx["volume"], fx["grade_head"], fx["target_layer"],
                                   n_steps=20)
    assert "spearman_rho" in res
    assert isinstance(res["spearman_rho"], float)
    for key in ("cam_original", "cam_counterfactual", "flip_achieved", "delta_logit"):
        assert key in res, f"missing output key {key}"
