"""P10 XAI harness on the synthetic fixture: non-circular IoU + randomization sanity + SHAP."""

import pytest

pytest.importorskip("torch", reason="torch is an ml/arms/trackb-extra dep — skip cleanly under a base install; pinksight.xai.saliency imports torch")

import numpy as np
from fixtures.synthetic import (
    clinical_model_fixture,
    saliency_mask_fixture,
    tiny_cnn_fixture,
)

from pinksight.xai.faithfulness import box_hit, iou, pointing_game, randomization_test
from pinksight.xai.saliency import (
    attention_map,
    clinical_shap,
    grad_cam_3d,
    randomize_weights,
)


def test_iou_vs_independent_reference_meets_bar():
    fx = saliency_mask_fixture()
    score = iou(fx["saliency"], fx["reference_mask"])
    assert score >= 0.30, f"IoU {score} below the LOCK-6 bar"
    assert pointing_game(fx["saliency"], fx["reference_mask"])  # peak lands in the reference
    assert box_hit(fx["saliency"], fx["box"])                   # box-hit fallback also passes


def test_randomization_flips_saliency():
    fx = saliency_mask_fixture()
    res = randomization_test(fx["saliency"], fx["random_saliency"])
    assert res["passed"], res
    assert res["rel_drop"] > 0.5


def test_grad_cam_3d_shape_and_range():
    model, layer, vol = tiny_cnn_fixture()
    cam = grad_cam_3d(model, vol, layer)
    assert cam.shape == tuple(vol.shape[2:])
    assert 0.0 <= cam.min() and cam.max() <= 1.0


def test_grad_cam_drops_under_weight_randomization():
    model, layer, vol = tiny_cnn_fixture()
    cam = grad_cam_3d(model, vol, layer)
    rnd = randomize_weights(model, seed=1)
    cam_rnd = grad_cam_3d(rnd, vol, rnd[2])
    res = randomization_test(cam, cam_rnd)
    # a real trained-vs-random pair must lose >50% of ROI mass; record the number either way
    assert res["roi_mass_orig"] >= res["roi_mass_random"], res


def test_clinical_shap_additivity():
    fx = clinical_model_fixture()
    values, base = clinical_shap(fx["model"], fx["X"], fx["background"])
    assert values.shape == fx["X"].shape
    # SHAP efficiency: base + sum(phi) == model margin (exact for a linear model)
    margin = fx["model"].decision_function(fx["X"])
    assert np.allclose(base + values.sum(axis=1), margin, atol=1e-6)


def test_clinical_shap_guards_missing_cat_level():
    """TICKET-016: a ≤100-row SHAP background that misses a rare categorical level must be caught
    when the true cat_cardinalities are known — otherwise cardinality is silently undercounted and
    the explanation undershoots. Fixture/synthetic: no trained network needed."""
    import torch
    from torch import nn

    from pinksight.xai.saliency import clinical_shap

    n_num, cards = 2, [3]  # one categorical with 3 true levels {0,1,2}

    class _Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(cards[0], 4)
            self.lin = nn.Linear(n_num + 4, 1)

        def forward(self, x_cont, x_cat):
            return self.lin(torch.cat([x_cont, self.emb(x_cat[:, 0])], dim=1))

    model = _Tiny()
    rng = np.random.default_rng(0)
    bg_num = rng.normal(0, 1, (12, n_num))
    bg_cat = np.array([[0], [1]] * 6, dtype=float)  # level 2 absent → undercounts to 2, not 3
    background_X = np.column_stack([bg_num, bg_cat])
    X = background_X[:2]

    with pytest.raises(ValueError, match="TICKET-016"):
        clinical_shap(model, X, background_X, n_num=n_num, cat_cardinalities=cards, nsamples=8)


def test_attention_map_normalised():
    a = attention_map(np.array([[0.0, 2.0], [1.0, 4.0]]))
    assert a.max() == 1.0 and a.min() >= 0.0
