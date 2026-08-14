"""P08 eval harness on the synthetic fixture: CIs on every metric, byte-stable metrics.json."""

import pytest

pytest.importorskip("scipy", reason="scipy is an ml/arms/trackb-extra dep — skip cleanly under a base install; pinksight.eval imports scipy")

import numpy as np
from fixtures.synthetic import classification_fixture, ki67_fixture, logits_fixture

from pinksight.eval.ablation import LADDER, build_metrics_json
from pinksight.eval.calibration import (
    apply_temperature,
    calibration_report,
    fit_temperature,
)
from pinksight.eval.metrics import classification_metrics, ki67_metrics


def test_every_classification_metric_has_a_ci():
    fx = classification_fixture()
    s = sorted(fx["y_true"])[0]
    m = classification_metrics(fx["y_true"][s], fx["probs"]["cross_attn"][s])
    for name in ("auroc", "pr_auc", "mcc", "f1", "sensitivity", "specificity"):
        lo, hi = m[name]["ci95"]
        assert lo <= m[name]["value"] <= hi, f"{name} value outside its CI"


def test_ki67_metrics_have_cis_and_correlate():
    k = ki67_fixture()
    m = ki67_metrics(k["y_true"], k["pred"], k["thresh"])
    assert m["pearson_r"]["value"] >= 0.40  # LOCK-6 minimum, by construction of the fixture
    for name in ("mae", "pearson_r", "spearman_rho"):
        lo, hi = m[name]["ci95"]
        assert lo <= m[name]["value"] <= hi


def test_ablation_ladder_is_ordered(tmp_path):
    fx = classification_fixture()
    doc = build_metrics_json(fx, ki67_fixture(), tmp_path, figures=False)
    aucs = [doc["ablation_ladder"][r]["auroc"]["value"] for r in LADDER]
    assert aucs == sorted(aucs), f"ladder AUROC not monotone: {aucs}"
    assert doc["delta_auroc_crossattn_vs_unimodal"] > 0


def test_classification_metrics_single_class_degrades_not_raises():
    # TICKET-012: a single-class OOF fold (all Luminal or all TNBC) has no measurable
    # AUROC. classification_metrics must return NaN AUROC instead of raising, so the
    # ablation call path surfaces a clear degraded result rather than an opaque crash.
    for y in (np.zeros(12, int), np.ones(12, int)):
        p = np.random.default_rng(0).random(12)
        m = classification_metrics(y, p)  # must not raise
        assert np.isnan(m["auroc"]["value"])


def test_aggregate_over_mixed_folds_degrades_gracefully():
    # TICKET-012: _aggregate must pool the folds that DO have both classes and mark the
    # rung degraded — one single-class (NaN AUROC) seed must not crash or NaN out the whole rung.
    from pinksight.eval.ablation import _aggregate

    rng = np.random.default_rng(1)
    y_both = np.r_[np.zeros(20), np.ones(20)].astype(int)
    both = classification_metrics(y_both, y_both + rng.normal(0, 0.6, 40))
    single = classification_metrics(np.zeros(40, int), rng.random(40))
    agg = _aggregate({0: both, 1: single, 2: both})

    assert agg.get("degraded") is True
    assert "1/3" in agg["note"] and "single-class" in agg["note"]  # one degraded seed of three
    assert np.isfinite(agg["auroc"]["value"])  # pooled from the two both-class seeds
    assert np.isnan(single["auroc"]["value"])  # the degenerate seed contributed NaN


def test_metrics_json_regenerates_byte_stable(tmp_path):
    fx, k = classification_fixture(), ki67_fixture()
    a = build_metrics_json(fx, k, tmp_path / "a", figures=False)
    b = build_metrics_json(fx, k, tmp_path / "b", figures=False)
    assert (tmp_path / "a" / "metrics.json").read_bytes() == (tmp_path / "b" / "metrics.json").read_bytes()
    assert a == b


def test_temperature_scaling_fits_on_val_only():
    lg = logits_fixture()
    t = fit_temperature(lg["val"]["logits"], lg["val"]["labels"])
    assert t > 0
    rep = calibration_report(lg["val"]["logits"], lg["val"]["labels"],
                             lg["test"]["logits"], lg["test"]["labels"])
    assert rep["fit_on"] == "val_only"
    # applying the fitted T must not degrade calibration on held-out test
    assert rep["ece_after"] <= rep["ece_before"] + 1e-6


def test_apply_temperature_softens_confidence():
    logits = np.array([4.0, -4.0, 2.0])
    hot = np.abs(apply_temperature(logits, 3.0) - 0.5)
    cold = np.abs(apply_temperature(logits, 1.0) - 0.5)
    assert np.all(hot < cold)  # T>1 pulls probabilities toward 0.5
