"""P09 stats harness on the synthetic fixture: ΔAUC bootstrap CI + DeLong across folds AND seeds."""

import pytest

pytest.importorskip("scipy", reason="scipy is an ml/arms/trackb-extra dep — skip cleanly under a base install; pinksight.stats.compare imports scipy.stats")

from fixtures.synthetic import classification_fixture

from pinksight.stats.compare import paired_bootstrap_delta_auc, stats_report


def _best_unimodal_vs_fusion(fx):
    # best unimodal here is 'unimodal'; candidate is the cross-attn fusion rung
    return fx["y_true"], fx["probs"]["unimodal"], fx["probs"]["cross_attn"]


def test_stats_report_has_all_required_fields():
    y, a, b = _best_unimodal_vs_fusion(classification_fixture())
    rep = stats_report(y, a, b, boot=500)
    assert "bootstrap_delta_ci95_mean" in rep      # ΔAUC bootstrap CI
    assert "delong_combined_p" in rep              # DeLong aggregated across seeds
    assert "power_to_detect_margin" in rep and "mde_80pct_power" in rep  # power / MDE stated
    assert rep["n_seeds"] == 3                      # aggregated across seeds
    for s in rep["per_seed"]:                       # per-seed (per-fold pooled-OOF) DeLong present
        assert "z" in rep["per_seed"][s]["delong"]


def test_fusion_beats_unimodal_and_is_significant():
    y, a, b = _best_unimodal_vs_fusion(classification_fixture())
    rep = stats_report(y, a, b, boot=500)
    assert rep["delta_auroc_mean"] >= 0.03          # clears the pre-registered margin
    assert rep["bootstrap_delta_ci95_mean"][0] > 0  # CI excludes 0
    assert rep["delong_combined_p"] < 0.05
    assert rep["meets_prereg_margin"] is True


def test_paired_bootstrap_ci_brackets_delta():
    y, a, b = _best_unimodal_vs_fusion(classification_fixture())
    s = sorted(y)[0]
    bs = paired_bootstrap_delta_auc(y[s], a[s], b[s], n=500, seed=0)
    assert bs["ci95"][0] <= bs["delta"] <= bs["ci95"][1]


def test_null_difference_is_not_significant():
    # same predictions on both arms → ΔAUC ~ 0, not significant
    fx = classification_fixture()
    rep = stats_report(fx["y_true"], fx["probs"]["unimodal"], fx["probs"]["unimodal"], boot=300)
    assert abs(rep["delta_auroc_mean"]) < 1e-6
    assert rep["meets_prereg_margin"] is False


def test_temperature_scaling_reduces_ece_on_overconfident_case():
    # deliberately over-confident logits: single-scalar T (fit on val) must lift T>1 and cut ECE.
    from pinksight.stats.temperature import overconfident_selfcheck
    rep = overconfident_selfcheck()
    assert rep["temperature"] > 1.0
    assert rep["ece_after"] < rep["ece_before"]


def test_floor_ece_recompute_refuses_to_fabricate_when_logits_absent():
    # the G3 floor report persists summary metrics only — no per-sample fused logits exported.
    from pinksight.stats.temperature import recompute_floor_ece
    out = recompute_floor_ece(None)
    assert out["status"] == "logits_unavailable"
    assert out["ece_before"] is None and out["ece_after"] is None


def test_floor_ece_recompute_works_when_logits_present():
    # when raw logits ARE re-exported, the same path returns real pre/post numbers.
    from pinksight.stats.temperature import overconfident_logits, recompute_floor_ece
    logits = {s: overconfident_logits(seed=s) for s in (0, 1, 2)}
    out = recompute_floor_ece(logits)
    assert out["status"] == "recomputed"
    assert out["ece_after_mean"] < out["ece_before_mean"]
