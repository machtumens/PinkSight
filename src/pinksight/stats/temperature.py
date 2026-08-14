"""G3 temperature scaling — single-scalar post-hoc calibration for the fusion floor.

The fusion FLOOR run (`reports/G3_fusion_floor/`) shows poor calibration (ECE 0.23–0.31 across
rungs) but persists ONLY summary metrics — no per-sample fused logits/probabilities are exported.
Single-scalar temperature scaling (fit T on val logits, apply to test logits) is the standard
post-hoc fix and needs NO retrain / NO Kaggle. The T-fit + apply primitives already live in
`pinksight.eval.calibration`; this module reuses them (ponytail — one scalar T, scipy only) and
adds two things the floor run needs:

1. `overconfident_selfcheck()` — a deliberately over-confident synthetic logit set where scaling
   MUST raise T>1 and reduce ECE. Runnable assert-based proof (no test framework required).
2. `recompute_floor_ece()` — recomputes real pre/post ECE FROM EXPORTED floor logits. It refuses
   to fabricate a number: if the floor run's raw per-fold logits are absent, it returns a status
   saying they must be re-exported first. Every gate produces a number — but a HONEST one.
"""

from __future__ import annotations

import numpy as np

from pinksight.eval.calibration import apply_temperature, fit_temperature
from pinksight.metrics import ece

# Re-export the fit/apply primitives so callers use `pinksight.stats` as the G3 stats surface.
__all__ = [
    "fit_temperature",
    "apply_temperature",
    "overconfident_logits",
    "temperature_scale_ece",
    "overconfident_selfcheck",
    "recompute_floor_ece",
]


def overconfident_logits(seed: int = 0, n: int = 800, scale: float = 4.0, flip_frac: float = 0.30):
    """Deliberately OVER-confident logits: large magnitude (`scale`) so sigmoid is pushed to the
    0/1 extremes, but a fraction (`flip_frac`) point CONFIDENTLY the WRONG way → confidence far
    exceeds accuracy (high ECE, the floor's failure mode). This is what makes T>1 the correct fix.
    Returned as val/test halves so T is fit on val and evaluated on test (fitting on test = leakage)."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.5).astype(int)
    signal = 2 * y - 1
    flip = rng.random(n) < flip_frac  # confidently-wrong minority → over-confidence, not noise
    signal = np.where(flip, -signal, signal)
    logits = scale * signal + rng.normal(0.0, 0.5, n)
    half = n // 2
    return {
        "val": {"logits": logits[:half], "labels": y[:half]},
        "test": {"logits": logits[half:], "labels": y[half:]},
    }


def temperature_scale_ece(val_logits, val_labels, test_logits, test_labels) -> dict:
    """Fit T on val logits, apply to test logits, recompute test ECE before/after. Real numbers only
    — pass it real exported floor logits and it returns the real recompute."""
    t = fit_temperature(val_logits, val_labels)
    p_before = apply_temperature(test_logits, 1.0)
    p_after = apply_temperature(test_logits, t)
    return {
        "temperature": round(float(t), 4),
        "fit_on": "val_only",
        "ece_before": round(float(ece(test_labels, p_before)), 4),
        "ece_after": round(float(ece(test_labels, p_after)), 4),
        "n_test": int(np.asarray(test_labels).size),
    }


def overconfident_selfcheck() -> dict:
    """Assert-based proof that single-scalar temperature scaling reduces ECE on a deliberately
    over-confident case. Fails loudly if the primitive ever regresses. Returns the numbers so a
    caller can print/log them (every check produces a number)."""
    fx = overconfident_logits()
    rep = temperature_scale_ece(fx["val"]["logits"], fx["val"]["labels"],
                                fx["test"]["logits"], fx["test"]["labels"])
    assert rep["temperature"] > 1.0, f"over-confident logits must fit T>1, got {rep['temperature']}"
    assert rep["ece_after"] < rep["ece_before"], (
        f"scaling must REDUCE ECE on over-confident case: "
        f"before={rep['ece_before']} after={rep['ece_after']}")
    return rep


def recompute_floor_ece(floor_logits: dict | None) -> dict:
    """Recompute real G3 floor pre/post ECE from EXPORTED per-seed fused logits, or report that the
    raw logits are absent. `floor_logits` shape (when available):
        {seed: {"val": {"logits", "labels"}, "test": {"logits", "labels"}}}.

    Does NOT fabricate a number when logits are missing — the floor report only persists summary
    metrics, so this returns a status telling the caller to re-export the raw fused logits first."""
    if not floor_logits:
        return {
            "status": "logits_unavailable",
            "note": ("reports/G3_fusion_floor/ persists summary metrics only "
                     "(per-seed AUROC/ECE) — no per-sample fused logits/probabilities were "
                     "exported. Re-run the floor with raw fused val/test logit export to enable "
                     "a real ECE recompute. No number fabricated."),
            "ece_before": None,
            "ece_after": None,
        }
    per_seed = {}
    for s, split in floor_logits.items():
        per_seed[str(s)] = temperature_scale_ece(
            split["val"]["logits"], split["val"]["labels"],
            split["test"]["logits"], split["test"]["labels"])
    ece_before = float(np.mean([r["ece_before"] for r in per_seed.values()]))
    ece_after = float(np.mean([r["ece_after"] for r in per_seed.values()]))
    return {
        "status": "recomputed",
        "per_seed": per_seed,
        "ece_before_mean": round(ece_before, 4),
        "ece_after_mean": round(ece_after, 4),
    }


if __name__ == "__main__":  # runnable check: `python -m pinksight.stats.temperature`
    r = overconfident_selfcheck()
    print(f"[temperature selfcheck] T={r['temperature']}  "
          f"ECE {r['ece_before']} -> {r['ece_after']}  (reduced; n_test={r['n_test']})  OK")
