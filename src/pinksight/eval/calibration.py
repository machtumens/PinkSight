"""P08 calibration: reliability curve + ECE + temperature scaling (fit on VAL only).

Temperature scaling learns ONE scalar T on the validation split and applies it to test logits —
the standard post-hoc calibrator. The signature keeps val and test apart: fitting T on test data
would be leakage. Reuses pinksight.metrics.ece.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from pinksight.metrics import ece


def reliability_curve(y_true, y_prob, n_bins=10) -> list[dict]:
    """Equal-width bins → per non-empty bin {bin, conf, acc, count} for a reliability plot."""
    y_true = np.asarray(y_true, float)
    y_prob = np.asarray(y_prob, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        sel = idx == b
        k = int(sel.sum())
        if k:
            rows.append(
                {
                    "bin": b,
                    "conf": round(float(y_prob[sel].mean()), 4),
                    "acc": round(float(y_true[sel].mean()), 4),
                    "count": k,
                }
            )
    return rows


def _bce_with_logits(logits, y) -> float:
    z = np.asarray(logits, float)
    return float(np.mean(np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))))


def fit_temperature(val_logits, val_labels, bounds=(0.05, 10.0)) -> float:
    """Fit T>0 minimising validation BCE of sigmoid(logits / T). VAL ONLY — never fit on test."""
    val_logits = np.asarray(val_logits, float)
    val_labels = np.asarray(val_labels, float)
    res = minimize_scalar(
        lambda t: _bce_with_logits(val_logits / t, val_labels), bounds=bounds, method="bounded"
    )
    return float(res.x)


def apply_temperature(logits, temperature) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(logits, float) / temperature))


def smooth_ece(y_true, y_prob, kernel_scale: float = 0.1) -> float:
    """Kernel-smoothed (bin-free) calibration error — SmoothECE (Błasiok & Nakkiran 2023).

    Binned ECE is bin-count-attackable at small N: the number/placement of bins changes the value, and
    a small cohort can be gamed by choosing lenient bins (plan G1: "small-N bin-count-attackable").
    SmoothECE removes the bin hyperparameter — it convolves the calibration residual `(y - p)` with a
    Gaussian kernel over the confidence axis and reports the probability-weighted mean absolute smoothed
    residual. No bins, so nothing to tune or attack.

    Estimator (reflected Gaussian kernel regression of the residual):
      r_smooth(i) = Σ_j w_ij (y_j - p_j) / Σ_j w_ij,   w_ij = exp(-((p_i - p_j)^2)/(2 s^2))
      SmoothECE   = Σ_i |r_smooth(i)| / N
    with `s = kernel_scale` on the [0,1] confidence axis. 0 = perfectly calibrated; larger = worse.
    Reflection at the {0,1} boundaries removes the edge bias a raw Gaussian would introduce near 0/1.

    ponytail: pure numpy, O(N^2) over the (small) calibration set — no scipy/sklearn, stays in the
    torch-free core. Deterministic → JSON-stable. Returns a Python float scalar.
    """
    y = np.asarray(y_true, float)
    p = np.asarray(y_prob, float)
    if y.shape != p.shape:
        raise ValueError(f"y_true {y.shape} and y_prob {p.shape} must match")
    n = len(p)
    if n == 0:
        return 0.0
    s = float(kernel_scale)
    resid = y - p
    total = 0.0
    for i in range(n):
        # reflected kernel: weight against p_j AND its reflections across 0 and 1 (boundary de-bias).
        d0 = p[i] - p  # direct
        d1 = p[i] - (-p)  # reflection across 0
        d2 = p[i] - (2.0 - p)  # reflection across 1
        w = (
            np.exp(-(d0**2) / (2 * s * s))
            + np.exp(-(d1**2) / (2 * s * s))
            + np.exp(-(d2**2) / (2 * s * s))
        )
        denom = float(w.sum())
        r_smooth = float((w * resid).sum() / denom) if denom > 0 else 0.0
        total += abs(r_smooth)
    return float(total / n)


def calibration_report(val_logits, val_labels, test_logits, test_labels, n_bins=10) -> dict:
    """Fit T on val, report test ECE before/after + reliability rows. Deterministic → JSON-stable.

    Reports BOTH the binned ECE (`ece_*`) and the bin-free SmoothECE (`smooth_ece_*`) so a small-N
    headline number is not hostage to a bin count (plan G1: prefer SmoothECE).
    """
    t = fit_temperature(val_logits, val_labels)
    p_before = apply_temperature(test_logits, 1.0)
    p_after = apply_temperature(test_logits, t)
    return {
        "temperature": round(t, 4),
        "fit_on": "val_only",
        "ece_before": round(ece(test_labels, p_before), 4),
        "ece_after": round(ece(test_labels, p_after), 4),
        "smooth_ece_before": round(smooth_ece(test_labels, p_before), 4),
        "smooth_ece_after": round(smooth_ece(test_labels, p_after), 4),
        "reliability_before": reliability_curve(test_labels, p_before, n_bins),
        "reliability_after": reliability_curve(test_labels, p_after, n_bins),
    }


def selfcheck() -> int:
    """Runnable check (ponytail): smooth_ece returns a scalar; ~0 on a calibrated draw, large when
    confidently wrong; bin-free (no n_bins arg). Mirrors metrics.ece's selfcheck discipline."""
    rng = np.random.default_rng(0)
    # perfectly calibrated: prob == empirical rate within each confidence group.
    p = np.r_[np.full(400, 0.2), np.full(400, 0.8)]
    yc = np.r_[rng.random(400) < 0.2, rng.random(400) < 0.8].astype(float)
    sece = smooth_ece(yc, p)
    assert isinstance(sece, float), type(sece)
    assert sece < 0.08, f"SmoothECE should be small (~0) for a calibrated draw, got {sece}"
    # confidently wrong: p=0.95 but truth all 0 -> SmoothECE ≈ 0.95.
    bad = smooth_ece(np.zeros(200), np.full(200, 0.95))
    assert bad > 0.8, f"SmoothECE should flag gross miscalibration, got {bad}"
    # empty input -> 0.0 scalar (no crash).
    assert smooth_ece([], []) == 0.0
    print(
        "calibration selfcheck OK — smooth_ece scalar; ~0 calibrated, large miscalibrated, bin-free."
    )  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(selfcheck())
