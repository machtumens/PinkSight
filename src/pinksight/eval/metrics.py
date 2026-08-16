
from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef

from pinksight.metrics import delong_ci, ece

BOOT = 2000
_ALPHA = 0.05


def _boot_ci(fn, *arrays, n=BOOT, seed=0):
    rng = np.random.default_rng(seed)
    m = len(arrays[0])
    vals = []
    for _ in range(n):
        idx = rng.integers(0, m, m)
        try:
            v = fn(*[a[idx] for a in arrays])
        except ValueError:
            continue  
        if np.isfinite(v):
            vals.append(v)
    lo, hi = np.percentile(vals, [100 * _ALPHA / 2, 100 * (1 - _ALPHA / 2)])
    return float(lo), float(hi)


def _sens_spec(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    return tp / max(tp + fn, 1), tn / max(tn + fp, 1)


def classification_metrics(y_true, y_prob, thr=0.5, boot_seed=0) -> dict:
    y_true = np.asarray(y_true, int)
    y_prob = np.asarray(y_prob, float)
    auc, lo, hi = delong_ci(y_true, y_prob)
    out = {"auroc": {"value": round(auc, 4), "ci95": [round(lo, 4), round(hi, 4)],
                     "ci_method": "delong"}}

    def add(name, fn, *arrs):
        clo, chi = _boot_ci(fn, *arrs, seed=boot_seed)
        out[name] = {"value": round(float(fn(*arrs)), 4), "ci95": [round(clo, 4), round(chi, 4)],
                     "ci_method": "bootstrap"}

    yb = (y_prob >= thr).astype(int)
    add("pr_auc", lambda yt, yp: average_precision_score(yt, yp), y_true, y_prob)
    add("mcc", lambda yt, b: matthews_corrcoef(yt, b), y_true, yb)
    add("f1", lambda yt, b: f1_score(yt, b, zero_division=0), y_true, yb)
    add("sensitivity", lambda yt, b: _sens_spec(yt, b)[0], y_true, yb)
    add("specificity", lambda yt, b: _sens_spec(yt, b)[1], y_true, yb)
    out["ece"] = {"value": round(ece(y_true, y_prob), 4), "ci_method": "none", "n_bins": 10}
    out["n"] = int(len(y_true))
    out["prevalence"] = round(float(y_true.mean()), 4)
    return out


def ki67_metrics(y_true, y_pred, thresh=14.0, boot_seed=0) -> dict:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    out = {}

    def add(name, fn):
        clo, chi = _boot_ci(fn, y_true, y_pred, seed=boot_seed)
        out[name] = {"value": round(float(fn(y_true, y_pred)), 4),
                     "ci95": [round(clo, 4), round(chi, 4)], "ci_method": "bootstrap"}

    add("mae", lambda a, b: float(np.mean(np.abs(a - b))))
    add("pearson_r", lambda a, b: float(pearsonr(a, b)[0]))
    add("spearman_rho", lambda a, b: float(spearmanr(a, b)[0]))
    yb = (y_true >= thresh).astype(int)
    auc, lo, hi = delong_ci(yb, y_pred)
    out["auroc_at_thresh"] = {"value": round(auc, 4), "ci95": [round(lo, 4), round(hi, 4)],
                              "ci_method": "delong", "thresh": thresh}
    out["n"] = int(len(y_true))
    out["high_ki67_prevalence"] = round(float(yb.mean()), 4)
    return out
