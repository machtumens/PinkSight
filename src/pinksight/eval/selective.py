
from __future__ import annotations

import numpy as np


def confidence(p) -> np.ndarray:
    return np.abs(np.asarray(p, float) - 0.5)


def confident_mask(p, coverage: float) -> np.ndarray:
    p = np.asarray(p, float)
    k = max(int(round(coverage * len(p))), 1)
    thresh = np.sort(confidence(p))[::-1][k - 1]  
    return confidence(p) >= thresh


def coverage_auroc_curve(y, p, coverages=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2)) -> list[dict]:
    from sklearn.metrics import roc_auc_score

    y, p = np.asarray(y, int), np.asarray(p, float)
    conf = confidence(p)
    order = np.argsort(-conf, kind="mergesort")  
    rows = []
    for c in coverages:
        k = max(int(round(c * len(y))), 2)
        idx = order[:k]
        ys, ps = y[idx], p[idx]
        auc = float(roc_auc_score(ys, ps)) if len(set(ys.tolist())) == 2 else float("nan")
        rows.append({"coverage": round(c, 3), "k": int(k), "auroc": auc,
                     "tnbc_frac": round(float(ys.mean()), 4)})
    return rows


def point_biserial(mask, x) -> float:
    mask = np.asarray(mask, float)
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    if ok.sum() < 3 or len(set(mask[ok].tolist())) < 2:
        return float("nan")
    return float(np.corrcoef(mask[ok], x[ok])[0, 1])


def categorical_assoc(mask, labels) -> float:
    from scipy.stats import chi2_contingency

    mask = np.asarray(mask)
    labels = np.asarray(labels, dtype=object)
    cats = [c for c in set(labels.tolist())]
    table = np.array([[int(((labels == c) & (mask == m)).sum()) for c in cats] for m in (False, True)])
    table = table[:, table.sum(0) > 0]  
    if table.shape[1] < 2 or (table.sum(1) == 0).any():
        return float("nan")
    return float(chi2_contingency(table)[1])
