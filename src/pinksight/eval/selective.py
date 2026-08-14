"""BP4: selective-prediction primitives — is a weak/null predictor UNIFORM or HETEROGENEOUS?

Blueprint 4 (reports/research_synthesis/BLUEPRINTS_16-07-26.md). An aggregate AUROC near chance
(imaging-only ~0.518) does not tell you whether the model is uniformly random or a mixture of a
confident-correct subpopulation and a random remainder. Selective prediction answers that: sort
patients by prediction confidence, and plot AUROC on the most-confident coverage fraction. A rising
curve => a reliable subpopulation exists (heterogeneous null); a flat curve => the null is uniform.

Pure numpy + sklearn.metrics (same tier as eval/ablation.py) — no torch, no new dependency.

Ledger: "selective at-diagnosis characterisation with a calibrated coverage view" — never a prognostic
claim. Selection is defined by imaging CONFIDENCE, not by outcome.
"""

from __future__ import annotations

import numpy as np


def confidence(p) -> np.ndarray:
    """Per-patient confidence = distance of the (calibrated) probability from the 0.5 boundary."""
    return np.abs(np.asarray(p, float) - 0.5)


def confident_mask(p, coverage: float) -> np.ndarray:
    """Boolean mask selecting the ``coverage`` fraction of patients with the highest confidence."""
    p = np.asarray(p, float)
    k = max(int(round(coverage * len(p))), 1)
    thresh = np.sort(confidence(p))[::-1][k - 1]  # k-th largest confidence
    return confidence(p) >= thresh


def coverage_auroc_curve(y, p, coverages=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2)) -> list[dict]:
    """AUROC on the most-confident ``coverage`` fraction, for each coverage. NaN AUROC if a
    single-class subset (records it rather than crashing). Rows carry coverage, k, auroc, tnbc_frac."""
    from sklearn.metrics import roc_auc_score

    y, p = np.asarray(y, int), np.asarray(p, float)
    conf = confidence(p)
    order = np.argsort(-conf, kind="mergesort")  # most confident first, stable
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
    """Correlation between a boolean selection mask and a numeric covariate (F-B4.2 easy-case test).

    High |r| => 'confident' patients are systematically the ones with extreme covariate values (large
    tumours, extreme age) — i.e. easy cases, not genuine imaging signal.
    """
    mask = np.asarray(mask, float)
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    if ok.sum() < 3 or len(set(mask[ok].tolist())) < 2:
        return float("nan")
    return float(np.corrcoef(mask[ok], x[ok])[0, 1])


def categorical_assoc(mask, labels) -> float:
    """Chi-square p-value of selection vs a categorical covariate (e.g. scanner/era). Small p => the
    confident subset is scanner/era-confounded rather than biological (F-B4.1 / F-B2.6)."""
    from scipy.stats import chi2_contingency

    mask = np.asarray(mask)
    labels = np.asarray(labels, dtype=object)
    cats = [c for c in set(labels.tolist())]
    table = np.array([[int(((labels == c) & (mask == m)).sum()) for c in cats] for m in (False, True)])
    table = table[:, table.sum(0) > 0]  # drop empty columns
    if table.shape[1] < 2 or (table.sum(1) == 0).any():
        return float("nan")
    return float(chi2_contingency(table)[1])
