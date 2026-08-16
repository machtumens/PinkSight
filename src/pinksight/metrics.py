
from __future__ import annotations

from math import erf, sqrt

import numpy as np

_Z95 = 1.959963984540054  


def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    z = x[order]
    n = len(x)
    t = np.empty(n, float)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1  
        i = j
    out = np.empty(n, float)
    out[order] = t
    return out


def delong_ci(y_true, y_score) -> tuple[float, float, float]:
    y_true = np.asarray(y_true, int)
    y_score = np.asarray(y_score, float)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    m, n = len(pos), len(neg)
    if m == 0 or n == 0:
        return float("nan"), float("nan"), float("nan")

    tx, ty = _midrank(pos), _midrank(neg)
    tz = _midrank(np.concatenate([pos, neg]))
    auc = tz[:m].sum() / m / n - (m + 1) / 2.0 / n
    v01 = (tz[:m] - tx) / n  
    v10 = 1.0 - (tz[m:] - ty) / m  
    var = v01.var(ddof=1) / m + v10.var(ddof=1) / n
    half = _Z95 * sqrt(var)
    return float(auc), float(max(0.0, auc - half)), float(min(1.0, auc + half))


def delong_paired(y_true, score_a, score_b) -> dict:
    y_true = np.asarray(y_true, int)
    sa, sb = np.asarray(score_a, float), np.asarray(score_b, float)
    if not (len(y_true) == len(sa) == len(sb)):
        raise ValueError("paired DeLong needs the same patients in all three arrays")
    pos_mask = y_true == 1
    m, n = int(pos_mask.sum()), int((~pos_mask).sum())
    if m == 0 or n == 0:
        raise ValueError("paired DeLong needs both classes present")

    aucs, v01, v10 = [], [], []
    for s in (sa, sb):
        pos, neg = s[pos_mask], s[~pos_mask]
        tx, ty = _midrank(pos), _midrank(neg)
        tz = _midrank(np.concatenate([pos, neg]))
        aucs.append(tz[:m].sum() / m / n - (m + 1) / 2.0 / n)
        v01.append((tz[:m] - tx) / n)
        v10.append(1.0 - (tz[m:] - ty) / m)

    s_cov = np.cov(np.vstack(v01), ddof=1) / m + np.cov(np.vstack(v10), ddof=1) / n
    delta = float(aucs[1] - aucs[0])
    var = float(s_cov[0, 0] - 2 * s_cov[0, 1] + s_cov[1, 1])  
    if var <= 0:
        return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "delta": delta,
                "ci95": [delta, delta], "z": 0.0, "p": 1.0}
    se = sqrt(var)
    z = delta / se
    p = 2.0 * (1.0 - _normal_cdf(abs(z)))
    half = _Z95 * se
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "delta": delta,
            "ci95": [delta - half, delta + half], "z": float(z), "p": float(p)}


def ece(y_true, y_prob, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true, float)
    y_prob = np.asarray(y_prob, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, edges[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        sel = idx == b
        k = int(sel.sum())
        if k:
            total += abs(y_prob[sel].mean() - y_true[sel].mean()) * k / len(y_true)
    return float(total)


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def selfcheck() -> int:
    rng = np.random.default_rng(0)
    n = 600
    y = np.r_[np.zeros(n // 2), np.ones(n // 2)].astype(int)
    s = y + rng.normal(0, 0.5, n)
    auc, lo, hi = delong_ci(y, s)

    ranks = np.empty(n, float)
    ranks[s.argsort(kind="mergesort")] = np.arange(1, n + 1)
    npos = int((y == 1).sum())
    mw = (ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * (n - npos))
    assert abs(auc - mw) < 1e-9, f"DeLong AUC {auc} != Mann-Whitney {mw}"
    assert 0.0 <= lo <= auc <= hi <= 1.0, f"CI not bracketing AUC: {lo},{auc},{hi}"

    assert abs(_normal_cdf(_Z95) - 0.975) < 1e-9, "z95 wrong"
    p = np.r_[np.full(300, 0.2), np.full(300, 0.8)]
    yc = np.r_[rng.random(300) < 0.2, rng.random(300) < 0.8].astype(int)
    assert ece(yc, p) < 0.05, "ECE should be ~0 for a calibrated draw"
    assert ece(np.zeros(200), np.full(200, 0.95)) > 0.9, "ECE should flag miscalibration"

    yb = np.r_[np.zeros(300), np.ones(300)].astype(int)
    base = yb + rng.normal(0, 1.2, 600)  
    better = yb + rng.normal(0, 0.4, 600)  
    same = delong_paired(yb, base, base)
    assert same["delta"] == 0.0 and same["p"] == 1.0, f"identical models must not differ: {same}"
    pr = delong_paired(yb, base, better)
    assert pr["auc_b"] > pr["auc_a"] and pr["delta"] > 0, f"B should win: {pr}"
    assert pr["p"] < 0.05, f"clear separation should be significant: p={pr['p']}"
    assert pr["ci95"][0] <= pr["delta"] <= pr["ci95"][1], f"CI must bracket delta: {pr}"
    assert abs(pr["auc_a"] - delong_ci(yb, base)[0]) < 1e-9, "paired auc_a != delong_ci auc"

    print("metrics selfcheck OK — DeLong CI brackets AUC; paired ΔAUC significant; ECE reads calibration")  
    return 0


if __name__ == "__main__":
    raise SystemExit(selfcheck())
