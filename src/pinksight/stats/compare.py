"""P09 inferential stats — the core research question: cross-attn fusion vs best unimodal.

(1) paired bootstrap CI on ΔAUC; (2) DeLong test AGGREGATED across folds AND seeds; (3) power /
MDE vs the pre-registered margin (ΔAUC ≥ 0.03, p<0.05). Reuses pinksight.metrics.delong_paired
(correlated-ROC ΔAUC). Never asserts significance from a single split — seeds are independent
replications combined by Stouffer's z (Rule 8). Claim-ledger: characterisation effects only.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from pinksight.metrics import delong_paired

Z975 = 1.959963984540054
Z80 = 0.8416212335729143  # norm.ppf(0.80), for MDE at 80% power
PREREG_MARGIN = 0.03  # decisions.md LOCK-6


def _auc(y, s) -> float:
    ranks = np.empty(len(s), float)
    ranks[np.asarray(s).argsort(kind="mergesort")] = np.arange(1, len(s) + 1)
    npos = int((y == 1).sum())
    nneg = len(s) - npos
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def paired_bootstrap_delta_auc(y, score_a, score_b, n=2000, seed=0) -> dict:
    """Percentile 95% CI of AUC_b − AUC_a on ONE split via a paired patient bootstrap."""
    y = np.asarray(y, int)
    a, b = np.asarray(score_a, float), np.asarray(score_b, float)
    m = len(y)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n):
        idx = rng.integers(0, m, m)
        yy = y[idx]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        deltas.append(_auc(yy, b[idx]) - _auc(yy, a[idx]))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"delta": round(float(np.mean(deltas)), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)], "n_boot": len(deltas)}


def _se_from_ci(ci) -> float:
    return (ci[1] - ci[0]) / (2 * Z975)


def stats_report(y_by_seed, score_a_by_seed, score_b_by_seed, margin=PREREG_MARGIN,
                 boot=2000) -> dict:
    """Aggregate ΔAUC across seeds. score_a = best unimodal, score_b = cross-attn fusion.

    Returns per-seed DeLong + bootstrap, a combined Stouffer z / p, a mean bootstrap ΔAUC CI, the
    MDE at 80% power, the power to detect `margin`, and the explicit pre-registered verdict.
    """
    seeds = sorted(y_by_seed)
    per_seed = {}
    for s in seeds:
        dl = delong_paired(y_by_seed[s], score_a_by_seed[s], score_b_by_seed[s])
        dl = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in dl.items()}
        bs = paired_bootstrap_delta_auc(y_by_seed[s], score_a_by_seed[s], score_b_by_seed[s],
                                        n=boot, seed=s)
        per_seed[str(s)] = {"delong": dl, "bootstrap": bs}

    deltas = np.array([per_seed[str(s)]["delong"]["delta"] for s in seeds])
    zs = np.array([per_seed[str(s)]["delong"]["z"] for s in seeds])
    ses = np.array([_se_from_ci(per_seed[str(s)]["delong"]["ci95"]) for s in seeds])
    boot_lo = float(np.mean([per_seed[str(s)]["bootstrap"]["ci95"][0] for s in seeds]))
    boot_hi = float(np.mean([per_seed[str(s)]["bootstrap"]["ci95"][1] for s in seeds]))

    z_comb = float(np.mean(zs) * np.sqrt(len(seeds)))  # Stouffer across independent seeds
    p_comb = float(2 * (1 - norm.cdf(abs(z_comb))))
    delta_mean = float(deltas.mean())
    se_mean = float(np.mean(ses))
    mde = float((Z975 + Z80) * se_mean)
    power_at_margin = float(1 - norm.cdf(Z975 - margin / se_mean)) if se_mean > 0 else 1.0
    boot_ci = [round(boot_lo, 4), round(boot_hi, 4)]

    verdict = bool(delta_mean >= margin and boot_ci[0] > 0 and p_comb < 0.05)
    return {
        "question": "cross-attention fusion vs best unimodal (subtype characterisation AUROC)",
        "per_seed": per_seed,
        "delta_auroc_mean": round(delta_mean, 4),
        "delta_auroc_across_seed_std": round(float(deltas.std()), 4),
        "bootstrap_delta_ci95_mean": boot_ci,
        "delong_combined_z": round(z_comb, 4),
        "delong_combined_p": round(p_comb, 6),
        "mean_se_delta": round(se_mean, 4),
        "prereg_margin": margin,
        "mde_80pct_power": round(mde, 4),
        "power_to_detect_margin": round(power_at_margin, 4),
        "verdict_criteria": "delta >= 0.03 AND bootstrap CI excludes 0 AND DeLong combined p < 0.05",
        "meets_prereg_margin": verdict,
        "n_seeds": len(seeds),
        "n_folds": "pooled-OOF per seed (5-fold, cv.py)",
        "source": "synthetic fixture — wired to prediction arrays, not a trained model",
    }
