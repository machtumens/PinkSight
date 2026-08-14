"""P08 ablation ladder: radiomics → unimodal → late-fusion → cross-attn over identical folds/seeds.

Model-agnostic — consumes each rung's per-seed pooled out-of-fold probabilities on a shared,
patient-aligned label vector. Emits a per-rung metrics block (AUROC = DeLong CI averaged across
seeds, cv.py convention; others = bootstrap CI averaged across seeds; plus across-seed std) and
regenerates metrics.json + ROC/PR/reliability figures from a script — no hand-typed numbers.

Significance of any ΔAUC is DEFERRED to the P09 stats agent; never asserted here (Rule 8).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pinksight.eval.calibration import reliability_curve
from pinksight.eval.metrics import classification_metrics, ki67_metrics

LADDER = ("radiomics", "unimodal", "late_fusion", "cross_attn")


def _aggregate(per_seed: dict) -> dict:
    """Collapse {seed: classification_metrics} → mean value + averaged CI + across-seed std."""
    seeds = sorted(per_seed)
    names = [k for k, v in per_seed[seeds[0]].items() if isinstance(v, dict) and "value" in v]
    agg = {}
    for m in names:
        # TICKET-012: a single-class OOF fold yields NaN AUROC (delong_ci no longer
        # raises). Pool only the finite seeds so one degenerate fold can't NaN out the
        # whole rung; if ALL seeds are NaN the value stays NaN and the rung is degraded.
        vals = [per_seed[s][m]["value"] for s in seeds]
        finite = [v for v in vals if v is not None and np.isfinite(v)]
        entry = {"value": round(float(np.mean(finite)), 4) if finite else float("nan"),
                 "across_seed_std": round(float(np.std(finite)), 4) if finite else float("nan"),
                 "ci_method": per_seed[seeds[0]][m].get("ci_method", "bootstrap")}
        cis = [per_seed[s][m]["ci95"] for s in seeds if "ci95" in per_seed[s][m]
               and all(c is not None and np.isfinite(c) for c in per_seed[s][m]["ci95"])]
        if cis:
            entry["ci95"] = [round(float(np.mean([c[0] for c in cis])), 4),
                             round(float(np.mean([c[1] for c in cis])), 4)]
        agg[m] = entry
    # Mark the rung degraded when any seed's AUROC was non-finite (single-class fold).
    per_seed_auroc = {str(s): per_seed[s]["auroc"]["value"] for s in seeds}
    n_degraded = sum(1 for v in per_seed_auroc.values()
                     if v is None or not np.isfinite(v))
    if n_degraded:
        agg["degraded"] = True
        agg["note"] = f"single-class fold: {n_degraded}/{len(seeds)} seed(s) had NaN AUROC"
    agg["per_seed_auroc"] = per_seed_auroc
    agg["n"] = per_seed[seeds[0]]["n"]
    agg["prevalence"] = per_seed[seeds[0]]["prevalence"]
    return agg


def ablation_ladder(probs_by_rung: dict, y_true_by_seed: dict, thr=0.5) -> dict:
    """Per rung, per seed classification_metrics → aggregated block. Rungs share the same folds."""
    table = {}
    for rung, probs_by_seed in probs_by_rung.items():
        per_seed = {s: classification_metrics(y_true_by_seed[s], probs_by_seed[s], thr, boot_seed=s)
                    for s in sorted(probs_by_seed)}
        table[rung] = _aggregate(per_seed)
    return table


def _save_figures(y, p_by_rung, out: Path) -> None:
    """ROC + PR (all rungs) and reliability (top rung) on seed 0. PNG bytes aren't asserted."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve

    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    for rung, p in p_by_rung.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=rung)
    ax.plot([0, 1], [0, 1], "k--", lw=0.7)
    ax.set(xlabel="FPR", ylabel="TPR", title="Subtype characterisation — ROC (fixture)")
    ax.legend()
    fig.savefig(out / "roc.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    for rung, p in p_by_rung.items():
        prec, rec, _ = precision_recall_curve(y, p)
        ax.plot(rec, prec, label=rung)
    ax.set(xlabel="Recall", ylabel="Precision", title="Subtype characterisation — PR (fixture)")
    ax.legend()
    fig.savefig(out / "pr.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    rows = reliability_curve(y, p_by_rung[LADDER[-1]])
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=0.7)
    ax.plot([r["conf"] for r in rows], [r["acc"] for r in rows], "o-")
    ax.set(xlabel="confidence", ylabel="accuracy", title="Reliability — cross-attn (fixture)")
    fig.savefig(out / "reliability.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def build_metrics_json(fixture: dict, ki67: dict, out_dir: Path, thr=0.5, figures=True) -> dict:
    """Assemble metrics.json (byte-stable via sort_keys) + figures. The single source of numbers."""
    table = ablation_ladder(fixture["probs"], fixture["y_true"], thr)
    doc = {
        "gate": "G2/G3 (fixture)",
        "task": "subtype characterisation (Luminal-like vs TNBC) + Ki-67 stratification at diagnosis",
        "ladder_order": list(LADDER),
        "ablation_ladder": table,
        "ki67": ki67_metrics(ki67["y_true"], ki67["pred"], ki67["thresh"]),
        "delta_auroc_crossattn_vs_unimodal": round(
            table["cross_attn"]["auroc"]["value"] - table["unimodal"]["auroc"]["value"], 4),
        "significance_note": "ΔAUROC significance is deferred to P09 stats (paired bootstrap + "
                             "DeLong across folds AND seeds); not asserted here (Rule 8).",
        "ci_note": "AUROC=DeLong CI averaged across seeds; others=bootstrap CI averaged across "
                   "seeds; across_seed_std also reported.",
        "source": "synthetic fixture — wired to prediction arrays, not a trained model",
        "seeds": sorted(fixture["y_true"]),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    if figures:
        s0 = sorted(fixture["y_true"])[0]
        _save_figures(fixture["y_true"][s0], {r: fixture["probs"][r][s0] for r in LADDER},
                      out_dir / "figures")
    return doc
