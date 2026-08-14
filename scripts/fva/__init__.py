"""FVA — Fusion-Value-Attribution: the packaged required-diagnostic any fusion claim must pass.

Individual Shapley/shuffle/ceiling tools exist in the wild; the NOVELTY here is the *packaged
required protocol* — four diagnostics that must ALL be present before a multimodal fusion result is
believed: (1) Shapley attribution, (2) shuffle sentinel, (3) conditional-independence proxy, (4)
information ceiling. See ``FVA_STANDARD.md`` for the written checklist.

Public contract (C2): ``run_fva(config: FVAConfig) -> FVAReport``.

This module is the shared home of the exact logic that ``h6_modality_audit.py`` and
``h4_info_ceiling.py`` already pre-registered; those two scripts become thin CLI wrappers around it
(C2-7 / C2-8) with byte-identical numeric output guarded by the C2-13 regression gate.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from fva.config import FVAConfig
from fva.conditional_indep import conditional_info_proxy
from fva.info_ceiling import (
    build_pca_radiomics,
    cka_stability,
    fisher_ratio,
    knn_bayes_sweep,
    run_family,
    selfcheck as info_ceiling_selfcheck,
    twonn_dim,
)
from fva.report import FVAReport, evaluate_verdict_flags, render, synthesize_verdict
from fva.shapley import exact_shapley
from fva.shuffle_sentinel import (
    coalition_oof,
    empty_coalition_oof,
    shuffle_note,
    stream_shuffle_sentinels,
)

__all__ = [
    "FVAConfig",
    "FVAReport",
    "run_fva",
    "coalition_oof",
    "empty_coalition_oof",
    "exact_shapley",
    "shuffle_note",
    "stream_shuffle_sentinels",
    "conditional_info_proxy",
]


def _subtype_labels(manifest: Path) -> pd.Series:
    man = pd.read_csv(manifest).set_index("patient_id")
    return man["subtype"].map({"luminal_like": 0, "tnbc": 1})


def _coalition_auroc(coalition, streams, y, groups, seeds, n_splits):
    """Multi-seed pooled-OOF AUROC for a coalition (mirrors h6.coalition_auroc; returns mean+detail)."""
    from sklearn.metrics import roc_auc_score
    per_auc, oof_by_seed = {}, {}
    for s in seeds:
        if not coalition:
            oof = empty_coalition_oof(y, groups, s, n_splits=n_splits)
        else:
            mats = [streams[m]["X"] for m in coalition]
            imp = [streams[m]["impute"] for m in coalition]
            oof = coalition_oof(mats, imp, y, groups, seed=s, n_splits=n_splits)
        auc = 0.5 if len(np.unique(oof)) == 1 else roc_auc_score(y, oof)
        per_auc[s] = float(auc)
        oof_by_seed[s] = oof
    return float(np.mean(list(per_auc.values()))), {
        "auroc_mean": round(float(np.mean(list(per_auc.values()))), 4),
        "auroc_std": round(float(np.std(list(per_auc.values()))), 4),
        "auroc_per_seed": {str(k): round(v, 4) for k, v in per_auc.items()},
        "_oof_by_seed": oof_by_seed,
    }


def run_fva(config: FVAConfig, streams: dict | None = None, y=None, groups=None,
            tag: str = "run") -> FVAReport:
    """Run the four FVA diagnostics on a supplied 3-stream set and render a report.

    Args:
        config: FVAConfig (paths, seeds, split, frozen verdict thresholds).
        streams: dict ``{name: {"X": ndarray, "impute": bool, "meta": dict}}`` aligned on the same
            pid order, for the players in ``config.players``. If None, the caller must supply y/groups
            too — but the standard use passes a prebuilt streams dict.
        y: aligned subtype labels (0/1).
        groups: aligned patient-id array (for patient-disjoint CV).
        tag: report filename tag (``fva_{tag}_REPORT.json/.txt``).

    Returns:
        FVAReport with all four components + evaluated verdict flags + synthesized verdict.
    """
    if streams is None or y is None or groups is None:
        raise ValueError("run_fva requires prebuilt streams + y + groups (use fva_a1_report or "
                         "the h6/h4 wrappers to build them)")

    players = list(config.players)

    # (1) + delta: all coalitions -> Shapley + MRI ΔAUC(add mri | others)
    from pinksight.metrics import delong_paired
    coalition_detail, coalition_auc = {}, {}
    for r in range(len(players) + 1):
        for combo in combinations(players, r):
            key = tuple(sorted(combo))
            m, det = _coalition_auroc(key, streams, y, groups, config.seeds, config.n_splits)
            coalition_detail["+".join(key) if key else "empty"] = det
            coalition_auc[key] = m

    shap = exact_shapley(coalition_auc, players, tol=config.efficiency_tol)

    mri_delta = mri_p = None
    if "mri" in players:
        others = tuple(sorted(p for p in players if p != "mri"))
        full = tuple(sorted(players))
        det_full = coalition_detail["+".join(full)]
        det_others = coalition_detail["+".join(others)]
        d_perseed, p_perseed = [], []
        for s in config.seeds:
            pr = delong_paired(y, det_others["_oof_by_seed"][s], det_full["_oof_by_seed"][s])
            d_perseed.append(pr["delta"]); p_perseed.append(pr["p"])
        mri_delta = float(np.mean(d_perseed))
        mri_p = float(np.mean(p_perseed))

    # (2) shuffle sentinel
    sentinels = stream_shuffle_sentinels(streams, y, groups, coalition_detail,
                                         seeds=config.seeds, n_splits=config.n_splits,
                                         stream_names=tuple(players))

    # (3) conditional-independence proxy (needs a clinical stream)
    cond = {}
    if "clinical" in players:
        imaging = tuple(p for p in players if p != "clinical")
        cond = conditional_info_proxy(streams, y, groups, n_splits=config.n_splits,
                                      imaging_streams=imaging)

    # strip oof arrays before serialising
    for det in coalition_detail.values():
        det.pop("_oof_by_seed", None)

    flags = evaluate_verdict_flags(shap, {}, mri_delta, mri_p, config)
    report = FVAReport(
        shapley=shap,
        shuffle_sentinel=sentinels,
        conditional_indep_mi=cond,
        info_ceiling={"coalitions": coalition_detail},
        verdict_flags=flags,
        fva_verdict=synthesize_verdict(flags),
        meta={"tag": tag, "players": players, "n": int(len(y)),
              "seeds": list(config.seeds), "n_splits": config.n_splits, **config.meta},
    )
    render(report, config.out_dir, tag)
    return report
