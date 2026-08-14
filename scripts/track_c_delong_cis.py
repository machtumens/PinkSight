#!/usr/bin/env python3
"""Track-C DeLong CIs — reproduce the 3 companion-panel point AUROCs + ADD canonical DeLong CIs.

Plan item 8 (model-integrity-remediation_12-08-26). The Track-C tabular ENSEMBLE companion panel
(Coimbra / BCSC / METABRIC per ADR-0010, Scope C) reported three point AUROCs in Table2 whose per-sample
OOF arrays were not on disk in a tracked location. This script:
  1. reads the seed-42 OUT-OF-FOLD predictions the panel demos produce (explore/tabular_risk/results/
     *_oof_probs.csv -- the demos are seed-fixed: random_state=42, StratifiedKFold shuffle -> re-running
     reproduces them bit-for-bit; Coimbra + METABRIC were re-run this session to confirm);
  2. verifies each point AUROC reproduces the Table2 value within +/-0.005;
  3. ADDS a 95% CI to each number using the canonical repo estimator:
       - Coimbra (N=116) + METABRIC (N=1917): patient-level FAST DeLong (pinksight.metrics.delong_ci);
       - BCSC: fast-DeLong is UNDEFINED for count-weighted aggregate data (2.39M women in 280660 strata
         rows), so the DeLong-slot is a STRATA (cluster) bootstrap (resample strata rows carrying `count`
         weights) -- the correct resampling unit, matching the method already documented in
         bcsc_metrics.json. Labelled as such, not passed off as a patient-level DeLong.
  4. persists durable tracked artifacts under reports/trackc/ (3 OOF npz + trackc_cis.json).

ADR-0010 FRAMING GUARD (verbatim in spirit): Track C is an ENSEMBLE companion panel (four cohorts share
ZERO patients) -- NOT cross-attention fusion, NEVER fused into the Duke imaging encoder. Reported as
independent per-cohort public benchmarks. What stays FORBIDDEN on ALL tracks: kinetics/doubling-time,
clinical-trial-grade FP/FN reduction, cross-institution generalisation/transfer (LOCK-1). This script
computes calibration/robustness CIs only; it makes NO cross-cohort number and moves NO LOCK.

$0-local, CPU only (LOCK-5). Seeded (LAW L-3): BCSC strata bootstrap uses a fixed seed.

Run:  uv run python scripts/track_c_delong_cis.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pinksight.metrics import delong_ci  # noqa: E402  (canonical fast-DeLong Sun & Xu 2014)

SANDBOX_RESULTS = ROOT / "explore" / "tabular_risk" / "results"
OUT_DIR = ROOT / "reports" / "trackc"
RESULTS_JSON = OUT_DIR / "trackc_cis.json"

_REPRO_TOL = 0.005
_BOOTSTRAP_N = 1000
_BOOTSTRAP_SEED = 42  # LAW L-3 — deterministic strata bootstrap

# Each cohort: OOF CSV, the probability column, the (optional) frequency-weight column, the Table2
# point AUROC + CI to reproduce, and the CI estimator to use.
COHORTS = {
    "coimbra": {
        "csv": "coimbra_oof_probs.csv",
        "prob_col": "oof_prob",
        "weight_col": None,
        "table_auroc": 0.806,
        "table_ci": [0.721, 0.887],
        "ci_method": "fast_delong",
        "unit": "patients",
        "framing": "UCI Breast Cancer Coimbra — metabolic/inflammatory host-state (ADR-0010 Scope C, off-ledger companion)",
    },
    "bcsc": {
        "csv": "bcsc_oof_probs.csv",
        "prob_col": "oof_prob",
        "weight_col": "count_weight",
        "table_auroc": 0.634,
        "table_ci": [0.625, 0.642],
        "ci_method": "strata_cluster_bootstrap",
        "unit": "strata_rows_count_weighted",
        "framing": "BCSC — count-weighted INCIDENCE risk of a screening population (ADR-0010 Scope C, off-ledger companion)",
    },
    "metabric": {
        "csv": "metabric_oof_probs.csv",
        "prob_col": "oof_prob",
        "weight_col": None,
        "table_auroc": 0.744,
        "table_ci": [0.719, 0.770],
        "ci_method": "fast_delong",
        "unit": "patients",
        "framing": "METABRIC — calibrated 5-yr overall-survival PROGNOSIS of diagnosed patients (ADR-0010 Scope C, off-ledger companion)",
    },
}


def _weighted_auroc(y: np.ndarray, score: np.ndarray, weight: np.ndarray | None) -> float:
    """AUROC, optionally frequency-weighted (weighted Mann-Whitney via sklearn sample_weight)."""
    from sklearn.metrics import roc_auc_score

    if weight is None:
        return float(roc_auc_score(y, score))
    return float(roc_auc_score(y, score, sample_weight=weight))


def _strata_cluster_bootstrap_ci(
    y: np.ndarray, score: np.ndarray, weight: np.ndarray, n_boot: int = _BOOTSTRAP_N, seed: int = _BOOTSTRAP_SEED
) -> tuple[float, float, int]:
    """95% percentile CI for a count-weighted AUROC via a STRATA (cluster) bootstrap.

    Resample the strata rows with replacement (each row carries its `count` weight), recompute the
    count-weighted AUROC, take the 2.5/97.5 percentiles. This is the correct resampling unit for
    aggregated risk-stratum data (fast-DeLong's structural-component variance assumes independent
    patient-level predictions, which count-weighted strata are not). Single-class resamples are redrawn.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    boots = []
    tries = 0
    max_tries = n_boot * 3
    while len(boots) < n_boot and tries < max_tries:
        tries += 1
        idx = rng.integers(0, n, size=n)
        yb, sb, wb = y[idx], score[idx], weight[idx]
        # need both classes present (weighted) for a defined AUROC
        if wb[yb == 1].sum() <= 0 or wb[yb == 0].sum() <= 0:
            continue
        boots.append(_weighted_auroc(yb, sb, wb))
    arr = np.asarray(boots, float)
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)), int(len(arr))


def _process(name: str, spec: dict) -> dict:
    csv_path = SANDBOX_RESULTS / spec["csv"]
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Track-C OOF CSV absent: {csv_path} — run explore/tabular_risk/demo_{name}.py first "
            f"(seed-fixed; regenerates the OOF)."
        )
    df = pd.read_csv(csv_path)
    y = df["y_true"].to_numpy().astype(int)
    score = df[spec["prob_col"]].to_numpy().astype(float)
    weight = df[spec["weight_col"]].to_numpy().astype(float) if spec["weight_col"] else None

    point_auroc = _weighted_auroc(y, score, weight)
    reproduces = abs(point_auroc - spec["table_auroc"]) <= _REPRO_TOL

    if spec["ci_method"] == "fast_delong":
        auc_dl, lo, hi = delong_ci(y, score)  # unweighted patient-level fast-DeLong
        ci = [round(lo, 4), round(hi, 4)]
        ci_note = "patient-level fast-DeLong (Sun & Xu 2014, pinksight.metrics.delong_ci)"
        n_boot = None
    elif spec["ci_method"] == "strata_cluster_bootstrap":
        lo, hi, n_boot = _strata_cluster_bootstrap_ci(y, score, weight)
        ci = [round(lo, 4), round(hi, 4)]
        ci_note = (
            f"strata cluster bootstrap (n={n_boot}, seed={_BOOTSTRAP_SEED}) — fast-DeLong is UNDEFINED "
            "for count-weighted aggregate data; this is the correct DeLong-slot resampling unit"
        )
    else:  # pragma: no cover
        raise ValueError(f"unknown ci_method {spec['ci_method']!r}")

    # durable tracked OOF npz (closes the 'OOF not on disk' gap in a git-tracked location)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    npz_path = OUT_DIR / f"{name}_oof.npz"
    save_kwargs = {"y": y, "oof": score}
    if weight is not None:
        save_kwargs["count_weight"] = weight
    np.savez(npz_path, **save_kwargs)

    return {
        "framing": spec["framing"],
        "unit": spec["unit"],
        "n_rows": int(len(y)),
        "point_auroc": round(point_auroc, 4),
        "table_auroc": spec["table_auroc"],
        "reproduces_table_auroc_within_0.005": bool(reproduces),
        "auroc_abs_delta_vs_table": round(abs(point_auroc - spec["table_auroc"]), 4),
        "ci95": ci,
        "ci_method": spec["ci_method"],
        "ci_note": ci_note,
        "bootstrap_n": n_boot,
        "table_ci_for_reference": spec["table_ci"],
        "oof_npz": str(npz_path.relative_to(ROOT)),
        "oof_source_csv": str(csv_path.relative_to(ROOT)),
    }


def run() -> dict:
    cohorts = {name: _process(name, spec) for name, spec in COHORTS.items()}
    all_reproduce = all(c["reproduces_table_auroc_within_0.005"] for c in cohorts.values())

    out = {
        "gate": "TrackC-DeLong-CIs (plan item 8)",
        "framing": (
            "ADR-0010 Scope-C ENSEMBLE companion panel (Coimbra/BCSC/METABRIC) — independent per-cohort "
            "public benchmarks; ensemble NOT fusion; ZERO shared patients; NEVER fused into the Duke "
            "encoder. Calibration/robustness CIs only; no cross-cohort number (LOCK-1)."
        ),
        "reproducibility_tol": _REPRO_TOL,
        "all_reproduce_table_auroc": bool(all_reproduce),
        "cohorts": cohorts,
        "ci_estimator_note": (
            "Coimbra + METABRIC use patient-level fast-DeLong (canonical repo delong_ci). BCSC uses a "
            "strata cluster bootstrap because it is count-weighted aggregate data (2.39M women / 280660 "
            "strata) for which a patient-level DeLong variance is undefined — the same DeLong-slot the "
            "existing bcsc_metrics.json documents. No LOCK moved; LOCK-1 (cross-institution FORBIDDEN) "
            "unchanged; ensemble not fusion, zero shared patients (ADR-0010)."
        ),
        "compute": "cpu-zero-dollar",
        "seed": _BOOTSTRAP_SEED,
    }
    RESULTS_JSON.write_text(json.dumps(out, indent=2) + "\n")
    return out


def main() -> int:
    out = run()
    print(json.dumps(out, indent=2))  # noqa: T201
    print("\n[trackc-cis] reproduction + DeLong CIs:")  # noqa: T201
    for name, c in out["cohorts"].items():
        print(  # noqa: T201
            f"  {name:9s} AUROC {c['point_auroc']} (table {c['table_auroc']}, "
            f"reproduce={c['reproduces_table_auroc_within_0.005']}) CI {c['ci95']} [{c['ci_method']}]"
        )
    print(f"[trackc-cis] all_reproduce={out['all_reproduce_table_auroc']}; wrote {RESULTS_JSON.relative_to(ROOT)} + 3 OOF npz")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
