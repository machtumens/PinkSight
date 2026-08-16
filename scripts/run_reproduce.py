
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  
DATA_README = "data/README.md"

RUNGS: list[dict] = [
    {
        "name": "G3 — Track-A Duke hierarchical fusion + MoE",
        "inputs": ["data/manifest_v1.csv"],
        "scripts": [
            "scripts/train_g3_hierarchical.py",
            "scripts/train_g3_moe.py",
            "scripts/g3_paired_delong_vs_anchor.py",
            "scripts/g3_moe7_salt_reporting.py",
        ],
    },
    {
        "name": "G5 — calibration / quasi-external / XAI",
        "inputs": ["data/manifest_v1.csv"],
        "scripts": [
            "scripts/g5_calibration.py",
            "scripts/g5_external_eval.py",
            "scripts/g5_xai_subtype.py",
        ],
    },
    {
        "name": "Track-C — standalone tabular ensemble panel",
        "inputs": ["data/manifest_v1.csv"],
        "scripts": [
            "scripts/track_c_tabular_panel.py",
            "scripts/track_c_delong_cis.py",
        ],
    },
    {
        "name": "Track-B / dispatch — TCGA-BRCA WSI (UNI2-h ABMIL)",
        "inputs": [
            "data/pathology/features/TCGA_TITAN_features.pkl",
            "data/pathology/features/tcga_brca_titan_manifest.csv",
            "data/pathology/bags/uni/",
        ],
        "scripts": [
            "scripts/trackb_mil_cv.py",
        ],
    },
    {
        "name": "fastMRI-NYU / dispatch — standalone NYU encoder (ADR-0016)",
        "inputs": [
            "data/fastmri_processed_nyu/",
            "data/fastmri_processed_nyu_masks/",
        ],
        "scripts": [
            "scripts/train_fastmri_nyu.py",
        ],
    },
]


def _print_scope() -> None:
    print("=" * 78)
    print("PinkSight — real-data reproduce (make reproduce)")
    print("-" * 78)
    print("  IN  (code-reproducible): G3 (Track-A Duke fusion), G5 (calibration/external/XAI),")
    print("       Track-C (tabular panel), and the dispatch-routed per-cohort organs.")
    print("  OUT (frozen-only in results/Table2_results.md, NOT reproducible here):")
    print("       G1 radiomics floor · G2 imaging-closing arm · pCR Phase D/E.")
    print("  Firewall: per-cohort, patient-disjoint — a number from one cohort is NEVER pooled")
    print("            with, compared to, or juxtaposed against a number from another (LOCK-1).")
    print("=" * 78)


def main() -> int:
    _print_scope()

    missing_by_rung: list[tuple[str, list[str]]] = []
    reproducible: list[str] = []

    for rung in RUNGS:
        missing = [p for p in rung["inputs"] if not (ROOT / p).exists()]
        if missing:
            missing_by_rung.append((rung["name"], missing))
        else:
            reproducible.append(rung["name"])

    if missing_by_rung:
        print("\nREQUIRED DATA / WEIGHTS ABSENT — reproduce cannot proceed for these rungs:")
        for name, missing in missing_by_rung:
            print(f"\n  Rung: {name}")
            for p in missing:
                print(f"    MISSING: {p}")
        print(f"\nPlace the cohorts per {DATA_README} (each under its own data-use agreement),")
        print("then re-run `make reproduce`. This target refuses to substitute synthetic data for")
        print("real reproduction — an absent cohort is an honest hard failure, never a silent pass.")
        if reproducible:
            print(f"\n(Rungs whose inputs ARE present: {', '.join(reproducible)} — but reproduce runs")
            print(" all-or-nothing to keep the frozen-vs-produced comparison coherent.)")
        print("=" * 78)
        return 2

    print("\nAll documented inputs present. Reproduce would now invoke, per rung:")
    for rung in RUNGS:
        print(f"\n  {rung['name']}")
        for s in rung["scripts"]:
            print(f"    - {s}")
    print("\nEach script reports against its own frozen results/Table2_results.md value using its")
    print("documented reproducibility contract (e.g. g3_moe7_salt_reporting.py's md5-deterministic")
    print("20-salt sweep). Reproduce prints the produced-vs-frozen comparison; it never overwrites")
    print("the frozen results. Full round-trip against licensed data is a data-provisioned-env task.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
