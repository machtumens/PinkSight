#!/usr/bin/env python3
"""E2E synthetic plumbing harness — Track-B WSI(bag) + 6-gene genomics organ (standalone; LOCK-1).

Forward-only, $0-local, CPU, no training-on-synthetic. Runs a synthetic Track-B sub-cohort through the
SAME control-sentinel spine every organ shares (mean-pool the frozen tile-bag -> coalition_oof real +
permutation-null shuffle -> ``control_verdict``), for both the negative and positive control, and
writes one non-reportable report JSON per control.

This is PLUMBING / INTEGRITY-CONTROL proof ONLY: no metric off synthetic data is a scientific result,
no LOCK is moved. Intra-cohort characterisation ONLY — LOCK-1 cross-institution generalisation stays
forbidden and no synthetic number is ever pooled or juxtaposed with a Duke (or any real) number. The
WSI bag is reduced by mean-pooling (a forward-only, no-training reduction — never a gradient-trained
MIL); the number proves WSI-bag + genomics fusion wiring, never biology.

Usage: uv run python scripts/e2e_synthetic_trackb_run.py --n-patients 10000
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))  # so `from fva.shuffle_sentinel import ...` resolves

from e2e_synthetic_common import permutation_null_oof  # noqa: E402
from fva.shuffle_sentinel import coalition_oof  # noqa: E402

from pinksight.data.synthetic_streams import (  # noqa: E402
    _B_NTILES_RANGE,
    MODALITY_C_GENES,
    build_stream_manifest,
    build_stream_report,
    generate_wsi_genomics_stream,
)
from pinksight.eval.e2e_report_contract import (  # noqa: E402
    assert_synthetic_provenance,
    control_verdict,
)

ORGAN = "trackb-wsi-genomics"
# Disclosed positive-control effect size (a deliberate small effect, NOT a leak). Calibrated so the
# mean-pooled bag + gene panel recovers real AUROC well above 0.75 while the permutation-null shuffle
# collapses to chance — see pinksight.data.synthetic_streams._B_GENE_SHIFT / _B_BAG_OFFSET.
DEFAULT_EFFECT_SIZE = 0.5
BAG_FEATURE = "wsi_bag_meanpool_1536d"  # images-of-tissue descriptor for the report `features` block


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:  # noqa: BLE001 — provenance is best-effort; 'unknown' is honest if git is absent
        return "unknown"


def _pool_stream(gen) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mean-pool each frozen tile-bag to a (1536,) embedding IMMEDIATELY and DISCARD the raw bag
    (firewall #5: no bulk (n, n_tiles, 1536) array ever persisted). Mean-pool is a forward-only, no
    training reduction (never a gradient-trained MIL). Returns (bag_emb, genes, y, pids)."""
    bag_embs: list[np.ndarray] = []
    gene_rows: list[np.ndarray] = []
    labels: list[int] = []
    pids: list[str] = []
    for pid, bag, genes, label in gen:
        bag_embs.append(bag.mean(axis=0).astype(np.float32))  # (n_tiles, 1536) -> (1536,), bag discarded
        gene_rows.append(genes)
        labels.append(label)
        pids.append(pid)
    return (
        np.stack(bag_embs, axis=0),
        np.stack(gene_rows, axis=0),
        np.asarray(labels, dtype=int),
        np.asarray(pids),
    )


def run_control(
    stream_name: str,
    n: int,
    seed: int,
    git_commit: str,
    effect_size: float = DEFAULT_EFFECT_SIZE,
    n_tiles_range: tuple[int, int] = _B_NTILES_RANGE,
) -> dict:
    """Generate one Track-B control sub-cohort, mean-pool the bags, run the real + permutation-null
    sentinel over the [bag_emb, gene_panel] coalition, and assemble a gated non-reportable report."""
    effect = 0.0 if stream_name == "negative_control" else effect_size
    gen = generate_wsi_genomics_stream(n, seed=seed, effect_size=effect, n_tiles_range=n_tiles_range)
    bag_emb, genes, y, pids = _pool_stream(gen)
    if not (np.isfinite(bag_emb).all() and np.isfinite(genes).all()):
        raise ValueError("non-finite bag/gene matrix — Stream-B plumbing bug (hard fail, not degraded)")

    # SAME forward spine as every organ: LogReg over StratifiedGroupKFold, each patient its own group
    # (unique IDs) -> patient-disjoint folds. The real OOF is a single pass; the shuffle sentinel is the
    # permutation-null MEAN (seed-robust, not one draw). Two numeric matrices [bag_emb, genes], no impute.
    real_oof = coalition_oof([bag_emb, genes], [False, False], y, pids, seed=seed, shuffle=False)
    shuffle_oof = permutation_null_oof([bag_emb, genes], [False, False], y, pids)
    verdict = control_verdict(stream_name, y=y, real_oof=real_oof, shuffle_oof=shuffle_oof)

    features = list(MODALITY_C_GENES) + [BAG_FEATURE]
    config = {
        "organ": ORGAN, "stream_name": stream_name, "n": n, "seed": seed, "effect_size": effect,
        "git_commit": git_commit, "n_tiles_min": int(n_tiles_range[0]), "n_tiles_max": int(n_tiles_range[1]),
    }
    manifest = build_stream_manifest(config)
    report = build_stream_report(ORGAN, stream_name, manifest, features, verdict)
    assert_synthetic_provenance(report, manifest["manifest_sha256"])  # firewall #2: gate before returning
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-patients", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--effect-size", type=float, default=DEFAULT_EFFECT_SIZE)
    ap.add_argument("--out-dir", type=Path,
                    default=ROOT / "process/general-plans/active/synthetic-all-streams-e2e_08-08-26")
    args = ap.parse_args()

    git_commit = _git_commit()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for stream_name in ("negative_control", "positive_control"):
        report = run_control(stream_name, args.n_patients, args.seed, git_commit, args.effect_size)
        out = args.out_dir / f"e2e_synthetic_trackb_{stream_name}.json"
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        v = report["controlVerdict"]
        print(f"[{ORGAN}] {stream_name}: verdict={v['verdict']} "  # noqa: T201
              f"auroc={v.get('auroc')} shuffle={v.get('shuffleAuroc')} -> {out.name}")
    print(f"[{ORGAN}] done, n={args.n_patients}, {time.time() - t0:.1f}s "  # noqa: T201
          "(SYNTHETIC — NOT A RESULT; forward-only plumbing, no LOCK moved)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
