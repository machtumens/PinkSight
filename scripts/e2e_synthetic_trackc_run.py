#!/usr/bin/env python3
"""E2E synthetic plumbing harness — Track-C tabular organ (standalone LightGBM panel; ADR-0010).

Forward-only, $0-local, CPU, no training-on-synthetic. Runs a synthetic Track-C sub-cohort through the
SAME control-sentinel spine every organ shares (``coalition_oof`` real+shuffle -> ``control_verdict``),
for both the negative and positive control, and writes one non-reportable report JSON per control.

This is PLUMBING / INTEGRITY-CONTROL proof ONLY: no metric off synthetic data is a scientific result,
no LOCK is moved. Track-C framing (detection/incidence) is the ADR-0010 path-scoped carve-out; the
number here proves tabular-panel wiring, never biology.

Usage: uv run python scripts/e2e_synthetic_trackc_run.py --n-patients 10000
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
    COIMBRA_FEATURES,
    DEFAULT_EFFECT_SIZE,
    build_stream_manifest,
    build_stream_report,
    generate_tabular_stream,
)
from pinksight.eval.e2e_report_contract import (  # noqa: E402
    assert_synthetic_provenance,
    control_verdict,
)

ORGAN = "trackc-coimbra"


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:  # noqa: BLE001 — provenance is best-effort; 'unknown' is honest if git is absent
        return "unknown"


def run_control(stream_name: str, n: int, seed: int, git_commit: str) -> dict:
    """Generate one Track-C control sub-cohort, run the real+shuffle sentinel, and assemble a gated
    non-reportable report. ``stream_name`` is the control type ('negative_control'/'positive_control')."""
    effect = 0.0 if stream_name == "negative_control" else DEFAULT_EFFECT_SIZE
    pids, x, y = generate_tabular_stream(n, seed=seed, feature_names=COIMBRA_FEATURES, effect_size=effect)

    # SAME forward spine as every organ: LogReg over StratifiedGroupKFold. Each patient is its own group
    # (unique IDs) -> patient-disjoint folds are guaranteed. One numeric matrix, no impute. The real OOF
    # is a single pass; the shuffle sentinel is the permutation-null MEAN (seed-robust, not one draw).
    real_oof = coalition_oof([x], [False], y, pids, seed=seed, shuffle=False)
    shuffle_oof = permutation_null_oof([x], [False], y, pids)
    verdict = control_verdict(stream_name, y=y, real_oof=real_oof, shuffle_oof=shuffle_oof)

    config = {"organ": ORGAN, "stream_name": stream_name, "n": n, "seed": seed,
              "effect_size": effect, "git_commit": git_commit}
    manifest = build_stream_manifest(config)
    report = build_stream_report(ORGAN, stream_name, manifest, COIMBRA_FEATURES, verdict)
    assert_synthetic_provenance(report, manifest["manifest_sha256"])  # firewall #2: gate before returning
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-patients", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path,
                    default=ROOT / "process/general-plans/active/synthetic-all-streams-e2e_08-08-26")
    args = ap.parse_args()

    git_commit = _git_commit()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for stream_name in ("negative_control", "positive_control"):
        report = run_control(stream_name, args.n_patients, args.seed, git_commit)
        out = args.out_dir / f"e2e_synthetic_trackc_{stream_name}.json"
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        v = report["controlVerdict"]
        print(f"[{ORGAN}] {stream_name}: verdict={v['verdict']} "  # noqa: T201
              f"auroc={v.get('auroc')} shuffle={v.get('shuffleAuroc')} -> {out.name}")
    print(f"[{ORGAN}] done, n={args.n_patients}, {time.time() - t0:.1f}s "  # noqa: T201
          "(SYNTHETIC — NOT A RESULT; forward-only plumbing, no LOCK moved)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
