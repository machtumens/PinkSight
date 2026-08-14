#!/usr/bin/env python3
"""E2E synthetic validation kit — Deliverable B: companion run-of-runs index over the 4 model organs.

Forward-only, $0-local, CPU, no training-on-synthetic. Indexes the fused Track-A scorecard (READ from
Deliverable A's output — never re-run, to avoid paying the heavy fused 10k cost twice) BESIDE fresh
n=10,000 control-sentinel runs of the 3 other model organs (Track-B WSI+genomics, Track-C tabular,
fastMRI-NYU images-only encoder), producing ONE ``manifest_of_runs``.

This is a run-of-runs INDEX, never a pooled number, never a cross-organ delta/ranking/comparison. Each
organ's own control-sentinel verdict is listed SEPARATELY. The ``assert_no_cross_organ_pooling`` guard
(LOCK-1, mechanical) raises-before-write if any pooled/combined/delta/ranking key ever appears. No
metric off synthetic data is a scientific result; no LOCK is moved. Characterisation framing only.

Usage: uv run --extra ml python scripts/e2e_synthetic_companion_index_run.py --n-patients 10000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))  # so the sibling runner modules + `fva.*` resolve

import e2e_synthetic_fastmri_run as fastmri  # noqa: E402
import e2e_synthetic_fused_10k_validation_run as fused_kit  # noqa: E402
import e2e_synthetic_harness_run as harness  # noqa: E402
import e2e_synthetic_trackb_run as trackb  # noqa: E402
import e2e_synthetic_trackc_run as trackc  # noqa: E402

from pinksight.eval.e2e_report_contract import assert_no_cross_organ_pooling  # noqa: E402
from pinksight.seed import set_seed  # noqa: E402

DEFAULT_OUT = fused_kit.DEFAULT_OUT
DEFAULT_NEG_SCORECARD = DEFAULT_OUT / "e2e_synthetic_fused_10k_negative_control_control_scorecard.json"
DEFAULT_POS_SCORECARD = DEFAULT_OUT / "e2e_synthetic_fused_10k_positive_control_control_scorecard.json"

# fastMRI + fused runs use cube=16 / batch=64 by default (matches each runner's own main()).
CUBE_SIZE = 16
BATCH_SIZE = 64

_RUN_OF_RUNS_NOTE = (
    "run-of-runs index — each organ's own control-sentinel verdict is listed SEPARATELY. "
    "SYNTHETIC — NOT A RESULT; forward-only plumbing, no LOCK moved. This index is NEVER a pooled or "
    "combined number, a cross-organ delta, or a ranking/comparison (LOCK-1; enforced by "
    "assert_no_cross_organ_pooling)."
)


def _verdict(x: dict[str, Any]) -> dict[str, Any]:
    """E1: uniform verdict extractor. Track-B/C/fastMRI ``run_control()`` return a WRAPPED report dict
    (``{...,"controlVerdict": {...}}``); Deliverable A's loaded scorecard IS already the raw verdict.
    ``x.get("controlVerdict", x)`` returns the inner verdict for a wrapped report and ``x`` unchanged for
    a raw verdict — one helper, all 4 organs, no whole-wrapped-report leaking into manifest_of_runs."""
    return x.get("controlVerdict", x)


def assemble_manifest_of_runs(
    n: int,
    seed: int,
    fused_neg_scorecard: dict[str, Any],
    fused_pos_scorecard: dict[str, Any],
    *,
    cube_size: int = CUBE_SIZE,
    batch_size: int = BATCH_SIZE,
    git_commit: str = "unknown",
) -> dict[str, Any]:
    """Assemble the 4-organ ``manifest_of_runs`` (Public Contract #4). The fused Track-A verdicts are
    passed in (READ from Deliverable A, never re-run); Track-B/C/fastMRI are run FRESH here at ``n`` for
    both control streams. Organ keys reuse each runner's own ``ORGAN`` constant (imported, not re-typed).
    """
    # E2: pin torch BEFORE building the fastMRI encoder (LAW L-3), then reuse the ONE frozen encoder for
    # both fastMRI streams — mirrors e2e_synthetic_fastmri_run.py:153-154. Track-B/C need no encoder.
    set_seed(seed)
    encoder = fastmri.build_encoder()

    return {
        fused_kit.ORGAN: {
            "negative_control": _verdict(fused_neg_scorecard),
            "positive_control": _verdict(fused_pos_scorecard),
        },
        trackb.ORGAN: {
            "negative_control": _verdict(trackb.run_control("negative_control", n, seed, git_commit)),
            "positive_control": _verdict(trackb.run_control("positive_control", n, seed, git_commit)),
        },
        trackc.ORGAN: {
            "negative_control": _verdict(trackc.run_control("negative_control", n, seed, git_commit)),
            "positive_control": _verdict(trackc.run_control("positive_control", n, seed, git_commit)),
        },
        fastmri.ORGAN: {
            "negative_control": _verdict(
                fastmri.run_control(encoder, "negative_control", n, seed, cube_size, batch_size, git_commit)
            ),
            "positive_control": _verdict(
                fastmri.run_control(encoder, "positive_control", n, seed, cube_size, batch_size, git_commit)
            ),
        },
        "_generatedAt": datetime.now(timezone.utc).isoformat(),
        "_note": _RUN_OF_RUNS_NOTE,
    }


def _load_scorecard(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"fused control scorecard {p} not found — run scripts/e2e_synthetic_fused_10k_validation_run.py "
            "first (Deliverable A) so the fused Track-A scorecards exist to index."
        )
    return json.loads(p.read_text(encoding="utf-8"))


def _print_consort(manifest: dict[str, Any], out: Path) -> None:
    print("[companion-index] run-of-runs index — NOT a pooled comparison, NOT a cross-organ ranking:")  # noqa: T201
    n_organs = 0
    for organ, block in manifest.items():
        if not isinstance(block, dict):  # skip _generatedAt / _note metadata
            continue
        n_organs += 1
        neg, pos = block["negative_control"], block["positive_control"]
        print(  # noqa: T201
            f"  {organ}: neg verdict={neg.get('verdict')} "
            f"(auroc={neg.get('auroc', 'n/a')}, leakFreeByShuffle={neg.get('leakFreeByShuffle')}) | "
            f"pos verdict={pos.get('verdict')} (auroc={pos.get('auroc', 'n/a')})"
        )
    print(  # noqa: T201
        f"  wrote {out.name} — {n_organs} organs, each listed SEPARATELY "
        "(SYNTHETIC — NOT A RESULT; no LOCK moved)"
    )


def run(args: argparse.Namespace) -> int:
    """Load the 2 fused scorecards (READ), run Track-B/C/fastMRI fresh at ``args.n_patients``, assemble
    the manifest_of_runs, GATE it with assert_no_cross_organ_pooling (raise-before-write), and write."""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Self-contained fallback: if Deliverable A's fused scorecards are absent (fresh checkout /
    # run standalone), produce them once via the fused runner's own main(). When they already
    # exist we skip re-running — preserving the "don't pay the heavy fused 10k cost twice" design.
    # This fallback covers the DEFAULT scorecard paths only; custom --fused-scorecard-* paths are
    # the caller's responsibility.
    if not Path(args.fused_scorecard_neg).exists() or not Path(args.fused_scorecard_pos).exists():
        print("[companion-index] fused Track-A scorecards absent — running Deliverable A once to produce them...")  # noqa: T201
        fused_kit.main(["--n-patients", str(args.n_patients), "--seed", str(args.seed)])
    fused_neg = _load_scorecard(args.fused_scorecard_neg)
    fused_pos = _load_scorecard(args.fused_scorecard_pos)
    git_commit = harness.git_commit_short()

    t0 = time.time()
    manifest = assemble_manifest_of_runs(args.n_patients, args.seed, fused_neg, fused_pos, git_commit=git_commit)
    assert_no_cross_organ_pooling(manifest)  # LOCK-1 raise-before-write gate — no pooled/compared key lands

    out = out_dir / "e2e_synthetic_companion_index_manifest_of_runs.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _print_consort(manifest, out)
    print(f"  n={args.n_patients} per companion organ, {time.time() - t0:.1f}s")  # noqa: T201
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-patients", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fused-scorecard-neg", type=Path, default=DEFAULT_NEG_SCORECARD)
    ap.add_argument("--fused-scorecard-pos", type=Path, default=DEFAULT_POS_SCORECARD)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
