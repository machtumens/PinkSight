#!/usr/bin/env python3
"""E2E synthetic validation kit — Deliverable A: the fused Track-A model at n=10,000 per control stream.

Forward-only, $0-local, CPU, no training-on-synthetic. Runs the REAL Track-A fusion stack
(``FusionModel`` via ``e2e_synthetic_harness_run.run_stream``, ``realistic=True`` — Duke-like clinical
marginals, SYN-A-* IDs) forward-only over an n=10,000 synthetic MRI+clinical cohort, for BOTH the
negative and positive control stream, and writes per stream: a per-sample JSONL (one row per
forward-passed patient + a self-describing header row), the existing single-representative
``CharacterisationReport``, and a standalone control scorecard. Both streams' generation manifests are
written to one cohort-manifest file.

This is PLUMBING / THROUGHPUT / INTEGRITY-CONTROL proof ONLY: no metric off synthetic data is a
scientific result, no LOCK is moved. Characterisation / at-diagnosis / stratification framing only. The
pipeline is IMPORTED and called FROZEN/forward-only (DD-2 single-source-of-truth) — this kit adds no
model or dataset-generation code.

Per-sample honesty (D2/D3, restated in every JSONL header): the ``nottinghamGrade*`` columns are the
EXISTING constant honest placeholder (NO trained grade head — Head-2 grade was DROPPED at G2), and
``ki67Stratum`` is the constant ``"not_assessed"`` (no calibrated @14% threshold — G0 Ki-67 N=0);
``ki67RawValue`` and ``subtypeProbability`` are the genuine per-patient forward-pass outputs.

Usage: uv run --extra ml python scripts/e2e_synthetic_fused_10k_validation_run.py --n-patients 10000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))  # so `import e2e_synthetic_harness_run` + `fva.*` resolve

import e2e_synthetic_harness_run as harness  # noqa: E402

from pinksight.eval.e2e_report_contract import (  # noqa: E402
    SYNTHETIC_TAG,
    assert_synthetic_provenance,
)

ORGAN = "fused-track-a-realistic"
DEFAULT_OUT = ROOT / "process/general-plans/active/fused-10k-validation-kit_08-08-26"

# Self-describing header note (D2/D3): makes the JSONL honest even if extracted out of context — nobody
# downstream mistakes a constant placeholder column for a real per-patient head signal.
_JSONL_NOTE = (
    "D2: nottinghamGrade* is the EXISTING constant honest placeholder (label NHG1, probability 0.5) — "
    "there is NO trained Nottingham-grade head (Head-2 grade was DROPPED at G2); this column is constant "
    "across every row by construction, never a per-patient signal. "
    "D3: ki67Stratum is the constant 'not_assessed' (no calibrated @14% threshold — G0 Ki-67 N=0); "
    "ki67RawValue is the raw untrained Ki-67 head forward-pass value (varies per patient — tensor-wiring "
    "proof, descriptive/untrained, NOT a stratification). subtypeProbability is the real per-patient "
    "forward-pass sigmoid (varies per patient). SYNTHETIC — NOT A RESULT."
)


def _per_sample_rows(res: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """One JSON-serialisable row per forward-passed patient (Public Contract #3). ``subtypeProbability``
    and ``ki67RawValue`` are the genuine per-patient forward-pass outputs; the grade + ki67 stratum
    fields are the documented constant honest placeholders."""
    pids = res["patient_ids"]
    probs = np.asarray(res["per_patient_subtype_prob"], dtype=float)
    ki67 = np.asarray(res["per_patient_ki67_raw"], dtype=float)
    for pid, p, k in zip(pids, probs, ki67):
        yield {
            "pid": str(pid),
            "subtypeLabel": "Triple-Negative" if p > 0.5 else "Luminal A",
            "subtypeProbability": round(float(p), 6),
            "ki67RawValue": round(float(k), 6),
            "ki67Stratum": "not_assessed",
            "nottinghamGradeLabel": "NHG1",
            "nottinghamGradeProbability": 0.5,
            "datasetTag": SYNTHETIC_TAG,
        }


def _write_jsonl(path: Path, res: dict[str, Any], stream_name: str) -> int:
    """Write the per-sample JSONL: a self-describing header row (D2/D3 note + manifest hash) followed by
    one data row per forward-passed patient. Returns the number of DATA rows written."""
    n = len(res["patient_ids"])
    header = {
        "_manifest": True,
        "_note": _JSONL_NOTE,
        "manifestSha256": res["manifest"]["manifest_sha256"],
        "n": n,
        "stream": stream_name,
        "datasetTag": SYNTHETIC_TAG,
    }
    written = 0
    with path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(header) + "\n")
        for row in _per_sample_rows(res):
            fh.write(json.dumps(row) + "\n")
            written += 1
    return written


def _consort_line(stream_name: str, res: dict[str, Any]) -> str:
    cv = res["control_verdict"]
    return (
        f"[{ORGAN}] {stream_name}: n={len(res['patient_ids'])} forward-passed, "
        f"control={cv.get('verdict')} (auroc={cv.get('auroc', 'n/a')}, "
        f"shuffle={cv.get('shuffleAuroc', 'n/a')}, leakFreeByShuffle={cv.get('leakFreeByShuffle')}), "
        f"wall_clock={res['wall_clock_s']}s — SYNTHETIC — NOT A RESULT."
    )


def run(args: argparse.Namespace) -> int:
    """Run both control streams of the fused Track-A model at ``args.n_patients`` and write the 3
    per-stream artifacts + the cohort manifest. Provenance-gated before every write."""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    git_commit = harness.git_commit_short()
    manifests: dict[str, Any] = {}

    for stream_name in ("negative_control", "positive_control"):
        res = harness.run_stream(
            stream_name, n=args.n_patients, seed=args.seed, cube_size=args.cube_size,
            channels=args.channels, effect_size=args.effect_size, xai_subsample=args.xai_subsample,
            batch_size=args.batch_size, out_dir=out_dir, git_commit=git_commit, realistic=True,
        )
        # Gate before write (firewall #2): the report's datasetTag + manifest hash must match.
        assert_synthetic_provenance(res["report"], res["manifest"]["manifest_sha256"])

        report_path = out_dir / f"e2e_synthetic_fused_10k_{stream_name}_report.json"
        report_path.write_text(json.dumps(res["report"], indent=2), encoding="utf-8")

        jsonl_path = out_dir / f"e2e_synthetic_fused_10k_{stream_name}_per_sample.jsonl"
        n_rows = _write_jsonl(jsonl_path, res, stream_name)

        scorecard_path = out_dir / f"e2e_synthetic_fused_10k_{stream_name}_control_scorecard.json"
        scorecard_path.write_text(json.dumps(res["control_verdict"], indent=2), encoding="utf-8")

        manifests[stream_name] = res["manifest"]
        print(_consort_line(stream_name, res))  # noqa: T201
        print(f"  wrote {report_path.name}, {jsonl_path.name} ({n_rows} data rows), "  # noqa: T201
              f"{scorecard_path.name}")

    manifest_path = out_dir / "e2e_synthetic_fused_10k_cohort_manifest.json"
    manifest_path.write_text(json.dumps(manifests, indent=2), encoding="utf-8")
    print(f"  wrote {manifest_path.name} — run-of-streams generation manifests "  # noqa: T201
          "(SYNTHETIC — NOT A RESULT; forward-only plumbing, no LOCK moved)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-patients", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cube-size", type=int, default=16)
    ap.add_argument("--channels", type=str, default="pre_post")
    ap.add_argument("--xai-subsample", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--effect-size", type=float, default=1.2)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
