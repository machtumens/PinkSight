#!/usr/bin/env python3
"""PinkSight zero-data demo orchestrator — 3-tier, dependency-aware, no network, no data.

`make demo` invokes this. It proves the packaged pipeline is *runnable end-to-end* on a clean
clone with **zero data files and zero network access**, degrading gracefully by which optional
dependency extra the cloner installed:

  * **Tier 0** (stdlib, base `pip install -e .`, ALWAYS runs): `pinksight_cli.py --selfcheck` —
    exercises the backend / dispatch / ledger-guard *wiring* (the forbidden-term firewall scan).
    If Tier 0 fails the base install itself is broken, so nothing heavier is attempted.
  * **Tier 1** (`pip install -e '.[arms]'` — scikit-learn + lightgbm, **no torch**): the Track-B
    (WSI+genomics) and Track-C (tabular) synthetic control-sentinels.
  * **Tier 2** (`pip install -e '.[ml]'` — torch + monai + sklearn, heavy): the Track-A and
    fastMRI-NYU synthetic control-sentinels.

Every number printed here is **SYNTHETIC — NOT A RESULT**: forward-only plumbing/integrity proof,
no real patient, no LOCK moved, no biology. A control-sentinel stream PASSES when BOTH of its
controls behave: the *negative* control's real signal collapses to the shuffle (label-free) floor
(DeLong CI crosses 0.50), and the *positive* control recovers its known injected signal while its
shuffle companion collapses to ~0.50.

Streams whose extra is not installed print an explicit `SKIPPED (needs pip install -e '.[…]')`
line — they NEVER crash the run and NEVER count as a real failure. Exit code is 0 iff Tier 0
passed AND every dependency-satisfied stream that actually ran is PASS.

Usage:
    python3 scripts/run_demo.py [--n-patients N] [--seed S]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # scripts/ -> repo root
SCRIPTS = ROOT / "scripts"
CLI = ROOT / "pinksight_cli.py"

SYNTHETIC_FOOTER = (
    "SYNTHETIC — NOT A RESULT. Forward-only plumbing/integrity proof on fabricated data; "
    "no real patient, no LOCK moved, no group-vs-group comparison, no biology."
)

# Per-stream demo patient counts, tuned empirically at seed 0 (deterministic — verified byte-stable
# across repeat runs) against e2e_report_contract's control-sentinel PASS bars: positive real AUROC
# > 0.75, positive shuffle <= 0.55, negative DeLong CI crosses 0.50. Track-C (9-feature tabular) is
# cheap, so it runs LARGE for a robust shuffle collapse (~11s); Track-B (1542-dim WSI+genomics) is
# slow, so it runs SMALL — it still clears every bar deterministically at seed 0 (~36s). Tier-2
# (torch) counts are best-effort defaults, UNVERIFIED in a torch-free sandbox (see the phase-3
# EXECUTE T10 known-gap). `--n-patients` overrides ALL of these.
DEMO_N = {
    "Track-B": 800,
    "Track-C": 4000,
    "Track-A": 600,
    "fastMRI-NYU": 600,
}

# (label, script filename, tier, dependency module that must be importable)
STREAMS = [
    ("Track-B", "e2e_synthetic_trackb_run.py", 1, "sklearn"),
    ("Track-C", "e2e_synthetic_trackc_run.py", 1, "sklearn"),
    ("Track-A", "e2e_synthetic_harness_run.py", 2, "torch"),
    ("fastMRI-NYU", "e2e_synthetic_fastmri_run.py", 2, "torch"),
]

TIER_EXTRA = {1: ".[arms]", 2: ".[ml]"}


def _has(module: str) -> bool:
    """True iff `module` is importable, with NO import side effects (find_spec never imports)."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def run_tier0() -> bool:
    """Run the stdlib CLI selfcheck. Returns True on exit 0. Prints its own result line."""
    print("── Tier 0 (stdlib, base install) ─────────────────────────────────────────────")
    proc = subprocess.run(
        [sys.executable, str(CLI), "--selfcheck"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    out = (proc.stdout or "").strip()
    ok = proc.returncode == 0
    tag = "PASS" if ok else "FAIL"
    print(f"  [selfcheck]  {tag}  (exit {proc.returncode})  {out.splitlines()[-1] if out else ''}")
    if not ok:
        err = (proc.stderr or "").strip()
        print(f"  Tier-0 selfcheck FAILED — base wiring is broken; aborting before heavier tiers.")
        if err:
            print("  " + "\n  ".join(err.splitlines()[-6:]))
    return ok


def _load_verdicts(out_dir: Path) -> list[dict]:
    """Collect every controlVerdict block written into out_dir (top-level *.json only)."""
    verdicts = []
    for jf in sorted(out_dir.glob("*.json")):
        try:
            payload = json.loads(jf.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        cv = payload.get("controlVerdict")
        if isinstance(cv, dict) and cv:
            verdicts.append(cv)
    return verdicts


def run_stream(label: str, script: str, n: int, seed: int, out_dir: Path) -> dict:
    """Run one synthetic control-sentinel script as a subprocess; return a result dict.

    result = {label, status: PASS|FAIL, controls: [ {stream, verdict, auroc, shuffle}... ], note}
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            sys.executable, str(SCRIPTS / script),
            "--n-patients", str(n),
            "--seed", str(seed),
            "--out-dir", str(out_dir),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        err = (proc.stderr or "").strip()
        note = err.splitlines()[-1] if err else f"exit {proc.returncode}"
        return {"label": label, "status": "FAIL", "controls": [], "note": note}

    verdicts = _load_verdicts(out_dir)
    if not verdicts:
        return {"label": label, "status": "FAIL", "controls": [],
                "note": "ran clean but wrote no controlVerdict JSON"}

    controls = []
    all_pass = True
    for cv in verdicts:
        v = str(cv.get("verdict", "NOT_ASSESSED"))
        controls.append({
            "stream": cv.get("stream", cv.get("controlType", "?")),
            "verdict": v,
            "auroc": cv.get("auroc"),
            "shuffle": cv.get("shuffleAuroc"),
            "leakFree": cv.get("leakFreeByShuffle"),
        })
        if v != "PASS":
            all_pass = False
    return {"label": label, "status": "PASS" if all_pass else "FAIL",
            "controls": controls, "note": ""}


def _fmt(x) -> str:
    return f"{x:.4f}" if isinstance(x, (int, float)) else str(x)


def main() -> int:
    ap = argparse.ArgumentParser(description="PinkSight zero-data 3-tier demo orchestrator.")
    ap.add_argument("--n-patients", type=int, default=None,
                    help="override synthetic patients for ALL streams (default: per-stream tuned "
                         "values — Track-C 4000, Track-B 800, Tier-2 600)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t_start = time.time()
    print("=" * 78)
    print("PinkSight — zero-data synthetic demo (make demo)")
    n_desc = str(args.n_patients) if args.n_patients else "per-stream (Track-C=4000, Track-B=800, Tier-2=600)"
    print(f"  n-patients={n_desc}  seed={args.seed}  network=NONE  data=NONE")
    print("=" * 78)

    tier0_ok = run_tier0()
    if not tier0_ok:
        print("\n" + SYNTHETIC_FOOTER)
        return 1

    have = {1: _has("sklearn"), 2: _has("torch")}
    results: list[dict] = []
    skipped: list[tuple[str, int]] = []

    with tempfile.TemporaryDirectory(prefix="pinksight_demo_") as td:
        tmp = Path(td)
        for label, script, tier, dep in STREAMS:
            if not have[tier]:
                skipped.append((label, tier))
                continue
            n = args.n_patients if args.n_patients else DEMO_N[label]
            print(f"\n── {label} (Tier {tier}, control-sentinel, n={n}) ──────────────────────")
            res = run_stream(label, script, n, args.seed, tmp / label.replace("/", "_"))
            results.append(res)
            if res["controls"]:
                for c in res["controls"]:
                    print(f"  {c['stream']:<18} verdict={c['verdict']:<12} "
                          f"real={_fmt(c['auroc'])}  shuffle={_fmt(c['shuffle'])}  "
                          f"leak_free={c['leakFree']}")
            else:
                print(f"  {label}: FAIL — {res['note']}")

    for label, tier in skipped:
        print(f"\n── {label} (Tier {tier}) ──")
        print(f"  SKIPPED (needs pip install -e '{TIER_EXTRA[tier]}')")

    # ── Summary table ──────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("-" * 78)
    print(f"  {'Tier 0 selfcheck':<22} {'PASS':<10}")
    ran_fail = False
    for res in results:
        print(f"  {res['label']:<22} {res['status']:<10} {res['note']}")
        if res["status"] == "FAIL":
            ran_fail = True
    for label, tier in skipped:
        print(f"  {label:<22} {'SKIPPED':<10} needs pip install -e '{TIER_EXTRA[tier]}'")
    print("-" * 78)
    print(f"  elapsed: {time.time() - t_start:.1f}s")
    print("=" * 78)
    print(SYNTHETIC_FOOTER)
    print("NOTE: this is a fast smoke-proof, NOT the full n=10,000 validation-kit run "
          "(scripts/e2e_synthetic_fused_10k_validation_run.py).")

    return 1 if ran_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
