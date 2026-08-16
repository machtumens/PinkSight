
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  
SCRIPTS = ROOT / "scripts"
CLI = ROOT / "pinksight_cli.py"

SYNTHETIC_FOOTER = (
    "SYNTHETIC — NOT A RESULT. Forward-only plumbing/integrity proof on fabricated data; "
    "no real patient, no LOCK moved, no group-vs-group comparison, no biology."
)

DEMO_N = {
    "Track-B": 800,
    "Track-C": 4000,
    "Track-A": 600,
    "fastMRI-NYU": 600,
}

STREAMS = [
    ("Track-B", "e2e_synthetic_trackb_run.py", 1, "sklearn"),
    ("Track-C", "e2e_synthetic_trackc_run.py", 1, "sklearn"),
    ("Track-A", "e2e_synthetic_harness_run.py", 2, "torch"),
    ("fastMRI-NYU", "e2e_synthetic_fastmri_run.py", 2, "torch"),
]

TIER_EXTRA = {1: ".[arms]", 2: ".[ml]"}


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def run_tier0() -> bool:
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
