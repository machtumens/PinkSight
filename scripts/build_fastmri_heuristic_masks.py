
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from pinksight.data.fastmri_heuristic_mask import enhancement_mask
from pinksight.data.fastmri_nyu import LESION_BENIGN, LESION_MALIG, hchar_items
from pinksight.data.lesion_crop import derive_lesion_box
from pinksight.models.h0_localizer import RIM_MM_DEFAULT

PROC = Path("data/fastmri_processed_nyu")
MASKS = Path("data/fastmri_processed_nyu_masks")
DIAG_OUT = Path("process/general-plans/active/fastmri-lesion-crop_06-08-26/rung0_crop_diagnostics.json")
GRID = (96, 96, 96)  


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)  


def build_masks(limit: int | None) -> int:
    MASKS.mkdir(parents=True, exist_ok=True)
    cubes = sorted(PROC.glob("*.npy"))
    if limit:
        cubes = cubes[:limit]
    if not cubes:
        log(f"NO cached cubes in {PROC} — run scripts/preprocess_fastmri_nyu.py first.")
        return 0
    n = 0
    for p in cubes:
        out = MASKS / p.name
        if out.exists():
            continue
        cube = np.load(p)  
        if cube.ndim != 4 or cube.shape[0] < 2:
            log(f"  WARN {p.name}: unexpected shape {cube.shape} — need >=2 channels; skipping")
            continue
        mask = enhancement_mask(cube[0], cube[1])  
        np.save(out, mask)  
        n += 1
        if n % 25 == 0:
            log(f"    masks built {n}")
    log(f"heuristic masks: {n} newly built ({len(cubes)} cached patients, {MASKS} total) rim will be "
        f"applied downstream at rim_mm={RIM_MM_DEFAULT}")
    return n


def _box_volume(mask: np.ndarray) -> tuple[float, bool]:
    box, used_fallback = derive_lesion_box(mask, GRID, rim_mm=RIM_MM_DEFAULT)
    vol = float((box.row[1] - box.row[0]) * (box.col[1] - box.col[0]) * (box.slice[1] - box.slice[0]))
    return vol, bool(used_fallback)


def confound_diagnostic() -> dict:
    test_items = hchar_items()["test"]  
    per_class: dict[str, list[tuple[float, bool]]] = {"malignant": [], "benign": []}
    missing: list[str] = []
    for pid, label in test_items:
        mp = MASKS / f"{pid}.npy"
        if not mp.exists():
            missing.append(pid)
            continue
        vol, used_fb = _box_volume(np.load(mp))
        per_class["malignant" if int(label) == 1 else "benign"].append((vol, used_fb))

    cube_vol = float(GRID[0] * GRID[1] * GRID[2])
    table: dict[str, dict] = {}
    for cls, rows in per_class.items():
        n = len(rows)
        vols = np.array([v for v, _ in rows], dtype=np.float64)
        fbs = np.array([1.0 if fb else 0.0 for _, fb in rows], dtype=np.float64)
        table[cls] = {
            "n": n,
            "fallback_rate": round(float(fbs.mean()), 4) if n else None,
            "mean_box_volume": round(float(vols.mean()), 1) if n else None,
            "box_volume_std": round(float(vols.std()), 1) if n else None,
            "mean_box_fill_fraction": round(float(vols.mean() / cube_vol), 4) if n else None,
        }

    diag = {
        "task": "fastMRI-NYU Rung-0 lesion-crop confound diagnostic (NYU-INTERNAL; Section 2 rule 1)",
        "cohort": "sealed H-char test fold (malignant vs benign)",
        "grid": list(GRID),
        "rim_mm": RIM_MM_DEFAULT,
        "cube_voxels": cube_vol,
        "by_class": table,
        "missing_masks": missing,
        "interpretation": (
            "A large malignant-vs-benign gap in fallback_rate OR mean_box_volume is the confound "
            "signature (crop geometry correlated with the true label). Percentile-90 masks rarely "
            "produce a literally-empty mask, so if fallback_rate is ~0 for BOTH classes, read "
            "mean_box_volume / box_fill_fraction as the primary signal (plan Section 2 rule 1 note)."
        ),
    }
    return diag


def print_diagnostic(diag: dict) -> None:
    log("=" * 78)
    log("RUNG-0 CONFOUND DIAGNOSTIC (Section 2 rule 1 — unconditional disclosure)")
    log(f"  grid={diag['grid']} rim_mm={diag['rim_mm']} cube_voxels={diag['cube_voxels']:.0f}")
    hdr = f"  {'class':<10} {'n':>3} {'fallback_rate':>14} {'mean_box_vol':>14} {'box_vol_std':>12} {'fill_frac':>10}"
    log(hdr)
    log("  " + "-" * (len(hdr) - 2))
    for cls in ("malignant", "benign"):
        r = diag["by_class"][cls]
        log(f"  {cls:<10} {r['n']:>3} {str(r['fallback_rate']):>14} {str(r['mean_box_volume']):>14} "
            f"{str(r['box_volume_std']):>12} {str(r['mean_box_fill_fraction']):>10}")
    if diag["missing_masks"]:
        log(f"  MISSING masks for {len(diag['missing_masks'])} test patient(s): {diag['missing_masks'][:5]}")
    log("=" * 78)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="build masks for only the first N cubes")
    ap.add_argument("--diagnostic-only", action="store_true", help="skip building; just emit the diagnostic")
    args = ap.parse_args()

    if not args.diagnostic_only:
        build_masks(args.limit)

    diag = confound_diagnostic()
    print_diagnostic(diag)
    DIAG_OUT.parent.mkdir(parents=True, exist_ok=True)
    DIAG_OUT.write_text(json.dumps(diag, indent=2) + "\n")
    log(f"WROTE diagnostic -> {DIAG_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
