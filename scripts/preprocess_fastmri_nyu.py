"""fastMRI-NYU preprocessing (ADR-0016): DICOM -> per-patient DCE cube cache + train-only Nyul.

Disk-safe by construction: the two DCM tars are ~44 GB + ~19 GB uncompressed (63 GB) but only ~74 GB
is free, so we extract ONE tar at a time, cache each patient's small resized cube, then delete that
tar's raw extraction before touching the next. Peak raw-on-disk ≈ one 44 GB tar. Resumable: a patient
whose work cube already exists is skipped.

Pipeline per patient (NYU-only; no Duke):
    frame-grouped DCE phases (pre first)  ->  trilinear resize to a fixed CUBE (channels = phases)
    ->  work cube  data/fastmri_work/{pid}.npy  (pre-Nyul, (C, S, S, S) float32)
Then, ONCE all work cubes exist:
    fit Nyul standard histogram on TRAIN-fold pre-contrast (channel 0) ONLY (LOCK-2)
    ->  configs/nyul_fastmri_nyu_v1.npy
    apply Nyul per channel  ->  processed cube  data/fastmri_processed_nyu/{pid}.npy

Deviations from the Duke LOCK-3 chain (within blast radius; documented in the task REPORT):
  * N4 bias correction DROPPED — full-res N4 on 300*4 DCE volumes is ~tens of hours on $0-local; a
    whole-volume characterisation CNN tolerates residual bias (Nyul + per-channel z-norm normalize).
  * No lesion-ROI crop — NYU ships no annotation boxes / masks; the encoder eats the whole breast cube.
  * resample_iso folded into the cube resize (native grid is already ~1mm-iso: (1.0,1.0,1.1)).
The load-bearing LOCK-2 guard (Nyul fit on TRAIN only) is RETAINED.

    PYTHONPATH=src .venv/bin/python scripts/preprocess_fastmri_nyu.py                 # full 300
    PYTHONPATH=src .venv/bin/python scripts/preprocess_fastmri_nyu.py --limit 8       # smoke subset
    PYTHONPATH=src .venv/bin/python scripts/preprocess_fastmri_nyu.py --nyul-only     # (re)fit+apply Nyul
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from pinksight.data.fastmri_nyu import (
    COL_PID,
    SPLIT_TRAIN,
    SPLIT_TEST,
    COL_SPLIT,
    load_labels,
    load_quarantine,
)
from pinksight.data.fastmri_series import find_scan_dir, load_dce_phases, to_rcs

DATA = Path("data/fastmri_breast")
RAW = Path("data/fastmri_raw")
WORK = Path("data/fastmri_work")
PROC = Path("data/fastmri_processed_nyu")
NYUL = Path("configs/nyul_fastmri_nyu_v1.npy")
SHA_FILE = DATA / "SHA256"
TARS = {
    "fastMRI_breast_IDS_001_150_DCM.tar.gz": range(1, 151),
    # ranges overlap at 150 by design: whichever archive actually holds patient 150 wins; the other
    # attempt is a harmless cache-hit / caught-WARN. Closes the archive-naming boundary ambiguity.
    "fastMRI_breast_IDS_150_300_DCM.tar.gz": range(150, 301),
}
CUBE = 96  # cached cube edge (training may resize down for VRAM)
MAX_PHASES = 4  # fastMRI breast DCE ships 4 frames (pre + 3 post) — cap for a fixed channel count


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)  # noqa: T201


def sha256(path: Path, buf: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def expected_hashes() -> dict[str, str]:
    out: dict[str, str] = {}
    for line in SHA_FILE.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] in TARS:
            out[parts[1]] = parts[0]
    return out


def verify_tar(name: str, exp: dict[str, str]) -> None:
    if name not in exp:
        raise ValueError(f"{name} not listed in {SHA_FILE}")
    got = sha256(DATA / name)
    if got != exp[name]:
        raise ValueError(f"SHA256 MISMATCH for {name}: got {got[:16]}.. expected {exp[name][:16]}..")
    log(f"  SHA256 OK: {name}")


def _resize_cube(vol_rcs_channels: list[np.ndarray], cube: int) -> np.ndarray:
    """Stack per-phase (R,C,S) arrays -> (C, R, C, S), trilinear-resize to (C, cube, cube, cube)."""
    from monai.transforms import Resize

    x = np.stack(vol_rcs_channels, axis=0).astype(np.float32)  # (C, R, C, S)
    resize = Resize(spatial_size=(cube, cube, cube), mode="trilinear", align_corners=False)
    return np.ascontiguousarray(resize(x), dtype=np.float32)


def build_work_cube(pid: str, extract_root: Path, cube: int) -> np.ndarray:
    scan_dir = find_scan_dir(extract_root, pid)  # de-dups repeats (lowest scan index)
    phases = load_dce_phases(scan_dir, max_phases=MAX_PHASES)  # pre first
    return _resize_cube([to_rcs(p) for p in phases], cube)


def extract_tar(name: str) -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    log(f"  extracting {name} -> {RAW} (this writes tens of GB; deleted after caching)")
    subprocess.run(["tar", "-xzf", str(DATA / name), "-C", str(RAW)], check=True)
    return RAW


def process_all(pids: list[str], cube: int, limit: int | None) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    quarantine = load_quarantine()
    exp = expected_hashes()
    want = set(pids if limit is None else pids[:limit])
    done_total, skipped_norm = 0, 0
    for name, id_range in TARS.items():
        tar_pids = [p for p in want if int(p.split("_")[-1]) in id_range]
        if not tar_pids:
            continue
        # skip whole tar if every wanted cube already cached (resumable)
        if all((WORK / f"{p}.npy").exists() for p in tar_pids):
            log(f"  all {len(tar_pids)} cubes for {name} already cached — skipping extract")
            done_total += len(tar_pids)
            continue
        verify_tar(name, exp)
        root = extract_tar(name)
        try:
            for i, pid in enumerate(sorted(tar_pids)):
                out = WORK / f"{pid}.npy"
                if out.exists():
                    continue
                if pid in quarantine:
                    # cache the cube anyway (H6/deferred could use it) BUT it is never in an H-char/H5
                    # item list — the loader-level assert_no_normals is the interlock. Kept for parity.
                    skipped_norm += 1
                try:
                    cube_arr = build_work_cube(pid, root, cube)
                    np.save(out, cube_arr)
                except Exception as e:  # noqa: BLE001 — one bad patient must not kill a detached run
                    log(f"    WARN {pid}: {type(e).__name__}: {str(e)[:120]}")
                    continue
                done_total += 1
                if (i + 1) % 10 == 0:
                    log(f"    {name}: {i + 1}/{len(tar_pids)} cached (total {done_total})")
        finally:
            log(f"  deleting raw extraction {root} to reclaim disk")
            shutil.rmtree(root, ignore_errors=True)
    log(f"work cubes done: {done_total} cached ({skipped_norm} normals cached but quarantined from items)")


def fit_and_apply_nyul(limit: int | None) -> None:
    """Fit Nyul on TRAIN pre-contrast (channel 0) ONLY (LOCK-2), then apply per channel -> processed."""
    from pinksight.data.preprocess import _apply_nyul, fit_nyul

    df = load_labels()
    train_pids = set(df.loc[df[COL_SPLIT] == SPLIT_TRAIN, COL_PID].astype(str))
    test_pids = set(df.loc[df[COL_SPLIT] == SPLIT_TEST, COL_PID].astype(str))
    quarantine = load_quarantine()
    work = sorted(WORK.glob("*.npy"))
    if limit:
        work = work[:limit]
    # LOCK-2: fit ONLY on train-fold, non-quarantined cubes.
    fit_paths = [p for p in work if p.stem in train_pids and p.stem not in quarantine]
    fit_pids = {p.stem for p in fit_paths}
    # LOCK-2 HARD GATE (ADR-0016 Step 0): the Nyul fit must consume ZERO test-fold IDs. A preprocessing
    # statistic (landmarks) fit on ANY test patient leaks the sealed-50 and inflates the reported AUROC.
    # The filter above already excludes test IDs; this assert PROVES it mechanically (not honor-system).
    # If it cannot hold, the run is BLOCKED rather than producing a leaky number.
    leaked_test = sorted(fit_pids & test_pids)
    if leaked_test:
        raise AssertionError(
            f"LOCK-2 VIOLATION: Nyul fit would consume {len(leaked_test)} TEST-fold patient ID(s) "
            f"{leaked_test[:5]} — preprocessing stats MUST be TRAIN-fold only. Refusing to fit."
        )
    if len(fit_paths) < 5:
        log(f"  only {len(fit_paths)} train cubes — Nyul fit skipped (need the full preprocess first)")
        return
    log(f"  LOCK-2 Nyul-fit ASSERT PASS: {len(fit_paths)} TRAIN-fold cubes, 0 of {len(test_pids)} "
        f"test IDs consumed; {len(quarantine)} normals quarantined out of the fit.")
    log(f"  fitting Nyul on {len(fit_paths)} TRAIN pre-contrast cubes (LOCK-2 train-only)")
    fit_vols = [np.load(p)[0] for p in fit_paths]
    nyul = fit_nyul(fit_vols)
    NYUL.parent.mkdir(parents=True, exist_ok=True)
    nyul.save_standard_histogram(str(NYUL))
    log(f"  Nyul landmarks -> {NYUL}")

    PROC.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in work:
        cube = np.load(p)  # (C, S, S, S)
        out = np.stack([_apply_nyul(nyul, cube[c]) for c in range(cube.shape[0])], axis=0)
        np.save(PROC / p.name, out.astype(np.float32))
        n += 1
        if n % 25 == 0:
            log(f"    Nyul-applied {n}/{len(work)}")
    log(f"processed cubes (Nyul-normalized) done: {n} -> {PROC}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="process only the first N patient IDs")
    ap.add_argument("--cube", type=int, default=CUBE)
    ap.add_argument("--nyul-only", action="store_true", help="skip extract/cache; just (re)fit+apply Nyul")
    args = ap.parse_args()

    df = load_labels()
    pids = sorted(df[COL_PID].astype(str).tolist(), key=lambda p: int(p.split("_")[-1]))
    log(f"fastMRI-NYU preprocess: {len(pids)} patients, cube={args.cube}, limit={args.limit}")

    if not args.nyul_only:
        process_all(pids, args.cube, args.limit)
    fit_and_apply_nyul(args.limit)
    log("PREPROCESS COMPLETE")


if __name__ == "__main__":
    sys.exit(main())
