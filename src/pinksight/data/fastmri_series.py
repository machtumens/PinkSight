
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import SimpleITK as sitk

_SLICE_FRAME = re.compile(r"slice_(\d+)_frame_(\d+)\.dcm$")
_SCAN_DIR = re.compile(r"fastMRI_breast_(\d+)_(\d+)_DCM$")


def pid_to_num(pid: str) -> int:
    m = re.search(r"(\d+)$", pid)
    if not m:
        raise ValueError(f"cannot parse a numeric index from patient id {pid!r}")
    return int(m.group(1))


def find_scan_dir(extract_root: str | Path, pid: str) -> Path:
    num = pid_to_num(pid)
    cands: list[tuple[int, Path]] = []
    for d in Path(extract_root).rglob(f"fastMRI_breast_{num:03d}_*_DCM"):
        if not d.is_dir():
            continue
        m = _SCAN_DIR.search(d.name)
        if m and int(m.group(1)) == num:
            cands.append((int(m.group(2)), d))
    if not cands:
        raise FileNotFoundError(f"no scan dir for {pid} (num {num}) under {extract_root}")
    return min(cands, key=lambda t: t[0])[1]  


def frame_files(scan_dir: str | Path) -> dict[int, list[Path]]:
    frames: dict[int, list[tuple[int, Path]]] = {}
    for f in Path(scan_dir).glob("*.dcm"):
        m = _SLICE_FRAME.search(f.name)
        if not m:
            continue
        sl, fr = int(m.group(1)), int(m.group(2))
        frames.setdefault(fr, []).append((sl, f))
    return {fr: [p for _, p in sorted(sls)] for fr, sls in sorted(frames.items())}


def load_frame_volume(files: list[Path]) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames([str(p) for p in files])
    return sitk.Cast(reader.Execute(), sitk.sitkFloat32)


def load_dce_phases(scan_dir: str | Path, max_phases: int | None = None) -> list[sitk.Image]:
    ff = frame_files(scan_dir)
    if not ff:
        raise ValueError(f"{scan_dir}: no slice_*_frame_*.dcm files found")
    order = sorted(ff)
    if max_phases is not None:
        order = order[:max_phases]
    return [load_frame_volume(ff[fr]) for fr in order]


def to_rcs(img: sitk.Image) -> np.ndarray:
    return np.transpose(sitk.GetArrayFromImage(img), (1, 2, 0))
