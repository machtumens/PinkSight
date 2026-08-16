
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_SERIES_DIR = re.compile(r"^(\d+)-(.+)-(\d+)$")
_DYN = re.compile(r"dyn|vibrant|multiphase", re.IGNORECASE)
_POST = re.compile(r"(\d+)\s*(?:st|nd|rd|th)\s*pass?|ph\s*(\d+)", re.IGNORECASE)
_EXCLUDE = re.compile(
    r"sub(tract)?|mip|\bseg|overlay|report|screen|secondary|key.?image|\bt2\b|adc|dwi|flair",
    re.IGNORECASE,
)


def _pass_num(description: str) -> int:
    m = _POST.search(description)
    return int(m.group(1) or m.group(2)) if m else 0


@dataclass(frozen=True)
class Series:
    number: int
    description: str
    path: Path


@dataclass(frozen=True)
class PhaseStack:
    patient: str
    pre: Series
    posts: tuple[Series, ...]  
    other: tuple[Series, ...] = ()  

    @property
    def stack(self) -> tuple[Series, ...]:
        return (self.pre, *self.posts)


def _leaf_series_dirs(patient_dir: Path) -> list[Path]:
    return [
        d
        for d in patient_dir.rglob("*")
        if d.is_dir() and any(f.suffix.lower() == ".dcm" for f in d.iterdir() if f.is_file())
    ]


def _scan_series_from_headers(dirs: list[Path]) -> list[Series]:
    import SimpleITK as sitk

    seen: dict[str, Series] = {}
    for d in dirs:
        files = sorted(f for f in d.iterdir() if f.suffix.lower() == ".dcm")
        r = sitk.ImageFileReader()
        r.SetFileName(str(files[0]))
        try:
            r.ReadImageInformation()
            uid = r.GetMetaData("0020|000e").strip()  
            desc = r.GetMetaData("0008|103e").strip()  
        except Exception:
            continue
        try:
            number = int(r.GetMetaData("0020|0011").strip())  
        except Exception:
            number = 0
        seen.setdefault(uid, Series(number, desc, d))  
    return list(seen.values())


def _scan_series(patient_dir: Path) -> list[Series]:
    leaves = _leaf_series_dirs(patient_dir)
    descriptive = [(d, _SERIES_DIR.match(d.name)) for d in leaves]
    descriptive = [(d, m) for d, m in descriptive if m]
    if descriptive:
        return [Series(int(m.group(1)), m.group(2).strip(), d) for d, m in descriptive]
    return _scan_series_from_headers(leaves)


def select_phase_stack(patient_dir: str | Path) -> PhaseStack:
    patient_dir = Path(patient_dir)
    pre: list[Series] = []
    posts: list[Series] = []
    other: list[Series] = []

    for s in _scan_series(patient_dir):
        if _EXCLUDE.search(s.description):
            continue
        if not _DYN.search(s.description):
            other.append(s)
        elif _POST.search(s.description):
            posts.append(s)
        else:
            pre.append(s)

    if len(pre) != 1 or not posts:
        raise ValueError(
            f"{patient_dir.name}: expected exactly 1 pre-contrast + >=1 post-contrast series, "
            f"found pre={[s.description for s in pre]} posts={[s.description for s in posts]}"
        )
    posts.sort(key=lambda s: _pass_num(s.description))
    return PhaseStack(patient_dir.name, pre[0], tuple(posts), tuple(other))
