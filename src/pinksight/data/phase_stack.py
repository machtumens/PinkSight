"""Phase-stack selector for Duke DCE-MRI series (P03, feeds the MRI encoder).

Duke ships each patient as several DICOM series whose SeriesDescription is *not*
uniform across patients ("ax dyn pre" vs "ax 3d dyn pre"; "ax t1 tse" vs
"ax 3d t1 bilateral"). TCIA's Data Retriever encodes that description in the
folder name as ``{SeriesNumber}-{SeriesDescription}-{seriesUID}``, so we classify
on the folder name and never hardcode a single expected name.

The dynamic phase stack the encoder wants is: pre-contrast first, then the
post-contrast passes in temporal order. Everything else (structural T1, +c) is
returned in ``other`` but kept out of the stack.

Two on-disk layouts are supported, detected per patient:

  * **Descriptive** (TCIA "descriptive directory" downloads + our test fixtures):
    folders are ``{seriesNumber}-{description}-{seriesUID}`` — classify on the folder
    name, no header read, and the duplicate numeric tree is skipped for free.
  * **Numeric** (the full 345 GB NBIA pull): every folder is a bare number, so the
    description lives ONLY in the DICOM header. We read SeriesDescription /
    SeriesInstanceUID / SeriesNumber from one slice per leaf dir and dedupe by UID.

ponytail: layout is auto-detected (any descriptive dir present -> folder-name path),
so the synthetic test fixtures and the real numeric pull both work through one entry
point. The classifier covers the Duke SeriesDescription zoo — "ax dyn pre" / "ax 3d
dyn" baselines and "1st pass" / "Ph1" / GE "Vibrant MultiPhase" post passes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# folder = "{seriesNumber}-{description}-{seriesUID}", e.g. "7-ax dyn 1st pass-45961"
_SERIES_DIR = re.compile(r"^(\d+)-(.+)-(\d+)$")
# dynamic DCE family (the only series allowed in the stack): GE "Vibrant"/"MultiPhase" too
_DYN = re.compile(r"dyn|vibrant|multiphase", re.IGNORECASE)
# post-contrast pass marker: "...1st pass...", "...4th pass...", "Ph1/...", "Ph 2 ..."
# ("pass?" tolerates the truncated "1st pas" typo seen in a few Duke series descriptions)
_POST = re.compile(r"(\d+)\s*(?:st|nd|rd|th)\s*pass?|ph\s*(\d+)", re.IGNORECASE)
# derived / structural series that must never enter the dynamic stack
_EXCLUDE = re.compile(
    r"sub(tract)?|mip|\bseg|overlay|report|screen|secondary|key.?image|\bt2\b|adc|dwi|flair",
    re.IGNORECASE,
)


def _pass_num(description: str) -> int:
    """Pass/phase number from a post description ('1st pass'->1, 'Ph3/...'->3)."""
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
    posts: tuple[Series, ...]  # ordered by pass number
    other: tuple[Series, ...] = ()  # structural T1 etc. — kept, not in the dynamic stack

    @property
    def stack(self) -> tuple[Series, ...]:
        """Pre-contrast first, then post-contrast passes in temporal order."""
        return (self.pre, *self.posts)


def _leaf_series_dirs(patient_dir: Path) -> list[Path]:
    """Leaf dirs that directly contain .dcm files (one per acquired series)."""
    return [
        d
        for d in patient_dir.rglob("*")
        if d.is_dir() and any(f.suffix.lower() == ".dcm" for f in d.iterdir() if f.is_file())
    ]


def _scan_series_from_headers(dirs: list[Path]) -> list[Series]:
    """Read SeriesDescription/Number/UID from one slice per dir; dedupe by SeriesInstanceUID.

    For the numeric full-pull layout, where folder names carry no description. Dirs whose
    first DICOM cannot be read (empty/invalid) are skipped, not fatal.
    """
    import SimpleITK as sitk

    seen: dict[str, Series] = {}
    for d in dirs:
        files = sorted(f for f in d.iterdir() if f.suffix.lower() == ".dcm")
        r = sitk.ImageFileReader()
        r.SetFileName(str(files[0]))
        try:
            r.ReadImageInformation()
            uid = r.GetMetaData("0020|000e").strip()  # SeriesInstanceUID
            desc = r.GetMetaData("0008|103e").strip()  # SeriesDescription
        except Exception:
            continue
        try:
            number = int(r.GetMetaData("0020|0011").strip())  # SeriesNumber
        except Exception:
            number = 0
        seen.setdefault(uid, Series(number, desc, d))  # first dir wins per UID (drops dups)
    return list(seen.values())


def _scan_series(patient_dir: Path) -> list[Series]:
    """Every real DICOM series under a patient dir.

    Layout auto-detect: if any descriptive ``{num}-{desc}-{uid}`` dir is present, classify
    on folder names (cheap, no header read, duplicate numeric tree ignored for free).
    Otherwise the pull is numeric — recover descriptions from DICOM headers.
    """
    leaves = _leaf_series_dirs(patient_dir)
    descriptive = [(d, _SERIES_DIR.match(d.name)) for d in leaves]
    descriptive = [(d, m) for d, m in descriptive if m]
    if descriptive:
        return [Series(int(m.group(1)), m.group(2).strip(), d) for d, m in descriptive]
    return _scan_series_from_headers(leaves)


def select_phase_stack(patient_dir: str | Path) -> PhaseStack:
    """Pick the DCE phase stack (pre + ordered post passes) for one Duke patient.

    A series enters the dynamic stack only if its description is in the DCE family
    (`dyn`/`vibrant`/`multiphase`) and not a derived/structural series. A post-contrast
    pass carries a pass/phase marker ("1st pass", "Ph2"); the unmarked dynamic baseline
    is the pre-contrast phase.

    Raises ValueError unless exactly one pre-contrast and at least one post-contrast
    series are found — a broken assumption needs a human look, never a silent wrong stack.
    """
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
