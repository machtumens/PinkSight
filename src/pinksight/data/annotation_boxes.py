"""Lesion ROI boxes for Duke (P03 — the ROI-crop source; resolves D1 = boxes, not masks).

Duke ships ONE 3D bounding box per patient in ``Annotation_Boxes.xlsx`` (922 rows,
voxel coordinates), not segmentation masks. This is the uniform annotation source the
per-patient series naming is NOT — every patient has exactly one box here. The phase
stack (see :mod:`pinksight.data.phase_stack`) picks which series; this picks where in
the volume the lesion is.

Box coordinates follow the Duke/community convention — half-open Python ranges
``volume[Start_Row:End_Row, Start_Column:End_Column, Start_Slice:End_Slice]`` on a
(row, col, slice)-ordered array.

ponytail: crop assumes (row, col, slice) axis order (matches the column names). If the
MONAI pipeline emits a different order, transpose there rather than generalising here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_PATH = Path("data/raw/Annotation_Boxes.xlsx")
_COLS = ["Patient ID", "Start Row", "End Row", "Start Column", "End Column", "Start Slice", "End Slice"]


@dataclass(frozen=True)
class Box:
    patient: str
    row: tuple[int, int]  # (start, end), half-open
    col: tuple[int, int]
    slice: tuple[int, int]


def load_boxes(path: str | Path = DEFAULT_PATH) -> dict[str, Box]:
    """patient_id -> Box for every Duke patient in Annotation_Boxes.xlsx."""
    df = pd.read_excel(path)
    missing = set(_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    return {
        r["Patient ID"]: Box(
            r["Patient ID"],
            (int(r["Start Row"]), int(r["End Row"])),
            (int(r["Start Column"]), int(r["End Column"])),
            (int(r["Start Slice"]), int(r["End Slice"])),
        )
        for _, r in df.iterrows()
    }


def crop(volume: np.ndarray, box: Box, margin: int = 0) -> np.ndarray:
    """Crop a (row, col, slice)-ordered volume to the lesion box + isotropic voxel margin.

    margin is the ROI breathing room ("ROI+margin crop" in the architecture); it is
    clipped to the volume bounds so a lesion near an edge never indexes out of range.
    """
    (r0, r1), (c0, c1), (s0, s1) = box.row, box.col, box.slice
    nr, nc, ns = volume.shape[:3]
    return volume[
        max(r0 - margin, 0) : min(r1 + margin, nr),
        max(c0 - margin, 0) : min(c1 + margin, nc),
        max(s0 - margin, 0) : min(s1 + margin, ns),
    ]
