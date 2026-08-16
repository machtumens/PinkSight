
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
    row: tuple[int, int]  
    col: tuple[int, int]
    slice: tuple[int, int]


def load_boxes(path: str | Path = DEFAULT_PATH) -> dict[str, Box]:
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
    (r0, r1), (c0, c1), (s0, s1) = box.row, box.col, box.slice
    nr, nc, ns = volume.shape[:3]
    return volume[
        max(r0 - margin, 0) : min(r1 + margin, nr),
        max(c0 - margin, 0) : min(c1 + margin, nc),
        max(s0 - margin, 0) : min(s1 + margin, ns),
    ]
