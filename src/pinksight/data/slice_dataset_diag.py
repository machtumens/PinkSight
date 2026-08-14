"""[HEAD2-GRADE-2D-SLICE DIAGNOSTIC] Full-iso 2D-per-slice dataset with a switchable crop SOURCE.

Same DeepRadGrade 2D recipe as `slice_dataset.SliceGradeDataset` (3x64x64 = first 3 post-contrast
phases as channels, min-max over the 4-phase pre+3post range, up-to-8 train slices around the tumour
centre, single supra-central OOF slice per patient, per-set 4h/4v/5none flips, `pid` on every sample
for the slice-level patient-disjoint guard). The ONE axis this module varies is the CROP SOURCE:

  * source="gt"   — ARM GT (ORACLE): centre + 64x64 window from the Duke radiologist box.
                    LOCK-2 waived for this diagnostic arm only (pre-reg DIAGNOSTIC ADDENDUM 2026-07-13).
  * source="pred" — ARM PRED (honest): centre + 64x64 window from the H0-PREDICTED mask. LOCK-2-clean.

The tiles are BAKED by `scripts/head2_grade_fulln_cache.py` (64x64 windows around the box centroid,
first-3-post, min-max over the 4-phase range) from the SAME full-iso volume for BOTH arms, so any
GT-vs-PRED gap is attributable to crop quality (box source), not grid/normalisation/phase differences.
This dataset only INDEXES those tiles and applies the train-time flip augmentation.

The shared crop/normalise/slice-plan helpers (`_center_crop_2d`, `_minmax_4phase`, `_slice_plan`,
constants) live here so the cache script and the dataset stay single-sourced.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

SLICE_HW = 64  # DeepRadGrade in-plane 64x64
TRAIN_SLICES_PER_PATIENT = 8
POST_CONTRAST_CHANNELS = (1, 2, 3)  # first 3 post-contrast phases (vol[0] is pre-contrast)

FULLN_DIR = Path("data/processed_fulln")


def _minmax_4phase(sl4: np.ndarray) -> np.ndarray:
    """Min-max a (4,H,W) pre+3post stack over the whole 4-phase range, keep the 3 post channels.
    One shared (min,max) across the 4 phases preserves inter-phase enhancement contrast. Identical to
    slice_dataset._minmax_4phase (same normalisation for both datasets => comparable)."""
    lo = float(sl4.min())
    hi = float(sl4.max())
    scaled = (sl4 - lo) / (hi - lo + 1e-6)
    return np.ascontiguousarray(scaled[list(POST_CONTRAST_CHANNELS)], dtype=np.float32)


def _center_crop_2d(x: np.ndarray, cr: int, cc: int, hw: int = SLICE_HW) -> np.ndarray:
    """Extract an hw x hw window from a (3, H, W) slice centred on (cr, cc), zero-padded at edges so a
    lesion near the border always yields a full hw x hw tile (no resize -> honest pixel-for-pixel crop)."""
    _, h, w = x.shape
    half = hw // 2
    r0, c0 = cr - half, cc - half
    out = np.zeros((x.shape[0], hw, hw), dtype=np.float32)
    sr0, sc0 = max(0, r0), max(0, c0)
    sr1, sc1 = min(h, r0 + hw), min(w, c0 + hw)
    dr0, dc0 = sr0 - r0, sc0 - c0
    out[:, dr0 : dr0 + (sr1 - sr0), dc0 : dc0 + (sc1 - sc0)] = x[:, sr0:sr1, sc0:sc1]
    return out


def _slice_plan(center_slice: int, n_slices: int) -> tuple[list[int], int]:
    """(train_slice_indices, test_slice_index): up to 8 slices around the centre; OOF = supra-central."""
    half = TRAIN_SLICES_PER_PATIENT // 2
    lo = max(0, center_slice - half)
    hi = min(n_slices, lo + TRAIN_SLICES_PER_PATIENT)
    lo = max(0, hi - TRAIN_SLICES_PER_PATIENT)
    train_idx = list(range(lo, hi))
    test_idx = min(center_slice + 1, n_slices - 1)  # immediately superior; clamp at the top slice
    return train_idx, test_idx


class DiagSliceGradeDataset(Dataset):
    """(pid,label) list -> per-slice samples indexed from the baked per-arm tile cache.

    __getitem__ yields (x:(3,64,64), y, pid). `split="train"` emits up-to-8 slices/patient (per-set
    4h/4v/5none flips when augment=True); `split="test"` emits exactly ONE (supra-central) slice per
    patient -> one OOF prediction per patient. `pid` rides on every sample for the slice-level
    patient-disjoint assertion.
    """

    def __init__(
        self,
        items: list[tuple[str, int]],
        source: str,
        proc_dir: Path = FULLN_DIR,
        split: str = "train",
        augment: bool = False,
    ) -> None:
        if source not in ("gt", "pred"):
            raise ValueError(f"source must be 'gt' or 'pred', got {source!r}")
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")
        self.source = source
        self.proc_dir = Path(proc_dir)
        self.split = split
        self.augment = augment and split == "train"
        # flat sample table: (pid, label, tile_row_index, flip_op) — tile_row_index indexes the cached
        # (K,3,64,64) tiles array for that patient/arm.
        self.samples: list[tuple[str, int, int, str]] = []
        self._cache: dict[str, dict] = {}  # pid -> loaded npz dict (small; kept in RAM)
        for pid, label in items:
            z = self._load(pid)
            slice_ids = z["slice_ids"].tolist()
            id_to_row = {int(s): i for i, s in enumerate(slice_ids)}
            if split == "test":
                self.samples.append((pid, label, id_to_row[int(z["test_id"])], "none"))
            else:
                train_rows = [id_to_row[int(s)] for s in z["train_ids"].tolist()]
                self.samples.extend(self._aug_train(pid, label, train_rows))

    def _load(self, pid: str) -> dict:
        if pid not in self._cache:
            with np.load(self.proc_dir / f"{pid}_{self.source}.npz") as z:
                self._cache[pid] = {k: z[k] for k in z.files}
        return self._cache[pid]

    def _aug_train(self, pid: str, label: int, train_rows: list[int]) -> list[tuple[str, int, int, str]]:
        """Per DeepRadGrade: per patient slice-set, 4 h-flip + 4 v-flip + 5 unchanged (deterministic)."""
        if not train_rows:
            return []
        if not self.augment:
            return [(pid, label, r, "none") for r in train_rows]
        plan = ["h"] * 4 + ["v"] * 4 + ["none"] * 5
        out: list[tuple[str, int, int, str]] = []
        for k, op in enumerate(plan):
            r = train_rows[k % len(train_rows)]
            out.append((pid, label, r, op))
        return out

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        pid, label, row, flip = self.samples[i]
        x = self._load(pid)["tiles"][row]  # (3, 64, 64) already normalised
        if flip == "h":
            x = np.ascontiguousarray(x[:, :, ::-1])
        elif flip == "v":
            x = np.ascontiguousarray(x[:, ::-1, :])
        else:
            x = np.ascontiguousarray(x)
        return (
            torch.as_tensor(x, dtype=torch.float32),
            torch.tensor(float(label), dtype=torch.float32),
            pid,
        )
