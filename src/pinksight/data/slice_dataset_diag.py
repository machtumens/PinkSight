
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

SLICE_HW = 64  
TRAIN_SLICES_PER_PATIENT = 8
POST_CONTRAST_CHANNELS = (1, 2, 3)  

FULLN_DIR = Path("data/processed_fulln")


def _minmax_4phase(sl4: np.ndarray) -> np.ndarray:
    lo = float(sl4.min())
    hi = float(sl4.max())
    scaled = (sl4 - lo) / (hi - lo + 1e-6)
    return np.ascontiguousarray(scaled[list(POST_CONTRAST_CHANNELS)], dtype=np.float32)


def _center_crop_2d(x: np.ndarray, cr: int, cc: int, hw: int = SLICE_HW) -> np.ndarray:
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
    half = TRAIN_SLICES_PER_PATIENT // 2
    lo = max(0, center_slice - half)
    hi = min(n_slices, lo + TRAIN_SLICES_PER_PATIENT)
    lo = max(0, hi - TRAIN_SLICES_PER_PATIENT)
    train_idx = list(range(lo, hi))
    test_idx = min(center_slice + 1, n_slices - 1)  
    return train_idx, test_idx


class DiagSliceGradeDataset(Dataset):

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
        self.samples: list[tuple[str, int, int, str]] = []
        self._cache: dict[str, dict] = {}  
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
        x = self._load(pid)["tiles"][row]  
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
