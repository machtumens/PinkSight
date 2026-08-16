
from __future__ import annotations

import numpy as np

from fva.shuffle_sentinel import coalition_oof


def permutation_null_oof(
    mats: list[np.ndarray],
    needs_impute: list[bool],
    y: np.ndarray,
    groups: np.ndarray,
    k: int = 8,
    base_seed: int = 1000,
) -> np.ndarray:
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    oofs = [
        coalition_oof(mats, needs_impute, y, groups, seed=base_seed + s, shuffle=True)
        for s in range(k)
    ]
    return np.mean(oofs, axis=0)
