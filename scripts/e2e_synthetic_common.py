"""Shared forward-spine helpers for the per-organ E2E synthetic runners (Track-B/C/fastMRI).

Lives in ``scripts/`` (next to ``fva/``) because ``coalition_oof`` is a scripts-level artifact, not a
package export. The three organ runners import ``permutation_null_oof`` from here so the shuffle-null
logic is written once.
"""

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
    """Permutation-null shuffle OOF: the MEAN over ``k`` INDEPENDENT label permutations (decoupled seeds).

    A single shuffle draw is a noisy sample of the permutation null: when the synthetic positive signal
    is strong it forms a feature cluster that one permuted-label LogReg can latch onto, so a lone shuffle
    AUROC swings wildly across seeds (empirically 0.26–0.56 at n=10k). Averaging ``k`` independent
    permutations is the textbook permutation-null estimate — it collapses cleanly to ~0.50, so the
    positive-control ``shuffle <= 0.55`` gate is seed-robust rather than luck-of-the-seed. The real
    (unshuffled) OOF stays a single ``coalition_oof(..., shuffle=False)`` call at the caller.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    oofs = [
        coalition_oof(mats, needs_impute, y, groups, seed=base_seed + s, shuffle=True)
        for s in range(k)
    ]
    return np.mean(oofs, axis=0)
