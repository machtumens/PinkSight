"""Deterministic seeding — the reproducibility floor (LAW L-3 / Rule 10).

ponytail: seeds stdlib + numpy now; torch only if the [ml] extra is installed (G1+).
"""

from __future__ import annotations

import os
import random

DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED) -> int:
    """Seed every RNG in use; return the seed so the run card can record it."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch  # ponytail: present only under the [ml] extra (G1+)

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    return seed
