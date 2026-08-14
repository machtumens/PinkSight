"""set_seed must be deterministic — same seed, same draws (reproducibility floor)."""

import random

from pinksight.seed import set_seed


def test_set_seed_is_deterministic():
    set_seed(123)
    a = [random.random() for _ in range(3)]
    set_seed(123)
    b = [random.random() for _ in range(3)]
    assert a == b
