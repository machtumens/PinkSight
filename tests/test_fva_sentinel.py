"""C2-12 unit gate: the ported shuffle-sentinel returns AUROC ≈ 0.5 on shuffled labels.

``fva.shuffle_sentinel`` holds the LOCK-2 integrity machinery extracted byte-identical from h6
(C2-3). This test proves the sentinel MECHANISM in isolation, on synthetic data with a KNOWN signal:

  - real labels  -> a separable stream must score well ABOVE chance (the pipeline can learn),
  - shuffled     -> the SAME pipeline must collapse to ≈ 0.5 (permuting labels destroys signal).

That contrast is exactly what the sentinel asserts beside every real number in H4/H6. This does NOT
prove the sentinel catches every real leakage path — only that a shuffled label decodes at chance,
which is the pre-reg integrity requirement. Fast, $0, no Duke data — always runs.
"""

from __future__ import annotations

import pytest

pytest.importorskip("sklearn", reason="sklearn is an ml/arms/trackb-extra dep — skip cleanly under a base install; fva.shuffle_sentinel imports sklearn")

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from fva.shuffle_sentinel import coalition_oof, stream_shuffle_sentinels


def _make_separable(n: int = 300, d: int = 6, seed: int = 0):
    """A patient-disjoint, class-separable synthetic stream (one patient per row)."""
    rng = np.random.default_rng(seed)
    y = (np.arange(n) % 2).astype(int)
    X = rng.normal(0, 1, (n, d))
    X[y == 1] += 1.2  # genuine class shift -> learnable signal
    groups = np.arange(n)  # each row its own patient (disjoint folds trivially satisfiable)
    return X, y, groups


def test_shuffled_labels_decode_at_chance():
    """coalition_oof(shuffle=True) on separable data must land AUROC ≈ 0.5 ± 0.05."""
    X, y, groups = _make_separable()

    # Sanity: the REAL pipeline learns the signal (guards against a dead pipeline giving fake 0.5).
    real_oof = coalition_oof([X], [False], y, groups, seed=0, n_splits=5, shuffle=False)
    real_auc = roc_auc_score(y, real_oof)
    assert real_auc > 0.75, f"real signal should be learnable, got AUROC={real_auc:.3f}"

    # The sentinel: shuffled labels must collapse to chance.
    shuf_oof = coalition_oof([X], [False], y, groups, seed=0, n_splits=5, shuffle=True)
    shuf_auc = roc_auc_score(y, shuf_oof)
    assert shuf_auc == pytest.approx(0.5, abs=0.05), f"shuffled AUROC not at chance: {shuf_auc:.3f}"
    assert shuf_auc < real_auc - 0.03, "shuffle must sit well below the real signal"


def test_stream_shuffle_sentinels_report_at_chance_per_stream():
    """The per-stream sentinel wrapper must flag shuffle_at_chance=True for every stream."""
    X, y, groups = _make_separable(seed=1)
    streams = {"clinical": {"X": X, "impute": False, "meta": {}}}

    # The wrapper reads the REAL number from coalition_detail (same pooled-OOF quantity).
    real_oof = coalition_oof([X], [False], y, groups, seed=0, n_splits=5, shuffle=False)
    coalition_detail = {"clinical": {"auroc_mean": round(float(roc_auc_score(y, real_oof)), 4)}}

    out = stream_shuffle_sentinels(
        streams, y, groups, coalition_detail,
        seeds=(0, 1, 2), n_splits=5, stream_names=("clinical",),
    )

    sent = out["clinical"]
    assert sent["shuffle_at_chance"] is True, f"shuffle not at chance: {sent['shuffle_auroc_mean']}"
    assert 0.45 <= sent["shuffle_auroc_mean"] <= 0.55
    assert sent["real_auroc_mean"] > 0.75  # real signal preserved
    assert sent["shuffle_passes"] is True   # shuffle < real − 0.03
