"""BP4 selective-prediction self-checks: the curve must SEPARATE a heterogeneous null from a uniform one."""

from __future__ import annotations

import pytest

pytest.importorskip("scipy", reason="scipy is an ml/arms/trackb-extra dep — skip cleanly under a base install; pinksight.eval.selective imports scipy")

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pinksight.eval.selective import (
    categorical_assoc,
    confident_mask,
    coverage_auroc_curve,
    point_biserial,
)


def test_confident_mask_selects_fraction():
    p = np.linspace(0.0, 1.0, 100)  # confidence |p-0.5| spans 0..0.5
    assert confident_mask(p, 0.5).sum() == 50
    # the selected half must be the extremes (most confident), not the middle
    assert confident_mask(p, 0.2)[[0, -1]].all()


def _heterogeneous():
    # 100 confident-correct patients (p perfectly separates y) + 100 uninformative (p==0.5, ties).
    y = np.array([0, 1] * 50 + [0, 1] * 50)
    p = np.array([0.02, 0.98] * 50 + [0.5] * 100)
    return y, p


def test_curve_rises_for_heterogeneous_null():
    y, p = _heterogeneous()
    curve = {r["coverage"]: r["auroc"] for r in coverage_auroc_curve(y, p)}
    assert curve[0.4] > curve[1.0] + 0.1        # tight coverage isolates the reliable subpopulation
    assert curve[0.4] == pytest.approx(1.0, abs=1e-9)


def test_curve_flat_for_uniform_null():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = 0.5 + rng.normal(0, 0.01, 200)          # confidence uncorrelated with y
    curve = {r["coverage"]: r["auroc"] for r in coverage_auroc_curve(y, p)}
    assert abs(curve[0.4] - 0.5) < 0.15 and abs(curve[1.0] - 0.5) < 0.15  # ~chance everywhere


def test_selection_drivers():
    mask = np.array([True] * 50 + [False] * 50)
    x = np.array([1.0] * 50 + [0.0] * 50)       # covariate perfectly tracks selection
    assert point_biserial(mask, x) == pytest.approx(1.0, abs=1e-9)
    cats = np.array(["A"] * 50 + ["B"] * 50, dtype=object)
    assert categorical_assoc(mask, cats) < 0.05  # scanner perfectly predicts selection


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
