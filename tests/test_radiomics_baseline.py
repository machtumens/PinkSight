
import pytest

pytest.importorskip("sklearn", reason="sklearn is an ml/arms/trackb-extra dep — skip cleanly under a base install; pinksight.baseline.radiomics_baseline imports sklearn")

import numpy as np

from pinksight.baseline.radiomics_baseline import (
    FeaturesNotProvisioned,
    cross_val_auroc,
    extract_features,
    make_classifier,
)


def _separable(n_per_class=15, n_feats=4, seed=0):
    rng = np.random.default_rng(seed)
    x0 = rng.normal(0.0, 1.0, (n_per_class, n_feats))
    x1 = rng.normal(3.0, 1.0, (n_per_class, n_feats))
    X = np.vstack([x0, x1])
    y = np.array([0] * n_per_class + [1] * n_per_class)
    groups = np.arange(len(y))  
    return X, y, groups


def test_cross_val_auroc_is_a_probability_and_separates():
    X, y, groups = _separable()
    mean, folds = cross_val_auroc(X, y, groups, n_splits=5)
    assert len(folds) == 5
    assert all(0.0 <= a <= 1.0 for a in folds)
    assert mean > 0.9  


def test_cross_val_auroc_runs_with_repeated_patients():
    X, y, base = _separable(n_per_class=15)
    groups = np.concatenate([base[:15] // 2, base[15:] // 2 + 100])  
    mean, folds = cross_val_auroc(X, y, groups, n_splits=3)
    assert 0.0 <= mean <= 1.0


def test_make_classifier_kinds():
    assert make_classifier("logreg") is not None
    assert make_classifier("rf") is not None
    with pytest.raises(ValueError):
        make_classifier("xgboost")  


def test_cross_val_auroc_rejects_length_mismatch():
    with pytest.raises(ValueError):
        cross_val_auroc(np.zeros((4, 3)), [0, 1, 0], [0, 1, 2, 3])


def test_extract_features_runs_when_provisioned_else_gates():
    try:
        import radiomics  
    except ImportError:
        with pytest.raises(FeaturesNotProvisioned):
            extract_features(np.zeros((1, 8, 8, 8)), np.ones((8, 8, 8)))
        return
    rng = np.random.default_rng(0)
    vol = rng.normal(100.0, 20.0, (1, 12, 12, 12)).astype(np.float32)
    mask = np.zeros((12, 12, 12), np.uint8)
    mask[3:9, 3:9, 3:9] = 1
    feats = extract_features(vol, mask, phase_names=["post1"], settings={"binCount": 32})
    assert feats and all(isinstance(v, float) for v in feats.values())
    assert all(k.startswith("post1_") for k in feats)  
