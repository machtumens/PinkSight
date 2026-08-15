"""G1 radiomics baseline (LOCK-4 / [2.1]): PyRadiomics features -> classical classifier.

The frozen comparison point every later model reports a delta against ([2.4]); the encoder's G2
SUCCESS is *defined* as beating it ([3.16]). Two halves:

  * feature extraction (GATED): PyRadiomics over each DCE phase inside the lesion ROI. pyradiomics
    is not installed and the cohort imaging is absent, so `extract_features` raises rather than
    fabricating features — wire it once the dep + imaging land.
  * classifier + evaluation (sklearn, runs now): LogReg/RF with **patient-level**
    StratifiedGroupKFold (LOCK-2 — a patient never spans train/test) -> AUROC mean + per-fold spread.

Scope now: radiomics->subtype only. radiomics->Ki-67 ([2.1]) is deferred — the G0 audit found usable
numeric Ki-67 N=0 (O-2), so there is no Ki-67 target to regress yet.

ponytail: DeLong CI + ECE + multi-seed (the eval-integrity reporting rule) attach to the real
headline number (Tier 2, when the cohort lands), not to this synthetic-testable harness.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from pinksight.metrics import delong_ci, ece  # pure-numpy; never-a-bare-number reporting (docs/CLAIM_LEDGER.md)

_CLASSIFIERS = {
    "logreg": lambda: LogisticRegression(max_iter=1000),
    "rf": lambda: RandomForestClassifier(n_estimators=300, random_state=0),
}
SEEDS = (0, 1, 2)  # eval-integrity minimum (docs/CLAIM_LEDGER.md "multi-seed spread 3 min / 5 target")
N_SPLITS = 5


class FeaturesNotProvisioned(RuntimeError):
    """Radiomics extraction requested but pyradiomics/imaging is not provisioned (G1 blocker)."""


def make_classifier(kind: str = "logreg") -> ClassifierMixin:
    """Build a frozen-config classical classifier for the baseline ([2.1] LogReg/RF)."""
    if kind not in _CLASSIFIERS:
        raise ValueError(f"kind must be one of {sorted(_CLASSIFIERS)}, got {kind}")
    return _CLASSIFIERS[kind]()


def cross_val_auroc(
    X: np.ndarray,
    y: Sequence[int],
    groups: Sequence,
    kind: str = "logreg",
    n_splits: int = 5,
) -> tuple[float, list[float]]:
    """Patient-grouped, class-stratified CV AUROC for radiomics->subtype (LOCK-2 + LOCK-4).

    `groups` is the patient id per row, so no patient spans train and test (patient-level splits
    only). Returns (mean_auroc, per_fold_aurocs).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    groups = np.asarray(groups)
    if not len(X) == len(y) == len(groups):
        raise ValueError("X, y, groups must have equal length")
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0)
    aurocs: list[float] = []
    for tr, te in cv.split(X, y, groups):
        if not set(groups[tr]).isdisjoint(groups[te]):
            raise AssertionError("patient leaked across a CV fold — LOCK-2 violation")
        clf = make_classifier(kind).fit(X[tr], y[tr])
        prob = clf.predict_proba(X[te])[:, 1]
        aurocs.append(float(roc_auc_score(y[te], prob)))
    return float(np.mean(aurocs)), aurocs


def _pipeline(kind: str):
    """Classifier with TRAIN-only median-impute + standardize (LOCK-2 — fit inside each fold)."""
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), make_classifier(kind))


def cross_val_metrics(
    X: np.ndarray,
    y: Sequence[int],
    groups: Sequence,
    kind: str = "logreg",
    n_splits: int = N_SPLITS,
    seeds: tuple[int, ...] = SEEDS,
) -> dict:
    """Multi-seed patient-grouped CV -> never-a-bare-number metrics dict (mirrors the P05 schema).

    Per seed: 5-fold StratifiedGroupKFold (patient-disjoint, LOCK-2), pooled out-of-fold preds ->
    AUROC + DeLong 95% CI + ECE. Reports mean/std across seeds. Impute+standardize fit train-only.
    """
    X = np.asarray(X, dtype=float)
    yv = np.asarray(list(y))
    groups = np.asarray(groups)
    per_seed, pooled, ci, ece_seed = {}, {}, {}, {}
    for seed in seeds:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fold_aucs: list[float] = []
        oof = np.full(len(yv), np.nan)
        for tr, te in cv.split(X, yv, groups):
            if not set(groups[tr]).isdisjoint(groups[te]):
                raise AssertionError("patient leaked across a CV fold — LOCK-2 violation")
            clf = _pipeline(kind).fit(X[tr], yv[tr])
            oof[te] = clf.predict_proba(X[te])[:, 1]
            fold_aucs.append(float(roc_auc_score(yv[te], oof[te])))
        if np.isnan(oof).any():
            raise RuntimeError("OOF preds incomplete — a patient-row was never in a test fold")
        per_seed[seed] = float(np.mean(fold_aucs))
        auc, lo, hi = delong_ci(yv, oof)
        pooled[seed], ci[seed], ece_seed[seed] = auc, (lo, hi), ece(yv, oof)
    means = np.array(list(per_seed.values()))
    pooled_v = np.array(list(pooled.values()))
    return {
        "classifier": kind,
        "auroc_mean": float(means.mean()),
        "auroc_std_across_seeds": float(means.std()),
        "auroc_min": float(means.min()),
        "auroc_max": float(means.max()),
        "per_seed_mean_auroc": {str(k): round(v, 4) for k, v in per_seed.items()},
        "auroc_pooled_oof_mean": float(pooled_v.mean()),
        "auroc_pooled_oof_per_seed": {str(k): round(v, 4) for k, v in pooled.items()},
        "delong_ci95_per_seed": {str(k): [round(lo, 4), round(hi, 4)] for k, (lo, hi) in ci.items()},
        "delong_ci95_mean": [
            round(float(np.mean([lo for lo, _ in ci.values()])), 4),
            round(float(np.mean([hi for _, hi in ci.values()])), 4),
        ],
        "ece_mean": round(float(np.mean(list(ece_seed.values()))), 4),
        "ece_per_seed": {str(k): round(v, 4) for k, v in ece_seed.items()},
        "ece_n_bins": 10,
        "n_dev": int(len(yv)),
        "tnbc_prevalence": round(float(np.mean(yv)), 4),
        "n_splits": n_splits,
        "seeds": list(seeds),
    }


def extract_features(
    volume: np.ndarray,
    mask: np.ndarray,
    phase_names: Sequence[str] | None = None,
    settings: dict | None = None,
) -> dict[str, float]:
    """PyRadiomics features per DCE phase inside the lesion ROI -> flat {phase_feature: value} dict.

    GATED: pyradiomics is not installed and the imaging cohort is absent. Raises rather than
    returning fabricated features. Real extraction runs once both land (Tier 2).
    """
    try:
        from radiomics import featureextractor
    except ImportError as e:  # pragma: no cover - exercised only without the dep
        raise FeaturesNotProvisioned(
            "pyradiomics not installed and Duke imaging absent. Add pyradiomics to deps and provide "
            "the cohort before extracting baseline features — do not fabricate."
        ) from e

    import SimpleITK as sitk

    vol = np.asarray(volume)
    label = sitk.GetImageFromArray(np.asarray(mask).astype(np.uint8))
    extractor = featureextractor.RadiomicsFeatureExtractor(**(settings or {}))
    names = (
        list(phase_names) if phase_names is not None else [f"phase{i}" for i in range(vol.shape[0])]
    )
    feats: dict[str, float] = {}
    for name, channel in zip(names, vol, strict=True):
        img = sitk.GetImageFromArray(channel.astype(np.float32))
        for key, value in extractor.execute(img, label).items():
            if not key.startswith("diagnostics_"):  # drop provenance entries
                feats[f"{name}_{key}"] = float(value)
    return feats
