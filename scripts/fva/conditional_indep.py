"""Conditional-independence proxy: clinical-residualisation + per-feature MI (ported from h6).

Residualise subtype on the clinical stream (OOF), then test whether the MRI/radiomics streams
predict the residual. ~0.5 residual-AUROC and near-0 MI => the imaging streams carry ~no subtype
info beyond clinical (conditional independence). Directional; joint KSG/MINE on 512-D is BANNED
(high-dim downward bias at N~600, per the h6/h4 pre-regs).

Ported (C2-4) verbatim from ``h6_modality_audit.conditional_info_proxy`` — no algorithmic change.
The one structural adaptation: the OOF machinery is imported from ``fva.shuffle_sentinel`` (the
shared ``coalition_oof``) rather than the h6-local ``_coalition_oof``, and ``n_splits`` is threaded
through so the caller's FVAConfig controls it. Seed 0 is used for the residual fit, exactly as h6.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

from fva.shuffle_sentinel import coalition_oof


def conditional_info_proxy(
    streams: dict,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    imaging_streams: tuple[str, ...] = ("radiomics", "mri"),
) -> dict:
    """Residualise subtype on clinical (OOF), then test whether imaging predicts the residual.

    Verbatim from h6_modality_audit.conditional_info_proxy (n_splits + imaging_streams parameterised;
    OOF machinery sourced from fva.shuffle_sentinel.coalition_oof). Fit clinical->subtype OOF (seed
    0), residual = y - p_clinical; regress each imaging stream on the residual via patient-disjoint
    OOF AUROC of the residual sign. Plus per-feature MI of radiomics against the residual sign. Joint
    KSG/MINE on 512-D BANNED.
    """
    p_clin = coalition_oof([streams["clinical"]["X"]], [True], y, groups, seed=0, n_splits=n_splits)
    resid = y - p_clin
    resid_sign = (resid > 0).astype(int)  # 1 where clinical under-predicts TNBC
    out = {"residual_mean": round(float(resid.mean()), 4),
           "residual_abs_mean": round(float(np.abs(resid).mean()), 4)}
    for m in imaging_streams:
        oof = coalition_oof([streams[m]["X"]], [True], resid_sign, groups, seed=0, n_splits=n_splits)
        # AUROC of the stream predicting residual-sign; ~0.5 => no info beyond clinical.
        auc = 0.5 if len(np.unique(resid_sign)) < 2 else float(roc_auc_score(resid_sign, oof))
        out[f"{m}_predicts_clinical_residual_auroc"] = round(auc, 4)
    Xr = SimpleImputer(strategy="median").fit_transform(streams["radiomics"]["X"])
    mi = mutual_info_classif(Xr, resid_sign, random_state=0)
    out["radiomics_mi_vs_residual_max"] = round(float(mi.max()), 5)
    out["radiomics_mi_vs_residual_mean"] = round(float(mi.mean()), 5)
    out["note"] = ("~0.5 residual-AUROC and near-0 MI => the MRI streams carry ~no subtype info beyond "
                   "clinical (conditional independence). Joint KSG/MINE on 512-D BANNED (high-dim bias).")
    return out
