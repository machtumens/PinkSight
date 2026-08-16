
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
    p_clin = coalition_oof([streams["clinical"]["X"]], [True], y, groups, seed=0, n_splits=n_splits)
    resid = y - p_clin
    resid_sign = (resid > 0).astype(int)  
    out = {"residual_mean": round(float(resid.mean()), 4),
           "residual_abs_mean": round(float(np.abs(resid).mean()), 4)}
    for m in imaging_streams:
        oof = coalition_oof([streams[m]["X"]], [True], resid_sign, groups, seed=0, n_splits=n_splits)
        auc = 0.5 if len(np.unique(resid_sign)) < 2 else float(roc_auc_score(resid_sign, oof))
        out[f"{m}_predicts_clinical_residual_auroc"] = round(auc, 4)
    Xr = SimpleImputer(strategy="median").fit_transform(streams["radiomics"]["X"])
    mi = mutual_info_classif(Xr, resid_sign, random_state=0)
    out["radiomics_mi_vs_residual_max"] = round(float(mi.max()), 5)
    out["radiomics_mi_vs_residual_mean"] = round(float(mi.mean()), 5)
    out["note"] = ("~0.5 residual-AUROC and near-0 MI => the MRI streams carry ~no subtype info beyond "
                   "clinical (conditional independence). Joint KSG/MINE on 512-D BANNED (high-dim bias).")
    return out
