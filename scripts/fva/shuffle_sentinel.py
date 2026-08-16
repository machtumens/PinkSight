
from __future__ import annotations

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


def coalition_oof(
    mats: list[np.ndarray],
    needs_impute: list[bool],
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
    n_splits: int = 5,
    shuffle: bool = False,
) -> np.ndarray:
    y = np.asarray(y)
    if shuffle:
        y = np.random.default_rng(seed).permutation(y)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan)
    for tr, te in cv.split(mats[0], y, groups):
        assert set(groups[tr]).isdisjoint(groups[te]), "patient leaked across a CV fold — LOCK-2"
        parts_tr, parts_te = [], []
        for M, imp_flag in zip(mats, needs_impute):
            Mtr, Mte = M[tr], M[te]
            if imp_flag:
                imp = SimpleImputer(strategy="median").fit(Mtr)
                Mtr, Mte = imp.transform(Mtr), imp.transform(Mte)
            sc = StandardScaler().fit(Mtr)
            parts_tr.append(sc.transform(Mtr))
            parts_te.append(sc.transform(Mte))
        Xtr, Xte = np.hstack(parts_tr), np.hstack(parts_te)
        clf = LogisticRegression(C=1.0, max_iter=1000).fit(Xtr, y[tr])
        pos = list(clf.classes_).index(1)
        oof[te] = clf.predict_proba(Xte)[:, pos]
    assert not np.isnan(oof).any(), "OOF preds incomplete — a patient-row was never in a test fold"
    return oof


def empty_coalition_oof(y: np.ndarray, groups: np.ndarray, seed: int, n_splits: int = 5) -> np.ndarray:
    y = np.asarray(y)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan)
    for tr, te in cv.split(np.zeros((len(y), 1)), y, groups):
        oof[te] = y[tr].mean()
    return oof


def shuffle_note(real: float, shuf: float) -> str:
    at_chance = 0.45 <= shuf <= 0.55
    if not at_chance:
        return f"WARN: shuffle {shuf:.3f} not at chance — possible peeking; treat this arm's number with care."
    if real - shuf < 0.03:
        return (f"shuffle {shuf:.3f} at chance AND real {real:.3f} at/near chance -> NULL signature "
                f"(no signal to inflate); 'shuffle<real-0.03' is vacuous here, not a peeking flag.")
    return f"shuffle {shuf:.3f} at chance and below real {real:.3f} by >=0.03 -> integrity OK, real signal is genuine."


def stream_shuffle_sentinels(
    streams: dict,
    y: np.ndarray,
    groups: np.ndarray,
    coalition_detail: dict,
    seeds: tuple[int, ...] = (0, 1, 2),
    n_splits: int = 5,
    stream_names: tuple[str, ...] = ("clinical", "radiomics", "mri"),
) -> dict:
    out = {}
    for name in stream_names:
        X, impute = streams[name]["X"], streams[name]["impute"]
        per_shuf = []
        for s in seeds:
            shuf_oof = coalition_oof([X], [impute], y, groups, seed=s, n_splits=n_splits, shuffle=True)
            per_shuf.append(float(roc_auc_score(y, shuf_oof)))
        shuf_mean = float(np.mean(per_shuf))
        real_mean = float(coalition_detail[name]["auroc_mean"])
        out[name] = {
            "shuffle_auroc_mean": round(shuf_mean, 4),
            "real_auroc_mean": round(real_mean, 4),
            "shuffle_auroc_per_seed": [round(v, 4) for v in per_shuf],
            "shuffle_passes": bool(shuf_mean < real_mean - 0.03),
            "shuffle_at_chance": bool(0.45 <= shuf_mean <= 0.55),
            "integrity_note": shuffle_note(real_mean, shuf_mean),
        }
    return out
