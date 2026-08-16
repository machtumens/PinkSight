
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from pinksight import FORBIDDEN_FEATURES
from pinksight.metrics import delong_ci, ece  

AGE_FEATURE = "Age at diagnosis (years)"  
DOB_DAYS_COL = "Date of Birth (Days)"  
FEATURES_NUM = (
    "Staging(Tumor Size)# [T]",
    "Staging(Nodes)#(Nx replaced by -1)[N]",
    "Nottingham grade",  
    AGE_FEATURE,
)
FEATURES_CAT = (
    "Menopause (at diagnosis)",
    "Race and Ethnicity",
    "Multicentric/Multifocal",
    "Metastatic at Presentation (Outside of Lymph Nodes)",
    "Lymphadenopathy or Suspicious Nodes",
)
FEATURES = FEATURES_NUM + FEATURES_CAT  

_LABEL = {0: 0, 3: 1}
SEEDS = (0, 1, 2)  
N_SPLITS = 5


class LeakageError(RuntimeError):
    pass


def _assert_leak_free() -> None:
    leaked = set(FEATURES) & set(FORBIDDEN_FEATURES)
    if leaked:
        raise LeakageError(f"forbidden fields in classifier inputs: {sorted(leaked)}")


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    s = df[name]
    return s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s  


def load_bpe(npz_path: Path) -> tuple[dict[str, np.ndarray], list[str]]:
    from pinksight import FORBIDDEN_FEATURES

    d = np.load(npz_path, allow_pickle=True)
    names = [str(n) for n in d["feature_names"]]
    if not set(names).isdisjoint(FORBIDDEN_FEATURES):
        raise LeakageError(f"BPE names hit forbidden set: {sorted(set(names) & FORBIDDEN_FEATURES)}")
    ids = [str(p) for p in d["patient_ids"]]
    return {pid: row for pid, row in zip(ids, np.asarray(d["features"], float))}, names


def load_xy(
    clin_path: Path, split_yaml: Path, bpe_npz: Path | None = None
) -> tuple[np.ndarray, np.ndarray, list[int], list[str], list[int]]:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
    from audit_ki67 import load as load_clinical  

    _assert_leak_free()
    df = load_clinical(clin_path)
    df.columns = [str(c) for c in df.columns]
    df[AGE_FEATURE] = -pd.to_numeric(_col(df, DOB_DAYS_COL), errors="coerce") / 365.25
    pid = _col(df, "Patient ID").astype(str).str.strip()
    subtype = pd.to_numeric(_col(df, "Mol Subtype"), errors="coerce").map(_LABEL)

    dev = set(yaml.safe_load(split_yaml.read_text())["dev"])
    keep = pid.isin(dev) & subtype.notna()
    df, pid, subtype = df[keep], pid[keep], subtype[keep]

    x_num = np.column_stack(
        [pd.to_numeric(_col(df, c), errors="coerce").to_numpy(float) for c in FEATURES_NUM]
    )
    cat_codes, cards = [], []
    for c in FEATURES_CAT:
        codes, uniq = pd.factorize(pd.to_numeric(_col(df, c), errors="coerce"), use_na_sentinel=False)
        cat_codes.append(codes + 1)  
        cards.append(len(uniq) + 1)  
    x_cat = np.column_stack(cat_codes).astype(int)

    if bpe_npz is not None:
        bpe_map, bpe_names = load_bpe(bpe_npz)
        n_bpe = len(bpe_names)
        bpe_cols = np.full((len(pid), n_bpe), np.nan)
        for i, p in enumerate(pid):
            if p in bpe_map:
                bpe_cols[i] = bpe_map[p]
        x_num = np.column_stack([x_num, bpe_cols])

    return x_num, x_cat, [int(v) for v in subtype], list(pid), cards


def _fit_eval_fold(x_num, x_cat, y, tr, te, cards, seed, epochs=80, return_state=False):
    import torch
    from sklearn.metrics import roc_auc_score
    from rtdl_revisiting_models import FTTransformer

    from pinksight.seed import set_seed

    set_seed(seed)
    med = np.nanmedian(x_num[tr], axis=0)
    xn = np.where(np.isnan(x_num), med, x_num)
    mu, sd = xn[tr].mean(0), xn[tr].std(0) + 1e-8
    xn = (xn - mu) / sd

    dev = "cpu"
    xc_t = torch.tensor(xn, dtype=torch.float32, device=dev)
    xq_t = torch.tensor(x_cat, dtype=torch.long, device=dev)
    y_t = torch.tensor(y, dtype=torch.float32, device=dev).unsqueeze(1)

    model = FTTransformer(
        n_cont_features=xn.shape[1],
        cat_cardinalities=cards,
        d_out=1,
        **FTTransformer.get_default_kwargs(n_blocks=2),
    ).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    pos = float(np.sum(np.asarray(y)[tr] == 1))
    neg = float(np.sum(np.asarray(y)[tr] == 0))
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / max(pos, 1.0)]))

    tr_t = torch.tensor(tr, device=dev)
    model.train()
    for _ in range(epochs):  
        opt.zero_grad()
        out = model(xc_t[tr_t], xq_t[tr_t])
        loss_fn(out, y_t[tr_t]).backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(xc_t[torch.tensor(te)], xq_t[torch.tensor(te)])).cpu().numpy()
    prob = prob.ravel()
    fold_auc = float(roc_auc_score(np.asarray(y)[te], prob))
    if return_state:
        state = {
            "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "mu": np.asarray(mu, dtype=float),
            "sd": np.asarray(sd, dtype=float),
            "med": np.asarray(med, dtype=float),
            "cards": list(cards),
            "seed": int(seed),
            "epochs": int(epochs),
        }
        return fold_auc, np.asarray(te), prob, state
    return fold_auc, np.asarray(te), prob  


def _fit_fold_embed(x_num, x_cat, y, tr, te, cards, seed, epochs=80):
    import torch
    from rtdl_revisiting_models import FTTransformer

    from pinksight.seed import set_seed

    set_seed(seed)
    med = np.nanmedian(x_num[tr], axis=0)
    xn = np.where(np.isnan(x_num), med, x_num)
    mu, sd = xn[tr].mean(0), xn[tr].std(0) + 1e-8
    xn = (xn - mu) / sd

    dev = "cpu"
    xc_t = torch.tensor(xn, dtype=torch.float32, device=dev)
    xq_t = torch.tensor(x_cat, dtype=torch.long, device=dev)
    y_t = torch.tensor(y, dtype=torch.float32, device=dev).unsqueeze(1)

    model = FTTransformer(
        n_cont_features=xn.shape[1],
        cat_cardinalities=cards,
        d_out=1,
        **FTTransformer.get_default_kwargs(n_blocks=2),
    ).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    pos = float(np.sum(np.asarray(y)[tr] == 1))
    neg = float(np.sum(np.asarray(y)[tr] == 0))
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / max(pos, 1.0)]))

    tr_t = torch.tensor(tr, device=dev)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        out = model(xc_t[tr_t], xq_t[tr_t])
        loss_fn(out, y_t[tr_t]).backward()
        opt.step()

    model.eval()
    te_t = torch.tensor(te)
    with torch.no_grad():
        prob = torch.sigmoid(model(xc_t[te_t], xq_t[te_t])).cpu().numpy().ravel()
        saved_output = model.backbone.output
        model.backbone.output = torch.nn.Identity()
        try:
            emb = model(xc_t[te_t], xq_t[te_t]).cpu().numpy()
        finally:
            model.backbone.output = saved_output
    bg_train = np.column_stack([xn[tr], x_cat[tr].astype(float)])
    return np.asarray(te), prob, np.asarray(emb, dtype=float), bg_train


def export_oof_embeddings(
    x_num: np.ndarray,
    x_cat: np.ndarray,
    y: Sequence[int],
    groups: Sequence,
    cards: list[int],
    out_dir: Path,
    seeds: tuple[int, ...] = SEEDS,
    background_n: int = 100,
) -> dict:
    from sklearn.model_selection import StratifiedGroupKFold

    _assert_leak_free()  
    feature_names = list(FEATURES)
    leaked = set(feature_names) & set(FORBIDDEN_FEATURES)
    if leaked:
        raise LeakageError(f"forbidden fields in exported clinical feature_names: {sorted(leaked)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    y = list(y)
    n = len(y)
    d_block = None
    bg_pool: list[np.ndarray] = []  
    written = []
    for seed in seeds:
        cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        emb_by_row: dict[int, np.ndarray] = {}
        for tr, te in cv.split(x_num, y, groups):
            if not set(np.asarray(groups)[tr]).isdisjoint(np.asarray(groups)[te]):
                raise LeakageError("patient leaked across a CV fold — LOCK-2 violation")
            te_idx, _prob, emb, bg_train = _fit_fold_embed(x_num, x_cat, y, tr, te, cards, seed)
            d_block = emb.shape[1]
            for j, row in zip(te_idx.tolist(), emb):
                emb_by_row[j] = np.asarray(row, dtype=float)
            if seed == seeds[0]:
                bg_pool.append(bg_train)
        if len(emb_by_row) != n:
            raise RuntimeError(
                f"OOF clinical embeddings incomplete for seed {seed}: {len(emb_by_row)}/{n} rows.")
        order = list(range(n))
        emb_mat = np.stack([emb_by_row[j] for j in order]).astype(float)
        pids_ordered = [str(groups[j]) for j in order]  
        p = out_dir / f"clinical_embed_s{seed}.npz"
        np.savez(p, pids=np.array(pids_ordered, dtype=object), emb=emb_mat)
        written.append(str(p))

    bg_all = np.concatenate(bg_pool) if bg_pool else np.zeros((0, len(feature_names)))
    rng = np.random.default_rng(0)
    k = min(background_n, bg_all.shape[0])
    bg_sample = bg_all[rng.choice(bg_all.shape[0], size=k, replace=False)] if k > 0 else bg_all
    bg_p = out_dir / "clinical_background.npz"
    np.savez(bg_p, background_X=bg_sample.astype(float),
             feature_names=np.array(feature_names, dtype=object),
             cards=np.array(list(cards), dtype=int))
    written.append(str(bg_p))
    return {"files": written, "d_block": d_block, "n_dev": n,
            "background_n": int(bg_sample.shape[0]), "feature_names": feature_names,
            "cards": [int(c) for c in cards]}


def fit_on_full_dev(x_num, x_cat, y, cards, seed, epochs=80):
    import torch
    from rtdl_revisiting_models import FTTransformer

    from pinksight.seed import set_seed

    set_seed(seed)
    med = np.nanmedian(x_num, axis=0)
    xn = np.where(np.isnan(x_num), med, x_num)
    mu, sd = xn.mean(0), xn.std(0) + 1e-8
    xn = (xn - mu) / sd

    dev = "cpu"
    xc_t = torch.tensor(xn, dtype=torch.float32, device=dev)
    xq_t = torch.tensor(x_cat, dtype=torch.long, device=dev)
    y_t = torch.tensor(y, dtype=torch.float32, device=dev).unsqueeze(1)

    model = FTTransformer(
        n_cont_features=xn.shape[1],
        cat_cardinalities=cards,
        d_out=1,
        **FTTransformer.get_default_kwargs(n_blocks=2),
    ).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    pos = float(np.sum(np.asarray(y) == 1))
    neg = float(np.sum(np.asarray(y) == 0))
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / max(pos, 1.0)]))

    model.train()
    for _ in range(epochs):  
        opt.zero_grad()
        out = model(xc_t, xq_t)
        loss_fn(out, y_t).backward()
        opt.step()

    return {
        "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "mu": np.asarray(mu, dtype=float),
        "sd": np.asarray(sd, dtype=float),
        "med": np.asarray(med, dtype=float),
        "cards": list(cards),
        "seed": int(seed),
        "epochs": int(epochs),
    }


def predict_with_state(state: dict, x_num_ext: np.ndarray, x_cat_ext: np.ndarray) -> np.ndarray:
    import torch
    from rtdl_revisiting_models import FTTransformer

    med = np.asarray(state["med"], float)
    mu = np.asarray(state["mu"], float)
    sd = np.asarray(state["sd"], float)
    cards = list(state["cards"])

    xn = np.where(np.isnan(x_num_ext), med, x_num_ext)  
    xn = (xn - mu) / sd  

    xc = np.asarray(x_cat_ext, dtype=int).copy()
    for j, card in enumerate(cards):
        bad = (xc[:, j] < 0) | (xc[:, j] >= card)
        xc[xc[:, j] >= card, j] = 0
        xc[bad, j] = 0

    model = FTTransformer(
        n_cont_features=xn.shape[1],
        cat_cardinalities=cards,
        d_out=1,
        **FTTransformer.get_default_kwargs(n_blocks=2),
    )
    model.load_state_dict({k: torch.as_tensor(v) for k, v in state["state_dict"].items()})
    model.eval()
    with torch.no_grad():
        logits = model(
            torch.tensor(xn, dtype=torch.float32), torch.tensor(xc, dtype=torch.long)
        ).cpu().numpy().ravel()
    return 1.0 / (1.0 + np.exp(-logits))


def _logreg_oh_encoder(x_cat_dev: np.ndarray):
    from sklearn.preprocessing import OneHotEncoder

    return OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(x_cat_dev)


def fit_logreg_on_full_dev(x_num, x_cat, y, cards, seed):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    _assert_leak_free()
    oh = _logreg_oh_encoder(np.asarray(x_cat, dtype=int))
    X = np.hstack([np.asarray(x_num, float), oh.transform(np.asarray(x_cat, dtype=int))])
    imputer = SimpleImputer(strategy="median").fit(X)  
    Xi = imputer.transform(X)
    scaler = StandardScaler().fit(Xi)  
    Xs = scaler.transform(Xi)
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=int(seed)).fit(Xs, np.asarray(y, int))
    return {
        "oh": oh,
        "imputer": imputer,
        "scaler": scaler,
        "clf": clf,
        "cards": list(cards),
        "seed": int(seed),
        "estimator": "LogReg(C=1.0, max_iter=1000) + OneHot(handle_unknown=ignore) — H6 coalition estimator",
    }


def predict_logreg_with_state(state: dict, x_num_ext: np.ndarray, x_cat_ext: np.ndarray) -> np.ndarray:
    oh, imputer, scaler, clf = state["oh"], state["imputer"], state["scaler"], state["clf"]
    X = np.hstack([np.asarray(x_num_ext, float), oh.transform(np.asarray(x_cat_ext, dtype=int))])
    Xs = scaler.transform(imputer.transform(X))
    pos = list(clf.classes_).index(1)
    return clf.predict_proba(Xs)[:, pos]


def logreg_oof(x_num, x_cat, y, groups, cards, seed):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    x_num = np.asarray(x_num, float)
    x_cat = np.asarray(x_cat, dtype=int)
    y = np.asarray(list(y), int)
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=int(seed))
    oof = np.full(len(y), np.nan)
    for tr, te in cv.split(x_num, y, groups):
        if not set(np.asarray(groups)[tr]).isdisjoint(np.asarray(groups)[te]):
            raise LeakageError("patient leaked across a CV fold — LOCK-2 violation")
        oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(x_cat[tr])
        Xtr = np.hstack([x_num[tr], oh.transform(x_cat[tr])])
        Xte = np.hstack([x_num[te], oh.transform(x_cat[te])])
        imp = SimpleImputer(strategy="median").fit(Xtr)
        sc = StandardScaler().fit(imp.transform(Xtr))
        Xtr, Xte = sc.transform(imp.transform(Xtr)), sc.transform(imp.transform(Xte))
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=int(seed)).fit(Xtr, y[tr])
        pos = list(clf.classes_).index(1)
        oof[te] = clf.predict_proba(Xte)[:, pos]
    if np.isnan(oof).any():
        raise RuntimeError("OOF preds incomplete — a patient-row was never in a test fold")
    return oof


def logreg_cross_val_auroc(x_num, x_cat, y, groups, cards, seeds=SEEDS):
    x_num = np.asarray(x_num, float)
    x_cat = np.asarray(x_cat, dtype=int)
    yv = np.asarray(list(y), int)
    pooled, ci, ece_seed = {}, {}, {}
    for seed in seeds:
        oof = logreg_oof(x_num, x_cat, yv, groups, cards, seed)
        auc, lo, hi = delong_ci(yv, oof)
        pooled[seed], ci[seed], ece_seed[seed] = auc, (lo, hi), ece(yv, oof)
    pooled_v = np.array(list(pooled.values()))
    return {
        "auroc_pooled_oof_mean": float(pooled_v.mean()),
        "auroc_pooled_oof_per_seed": {str(k): round(v, 4) for k, v in pooled.items()},
        "auroc_std_across_seeds": float(pooled_v.std()),
        "delong_ci95_mean": [
            round(float(np.mean([lo for lo, _ in ci.values()])), 4),
            round(float(np.mean([hi for _, hi in ci.values()])), 4),
        ],
        "delong_ci95_per_seed": {str(k): [round(lo, 4), round(hi, 4)] for k, (lo, hi) in ci.items()},
        "ece_mean": round(float(np.mean(list(ece_seed.values()))), 4),
        "ece_per_seed": {str(k): round(v, 4) for k, v in ece_seed.items()},
        "n_dev": len(yv),
        "tnbc_prevalence": round(float(yv.mean()), 4),
        "n_splits": N_SPLITS,
        "seeds": list(seeds),
        "estimator": "LogReg(C=1.0, max_iter=1000) + OneHot(handle_unknown=ignore) — H6 coalition estimator",
    }


def save_impute_stats(
    x_num: np.ndarray, x_cat: np.ndarray, groups: Sequence, cards: list[int], out_path: Path
) -> dict:
    import pickle

    _assert_leak_free()
    med = np.nanmedian(x_num, axis=0)
    xn = np.where(np.isnan(x_num), med, x_num)
    mu, sd = xn.mean(0), xn.std(0) + 1e-8
    stats = {
        "source": "duke_dev_full",
        "n_dev": int(len(groups)),
        "med": np.asarray(med, float),
        "mu": np.asarray(mu, float),
        "sd": np.asarray(sd, float),
        "cards": [int(c) for c in cards],
        "feature_names_num": list(FEATURES_NUM),
        "feature_names_cat": list(FEATURES_CAT),
        "note": (
            "Duke-TRAIN (full dev cohort) impute+standardize stats. LOCK-2: the ONLY stats that may "
            "transform the ISPY2 external table; never fit any statistic on external data."
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as fh:
        pickle.dump(stats, fh)
    return stats


def cross_val_auroc(
    x_num: np.ndarray, x_cat: np.ndarray, y: Sequence[int], groups: Sequence, cards: list[int]
) -> dict:
    from sklearn.model_selection import StratifiedGroupKFold

    y = list(y)
    yv = np.asarray(y)
    per_seed = {}  
    pooled, ci, ece_seed = {}, {}, {}  
    for seed in SEEDS:
        cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        fold_aucs = []
        oof = np.full(len(y), np.nan)  
        for tr, te in cv.split(x_num, y, groups):
            if not set(np.asarray(groups)[tr]).isdisjoint(np.asarray(groups)[te]):
                raise LeakageError("patient leaked across a CV fold — LOCK-2 violation")
            fold_auc, te_idx, probs = _fit_eval_fold(x_num, x_cat, y, tr, te, cards, seed)
            fold_aucs.append(fold_auc)
            oof[te_idx] = probs
        if np.isnan(oof).any():
            raise RuntimeError("OOF preds incomplete — a patient-row was never in a test fold")
        per_seed[seed] = float(np.mean(fold_aucs))
        auc, lo, hi = delong_ci(yv, oof)  
        pooled[seed], ci[seed] = auc, (lo, hi)
        ece_seed[seed] = ece(yv, oof)
    means = np.array(list(per_seed.values()))
    pooled_v = np.array(list(pooled.values()))
    return {
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
        "n_dev": len(y),
        "tnbc_prevalence": round(float(np.mean(y)), 4),
        "n_splits": N_SPLITS,
        "seeds": list(SEEDS),
    }


def _next_exp_dir(reports: Path) -> Path:
    reports.mkdir(parents=True, exist_ok=True)
    n = 1 + max(
        (int(p.name.split("-")[1]) for p in reports.glob("EXP-*") if p.name.split("-")[1].isdigit()),
        default=0,
    )
    d = reports / f"EXP-{n:03d}"
    d.mkdir(exist_ok=True)
    return d


def run(clin_path: Path, split_yaml: Path, reports: Path) -> dict:
    x_num, x_cat, y, groups, cards = load_xy(clin_path, split_yaml)
    metrics = cross_val_auroc(x_num, x_cat, y, groups, cards)
    metrics.update(
        model="FT-Transformer (rtdl_revisiting_models, default n_blocks=2)",
        features={"numeric": list(FEATURES_NUM), "categorical": list(FEATURES_CAT)},
        forbidden_excluded=sorted(FORBIDDEN_FEATURES),
        caveat=(
            "small-N FTT baseline (high seed variance); age recovered (DOB-derived, no impute); "
            "grade = composite Nottingham grade (was mis-wired to the col-31 Tubule sub-score), "
            "67% collected → train-fold median-imputed, at-diagnosis biopsy-derived, label-safe; "
            "continuous tumour-size-cm dropped (90/922 collected, rest 'NC'). AUROC reported with "
            "DeLong 95% CI + ECE (docs/CLAIM_LEDGER.md eval rule)."
        ),
        gate="G2",
    )
    exp = _next_exp_dir(reports)
    (exp / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return {"exp_dir": str(exp), **metrics}


def selfcheck() -> int:
    _assert_leak_free()  
    assert len(FEATURES) == len(set(FEATURES)) == 9, "feature set drifted"
    print("selfcheck OK — FEATURES (9) disjoint from FORBIDDEN; leak guard armed")  
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clinical", type=Path, default=Path("data/raw/Clinical_and_Other_Features.xlsx"))
    ap.add_argument("--split", type=Path, default=Path("configs/split_v2.yaml"))
    ap.add_argument("--reports", type=Path, default=Path("reports"))
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args(argv)
    if args.selfcheck:
        return selfcheck()
    if not args.clinical.exists():
        print(f"Not found: {args.clinical} (data is git-ignored / not downloaded).")  
        return 2
    out = run(args.clinical, args.split, args.reports)
    print(json.dumps(out, indent=2))  
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
