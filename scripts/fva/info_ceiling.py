"""Information-ceiling audit: kNN Bayes-error sweep + model-family AUROC envelope (ported from h4).

Ported (C2-5) verbatim from ``h4_info_ceiling.py`` — the estimator battery, hand-rolled primitives
(TwoNN, linear CKA, Fisher ratio), the learned-arm OOF machinery, the STOP-gate replication controls
(G1_FLOOR=0.567 gate A, EMB_ANCHOR=0.514 gate B), and the ceiling/residual verdict. These STOP-gate
values are STRUCTURAL INTEGRITY checks, not tuneable parameters (preserved per C2-5). No algorithmic
change: the code paths, RNG seeds, per-fold train-only fits, and patient-disjoint assertions are
byte-identical to h4. Constants are threaded through FVAConfig where the checklist calls for it, but
default to the frozen h4 values so a bare call reproduces h4 exactly.

banned methods (unchanged): kNN-on-raw-512-d, joint KSG/MINE on 512-D, xgboost, skdim.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from pinksight.metrics import delong_ci, ece

from fva.shuffle_sentinel import shuffle_note

# Frozen defaults (mirror h4 module constants verbatim; STOP-gates are structural, not tuneable) ---
SEEDS = [0, 1, 2]
N_SPLITS = 5
G1_FLOOR = 0.567          # radiomics-LR anchor (STOP-gate A) — within 0.02
EMB_ANCHOR = 0.514        # latent-probe subtype anchor (STOP-gate B) — within 0.03
RADIOMICS_TOL = 0.02
EMB_TOL = 0.03
PCA_VAR = 0.90
PCA_CAP = 50
KNN_KS = [1, 3, 5, 10, 15]

# Verdict thresholds — FIXED in the pre-reg. Evaluated, never chosen, here.
CEILING_AUROC_MAX = 0.62
CEILING_UB_MAX = 0.75
RESIDUAL_AUROC_MIN = 0.65
RESIDUAL_LB_MIN = 0.55
CKA_STABLE_MEAN = 0.80
CKA_STABLE_LB = 0.70


# =================================================================================================
# Hand-rolled primitives (TwoNN + CKA + Fisher — no skdim/external dep; verbatim from h4)
# =================================================================================================
def twonn_dim(X: np.ndarray) -> float:
    """TwoNN intrinsic-dimension estimator (Facco 2017): d = <log(r2/r1)>^-1 over 1st/2nd NN ratios."""
    X = np.asarray(X, float)
    d2 = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(d2, np.inf)
    dist = np.sqrt(np.sort(d2, axis=1)[:, :2])  # r1, r2 per point
    r1, r2 = dist[:, 0], dist[:, 1]
    keep = (r1 > 0) & np.isfinite(r2)
    mu = r2[keep] / r1[keep]
    mu = mu[mu > 1.0]
    return float(len(mu) / np.sum(np.log(mu)))


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA (Kornblith 2019): ||Y^T X||_F^2 / (||X^T X||_F ||Y^T Y||_F) on column-centered X,Y."""
    X = np.asarray(X, float) - np.asarray(X, float).mean(0)
    Y = np.asarray(Y, float) - np.asarray(Y, float).mean(0)
    hsic = np.linalg.norm(Y.T @ X, "fro") ** 2
    nx = np.linalg.norm(X.T @ X, "fro")
    ny = np.linalg.norm(Y.T @ Y, "fro")
    return float(hsic / (nx * ny)) if nx > 0 and ny > 0 else float("nan")


def fisher_ratio(X: np.ndarray, y: np.ndarray) -> float:
    """Multivariate Fisher discriminant ratio tr(S_b)/tr(S_w): between- over within-class scatter."""
    X = np.asarray(X, float)
    mu = X.mean(0)
    sb = sw = 0.0
    for c in np.unique(y):
        Xc = X[y == c]
        nc = len(Xc)
        muc = Xc.mean(0)
        sb += nc * np.sum((muc - mu) ** 2)
        sw += np.sum((Xc - muc) ** 2)
    return float(sb / sw) if sw > 0 else float("inf")


# =================================================================================================
# Learned-arm OOF machinery (verbatim from h4)
# =================================================================================================
def make_clf(kind: str):
    if kind == "logreg":
        return LogisticRegression(C=1.0, max_iter=2000)
    if kind == "linearsvc":
        from sklearn.calibration import CalibratedClassifierCV
        return CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=5000, dual="auto"), cv=3)
    if kind == "rbfsvm":
        from sklearn.calibration import CalibratedClassifierCV
        return CalibratedClassifierCV(SVC(C=1.0, kernel="rbf", random_state=0), cv=3, ensemble=False)
    if kind == "rf":
        return RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
    if kind == "gb":
        return GradientBoostingClassifier(random_state=0)
    raise ValueError(kind)


def _needs_impute(X: np.ndarray) -> bool:
    return bool(np.isnan(X).any())


def arm_oof(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int, kind: str,
            n_splits: int = N_SPLITS, shuffle: bool = False) -> np.ndarray:
    """Pooled OOF positive-class probability for one estimator, one seed (verbatim from h4)."""
    y = np.asarray(y)
    if shuffle:
        y = np.random.default_rng(seed).permutation(y)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan)
    impute = _needs_impute(X)
    for tr, te in cv.split(X, y, groups):
        assert set(groups[tr]).isdisjoint(groups[te]), "patient leaked across a CV fold — LOCK-2"
        Xtr, Xte = X[tr], X[te]
        if impute:
            imp = SimpleImputer(strategy="median").fit(Xtr)
            Xtr, Xte = imp.transform(Xtr), imp.transform(Xte)
        sc = StandardScaler().fit(Xtr)
        clf = make_clf(kind).fit(sc.transform(Xtr), y[tr])
        pos = list(clf.classes_).index(1)
        oof[te] = clf.predict_proba(sc.transform(Xte))[:, pos]
    assert not np.isnan(oof).any(), "OOF preds incomplete — a patient-row was never in a test fold"
    return oof


def run_family(X: np.ndarray, y: np.ndarray, groups: np.ndarray, kind: str,
               seeds=SEEDS, n_splits: int = N_SPLITS) -> dict:
    """Multi-seed pooled-OOF AUROC + DeLong CI + ECE + shuffle sentinel for one estimator family."""
    per_auc, per_ci, per_ece, per_shuf = {}, {}, {}, {}
    for s in seeds:
        oof = arm_oof(X, y, groups, seed=s, kind=kind, n_splits=n_splits, shuffle=False)
        shuf = arm_oof(X, y, groups, seed=s, kind=kind, n_splits=n_splits, shuffle=True)
        auc, lo, hi = delong_ci(y, oof)
        per_auc[s], per_ci[s] = auc, [round(lo, 4), round(hi, 4)]
        per_ece[s] = round(ece(y, oof), 4)
        per_shuf[s] = float(roc_auc_score(y, shuf))
    aucs = np.array(list(per_auc.values()))
    shufs = np.array(list(per_shuf.values()))
    lb_mean = float(np.mean([c[0] for c in per_ci.values()]))
    ub_mean = float(np.mean([c[1] for c in per_ci.values()]))
    return {
        "auroc_mean": round(float(aucs.mean()), 4),
        "auroc_std": round(float(aucs.std()), 4),
        "auroc_per_seed": {str(k): round(v, 4) for k, v in per_auc.items()},
        "delong_ci95_mean": [round(lb_mean, 4), round(ub_mean, 4)],
        "delong_lb_mean": round(lb_mean, 4),
        "delong_ub_mean": round(ub_mean, 4),
        "ece_mean": round(float(np.mean(list(per_ece.values()))), 4),
        "shuffle_auroc_mean": round(float(shufs.mean()), 4),
        "shuffle_passes": bool(shufs.mean() < aucs.mean() - 0.03),
        "shuffle_at_chance": bool(0.45 <= shufs.mean() <= 0.55),
        "integrity_note": shuffle_note(float(aucs.mean()), float(shufs.mean())),
    }


# =================================================================================================
# Arm A — kNN Bayes-error sweep on PCA-radiomics (Cover-Hart bounds; verbatim from h4)
# =================================================================================================
def knn_bayes_sweep(Xpca: np.ndarray, y: np.ndarray, groups: np.ndarray,
                    seeds=SEEDS, n_splits: int = N_SPLITS, knn_ks=KNN_KS) -> dict:
    """kNN pooled-OOF error + AUROC per k, with Cover-Hart 1-NN Bayes-error bounds (verbatim h4)."""
    out = {"per_k": {}}
    for k in knn_ks:
        per_seed_err, per_seed_auc = [], []
        for s in seeds:
            cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=s)
            oof_p = np.full(len(y), np.nan)
            oof_pred = np.full(len(y), np.nan)
            for tr, te in cv.split(Xpca, y, groups):
                assert set(groups[tr]).isdisjoint(groups[te]), "patient leak — LOCK-2"
                sc = StandardScaler().fit(Xpca[tr])
                clf = KNeighborsClassifier(n_neighbors=k).fit(sc.transform(Xpca[tr]), y[tr])
                Zte = sc.transform(Xpca[te])
                pos = list(clf.classes_).index(1)
                oof_p[te] = clf.predict_proba(Zte)[:, pos]
                oof_pred[te] = clf.predict(Zte)
            per_seed_err.append(float(np.mean(oof_pred != y)))
            per_seed_auc.append(float(roc_auc_score(y, oof_p)))
        err = float(np.mean(per_seed_err))
        out["per_k"][str(k)] = {
            "oof_error_mean": round(err, 4),
            "oof_auroc_mean": round(float(np.mean(per_seed_auc)), 4),
        }
    r1 = out["per_k"]["1"]["oof_error_mean"]
    disc = max(0.0, 1.0 - 2.0 * r1)
    bayes_lb = (1.0 - np.sqrt(disc)) / 2.0
    out["cover_hart_1nn"] = {
        "oof_error_1nn": round(r1, 4),
        "bayes_error_lower_bound": round(float(bayes_lb), 4),
        "bayes_error_upper_bound_loose": round(float(r1), 4),
        "note": ("Cover-Hart 1967: R* >= (1-sqrt(1-2 R_1NN))/2; bound is LOOSE at N~600 — the ceiling "
                 "claim rests on cross-method convergence (A+B+C+D), never on this single estimator."),
    }
    return out


# =================================================================================================
# Arm F — folded linear-CKA recipe+seed stability across the 6 representation instances (verbatim h4)
# =================================================================================================
def cka_stability(pids_ref: np.ndarray, emb_dir, seeds=SEEDS) -> dict:
    """Linear CKA across 2 recipes x 3 seeds (6 instances), aligned on the shared pid order."""
    from pathlib import Path
    emb_dir = Path(emb_dir)
    recipes = {
        "mri_embed": emb_dir / "mri_embed_s{s}.npz",
        "r18_mn_bn_fixed4": emb_dir / "r18_mn_bn_fixed4/mri_embed_s{s}.npz",
    }
    instances = {}
    for rname, tmpl in recipes.items():
        for s in seeds:
            z = np.load(str(tmpl).format(s=s), allow_pickle=True)
            pids = np.array([str(p) for p in z["pids"]])
            emb = z["emb"].astype(float)
            order = {p: i for i, p in enumerate(pids)}
            idx = [order[p] for p in pids_ref]
            instances[f"{rname}_s{s}"] = emb[idx]
    keys = list(instances.keys())
    mat = {}
    for a in keys:
        mat[a] = {}
        for b in keys:
            mat[a][b] = round(linear_cka(instances[a], instances[b]), 4)
    cross_pairs = [(a, b) for a in keys for b in keys
                   if a.split("_s")[0] == "mri_embed" and b.split("_s")[0] == "r18_mn_bn_fixed4"]
    n = len(pids_ref)
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(50):
        sub = rng.choice(n, n, replace=True)
        vals = [linear_cka(instances[a][sub], instances[b][sub]) for a, b in cross_pairs]
        boot.append(float(np.mean(vals)))
    boot = np.array(boot)
    cross_mean = float(np.mean([mat[a][b] for a, b in cross_pairs]))
    lb, ub = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    stable = bool(cross_mean >= CKA_STABLE_MEAN and lb > CKA_STABLE_LB)
    return {
        "matrix": mat,
        "cross_recipe_mean": round(cross_mean, 4),
        "cross_recipe_bootstrap_ci95": [round(lb, 4), round(ub, 4)],
        "sub_verdict": "STABLE" if stable else "REPORTED-AS-IS",
        "scope_caveat": ("recipe+seed invariance on ONE architecture (r18+MedicalNet) — NOT "
                         "architecture invariance; other encoders' weights were never saved."),
    }


# =================================================================================================
# Embedding-anchor replication control (verbatim from h4)
# =================================================================================================
def latent_probe_replication(manifest, emb_dir, seeds=SEEDS) -> dict:
    """Reproduce g2_latent_probe.py EXACTLY: StratifiedKFold(5, rs=0), scaler-only, LogReg(2000)."""
    from pathlib import Path
    emb_dir = Path(emb_dir)
    man = pd.read_csv(manifest).set_index("patient_id")
    per_seed = []
    for s in seeds:
        z = np.load(emb_dir / f"mri_embed_s{s}.npz", allow_pickle=True)
        pids = np.array([str(p) for p in z["pids"]])
        emb = z["emb"].astype(float)
        y = man.loc[pids, "subtype"].map({"luminal_like": 0, "tnbc": 1}).to_numpy()
        mask = ~pd.isna(y)
        e, yy = emb[mask], y[mask].astype(int)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        oof = np.zeros(len(yy))
        for tr, te in skf.split(e, yy):
            sc = StandardScaler().fit(e[tr])
            clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(e[tr]), yy[tr])
            oof[te] = clf.predict_proba(sc.transform(e[te]))[:, list(clf.classes_).index(1)]
        per_seed.append(float(roc_auc_score(yy, oof)))
    return {"auroc_mean": round(float(np.mean(per_seed)), 4),
            "auroc_per_seed": [round(v, 4) for v in per_seed],
            "protocol": "latent-probe exact: StratifiedKFold(5, rs=0), scaler-only, LogReg(max_iter=2000)"}


# =================================================================================================
# Self-check — synthetic + replication controls (verbatim from h4; STOP BLOCKED on any failure)
# =================================================================================================
def selfcheck(Xrad, y, groups, Xemb0, manifest, emb_dir) -> dict:
    rng = np.random.default_rng(0)
    t = rng.uniform(0, 1, (2000, 2))
    basis, _ = np.linalg.qr(rng.normal(size=(10, 2)))
    manifold = t @ basis.T
    dim = twonn_dim(manifold)
    assert 1.6 < dim < 2.6, f"self-check(f): TwoNN should read ~2 on a 2-D manifold, got {dim:.3f}"

    A = rng.normal(size=(300, 20))
    assert abs(linear_cka(A, A) - 1.0) < 1e-6, "self-check(g): CKA(X,X) must be 1.0"
    q, _ = np.linalg.qr(rng.normal(size=(20, 20)))
    assert linear_cka(A, A @ q) > 0.999, "self-check(g): CKA must be orthogonal-invariant (~1.0)"
    B = rng.normal(size=(300, 20))
    assert linear_cka(A, B) < 0.2, f"self-check(g): CKA of independent Gaussians must be ~0, got {linear_cka(A,B):.3f}"

    rad_lr = run_family(Xrad, y, groups, "logreg")
    rad_auc = rad_lr["auroc_mean"]
    a_ok = abs(rad_auc - G1_FLOOR) <= RADIOMICS_TOL
    assert rad_lr["shuffle_passes"], f"self-check(a): radiomics-LR shuffle sentinel failed: {rad_lr['shuffle_auroc_mean']}"
    assert a_ok, (f"STOP-gate A: radiomics-LR pooled-OOF {rad_auc:.4f} off the G1 floor {G1_FLOOR} "
                  f"by {abs(rad_auc-G1_FLOOR):.4f} > {RADIOMICS_TOL} — data/CV drift; STOP BLOCKED.")

    emb_rep = latent_probe_replication(manifest, emb_dir)
    emb_auc = emb_rep["auroc_mean"]
    b_ok = abs(emb_auc - EMB_ANCHOR) <= EMB_TOL
    assert b_ok, (f"STOP-gate B: embedding-LR replication {emb_auc:.4f} off the latent-probe anchor "
                  f"{EMB_ANCHOR} by {abs(emb_auc-EMB_ANCHOR):.4f} > {EMB_TOL} — embedding/label "
                  f"mismatch vs the latent probe; STOP BLOCKED.")

    Xi = SimpleImputer(strategy="median").fit_transform(Xrad)
    Xs = StandardScaler().fit_transform(Xi)
    pca = PCA(n_components=min(PCA_CAP, Xs.shape[1])).fit(Xs)
    cum = np.cumsum(pca.explained_variance_ratio_)
    npc = int(np.searchsorted(cum, PCA_VAR) + 1)
    assert cum[npc - 1] >= PCA_VAR - 1e-9, f"self-check(e): {npc} PCs explain {cum[npc-1]:.3f} < {PCA_VAR}"

    print(f"[selfcheck] OK — TwoNN={dim:.2f}; CKA identity/rotation/indep sane; "
          f"radiomics-LR={rad_auc:.4f} (floor {G1_FLOOR}, gate A {'OK' if a_ok else 'FAIL'}); "
          f"embed-LR={emb_auc:.4f} (anchor {EMB_ANCHOR}, gate B {'OK' if b_ok else 'FAIL'}); "
          f"{npc} PCs>={PCA_VAR} var")
    return {"radiomics_lr": rad_lr, "embedding_replication": emb_rep,
            "twonn_selfcheck_2d": round(dim, 4), "n_pc_ge_90var": npc,
            "gate_a_ok": bool(a_ok), "gate_b_ok": bool(b_ok)}


def build_pca_radiomics(Xrad: np.ndarray) -> tuple[np.ndarray, int, float]:
    """Global PCA on median-imputed+scaled radiomics: smallest #PC >=90% var, capped <=50 (verbatim)."""
    Xi = SimpleImputer(strategy="median").fit_transform(Xrad)
    Xs = StandardScaler().fit_transform(Xi)
    pca = PCA(n_components=min(PCA_CAP, Xs.shape[1])).fit(Xs)
    cum = np.cumsum(pca.explained_variance_ratio_)
    npc = int(np.searchsorted(cum, PCA_VAR) + 1)
    Xpca = pca.transform(Xs)[:, :npc]
    return Xpca, npc, float(cum[npc - 1])
