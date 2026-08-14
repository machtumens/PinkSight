"""P05 / G2 clinical-only baseline: FT-Transformer over LABEL-SAFE clinical features (LOCK-3).

The clinical arm's standalone number, reported BEFORE fusion so the fusion delta has a floor.
Two integrity rails (LOCK-2), enforced in code, not just docs:

  * leak guard — the input feature list is asserted disjoint from `FORBIDDEN_FEATURES` at load
    time (ER/PR/HER2/Ki-67/Mol-Subtype/Oncotype define the IHC label → using them is circular).
  * train-only stats — numeric impute+standardize fit on the TRAIN fold only; patient-grouped,
    class-stratified CV on the DEV cohort. The sealed Avanto holdout is never touched here.

ponytail caveat (decisions.md): FT-Transformer on ~624 rows × 9 low-cardinality features is the
small-N regime where it underperforms LogReg/GBDT with wide seed variance (its own rtdl paper says
so). We report it per the locked P05 SUCCESS criterion; treat the number as a high-variance floor,
not a tuned result. Age-at-diagnosis IS recovered (derived from `Date of Birth (Days)`, day-0 =
diagnosis → -DOB/365.25; 624/624 dev, no imputation). Continuous tumour-size-cm stays dropped: only
90/922 rows carry it (the rest are literally 'NC' = not collected) — imputing 85% would be the
silent-imputation the ledger forbids, so the honest gap stands (T-stage already carries staging).

Grade is the composite **Nottingham grade** (= bin(Tubule+Nuclear+Mitotic), 1=low/2=int/3=high,
99.2% reconstructs from the T/N/M sub-cols), NOT the Tubule-only sub-score that the col-31
"Tumor Grade" field actually holds — the prior wiring fed that single component (solo CV-AUROC
~0.51, near chance) and left the real grade signal on the floor. Nottingham grade is at-diagnosis
(biopsy-derived "Tumor Characteristics" block, upstream of surgery), label-safe (not in the IHC
panel), and 33% missing → train-fold median-imputed (lift measured WITH imputation in place:
pooled-OOF 0.598→0.644, EXP-002→EXP-003, age held fixed). Its mitotic component ∝ proliferation:
do NOT reuse it naively in the Ki-67 head
(closer to circular there); it is clean for the ER/PR/HER2-defined subtype task. See decisions.md.

Numbers are reported per CLAUDE.md's eval rule — never bare: multi-seed spread + per-seed pooled
out-of-fold AUROC with a DeLong 95% CI + ECE calibration (closes red-team M1, see metrics module).

torch/sklearn are imported LAZILY inside the training fns so `tests/test_leakage.py` can pull
`FEATURES` without the ml extra installed.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from pinksight import FORBIDDEN_FEATURES
from pinksight.metrics import delong_ci, ece  # pure-numpy; keeps this module's tests torch-free

# LABEL-SAFE at-diagnosis clinical features. Most parse >=0.97 on the dev cohort; the lone
# exception is Nottingham grade (67% collected → train-fold median-imputed, see _fit_eval_fold).
# Ordinal/continuous → numeric; the rest → categorical embeddings.
AGE_FEATURE = "Age at diagnosis (years)"  # DERIVED in load_xy from "Date of Birth (Days)"
DOB_DAYS_COL = "Date of Birth (Days)"  # day-0 == diagnosis, DOB is negative days before it
FEATURES_NUM = (
    "Staging(Tumor Size)# [T]",
    "Staging(Nodes)#(Nx replaced by -1)[N]",
    "Nottingham grade",  # composite histologic grade — replaces the col-31 "Tumor Grade" field,
                         # which is only the Tubule sub-score (see module docstring + decisions.md)
    AGE_FEATURE,
)
FEATURES_CAT = (
    "Menopause (at diagnosis)",
    "Race and Ethnicity",
    "Multicentric/Multifocal",
    "Metastatic at Presentation (Outside of Lymph Nodes)",
    "Lymphadenopathy or Suspicious Nodes",
)
FEATURES = FEATURES_NUM + FEATURES_CAT  # the classifier's full input set — guarded below

# Duke "Mol Subtype": 0 = luminal-like (neg class), 3 = TNBC (pos class). 1,2 out of LOCK-3 scope.
_LABEL = {0: 0, 3: 1}
SEEDS = (0, 1, 2)  # 3 = the eval-integrity minimum (CLAUDE.md "multi-seed spread 3 min / 5 target")
N_SPLITS = 5


class LeakageError(RuntimeError):
    """A FORBIDDEN (label-defining) field reached the classifier input set — LOCK-2 breach."""


def _assert_leak_free() -> None:
    leaked = set(FEATURES) & set(FORBIDDEN_FEATURES)
    if leaked:
        raise LeakageError(f"forbidden fields in classifier inputs: {sorted(leaked)}")


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    s = df[name]
    return s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s  # duplicate-label guard


def load_bpe(npz_path: Path) -> tuple[dict[str, np.ndarray], list[str]]:
    """BP1 contralateral-BPE features: (patient_id -> [n_feat] vector, feature_names).

    Written by ``scripts/extract_bpe_features.py``. Asserted disjoint from FORBIDDEN_FEATURES — BPE is
    imaging-derived microenvironment characterisation, never a leaking receptor/subtype field.
    """
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
    """Assemble the dev-cohort feature matrices + label, restricted to the frozen split's dev set.

    Returns (X_num [n,K] float w/ NaN, X_cat [n,5] int codes, y, groups=patient_id, cat_cardinalities).

    BP1: when ``bpe_npz`` is given, the contralateral-BPE features are appended as extra numeric
    columns (NaN for any patient without a BPE vector). The per-fold median-impute+standardize in
    ``_fit_eval_fold`` handles the NaNs and raw scale leakage-safely. ``bpe_npz=None`` (default) keeps
    X_num byte-identical to the clinical-only baseline — the ablation IS this flag.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
    from audit_ki67 import load as load_clinical  # 3-row-header resolver

    _assert_leak_free()
    df = load_clinical(clin_path)
    df.columns = [str(c) for c in df.columns]
    # Age-at-diagnosis is not a stored column — derive it (diagnosis = day 0, DOB negative days).
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
        # TICKET-022: reserve category code 0 as "unknown/OOV". pd.factorize assigns observed levels
        # 0..K-1 (global on the dev cohort → within-dev CV stays correct); we shift +1 so observed
        # levels occupy 1..K and code 0 is free for a category unseen at inference (holdout/external).
        # cards[j] = observed + 1 so the embedding table has a row for that reserved OOV slot. This
        # makes the published encoder safe to apply to patients with an unseen category (no index-
        # out-of-range) without changing which rows the dev-cohort model trains on.
        codes, uniq = pd.factorize(pd.to_numeric(_col(df, c), errors="coerce"), use_na_sentinel=False)
        cat_codes.append(codes + 1)  # observed levels -> 1..K; 0 reserved for OOV
        cards.append(len(uniq) + 1)  # +1 for the reserved OOV slot at code 0
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
    """One train/eval fold: train-only impute+standardize, train FTT, return val AUROC.

    TICKET-010: when ``return_state=True`` also returns the trained model's state bundle (state_dict
    + the train-fold impute/standardize stats + cards) so the caller can PUBLISH exactly the fit that
    produced the OOF prediction — no divergent re-train. The default (``return_state=False``) keeps the
    3-tuple return so `cross_val_auroc` and every existing caller are byte-for-byte unchanged. The
    returned state is a plain CPU-tensor dict (torch imported lazily inside)."""
    import torch
    from sklearn.metrics import roc_auc_score
    from rtdl_revisiting_models import FTTransformer

    from pinksight.seed import set_seed

    set_seed(seed)
    # numeric: median-impute + standardize, fit on TRAIN only (LOCK-2).
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
    for _ in range(epochs):  # full-batch GD — 624×8 is trivial
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
        # The published checkpoint IS this fit (same seed→construct order, same fixed epochs) — so
        # re-scoring these weights reproduces the reported OOF exactly (TICKET-010, no re-train).
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
    return fold_auc, np.asarray(te), prob  # fold AUROC + OOF preds (for pooled DeLong/ECE)


def _fit_fold_embed(x_num, x_cat, y, tr, te, cards, seed, epochs=80):
    """FIX-1: same fixed-epoch train-only-standardized fold fit as `_fit_eval_fold`, but ALSO returns
    the FT-Transformer PENULTIMATE embedding on the held-out test fold (the [CLS] representation
    before the final output Linear — the frozen clinical embedding the fusion arm consumes).

    Returns (te_idx, test_probs, test_emb[n_te, d_block], standardized_train_X[n_tr, n_num+n_cat]).
    The train-fold standardized matrix is returned so the caller can assemble SHAP `background_X`.
    Integrity: identical numeric impute+standardize-on-TRAIN-only and identical fixed-epoch loop as
    the scored path, so the embedding corresponds to the same OOF prediction (no test-fold peeking —
    epochs are fixed, there is no model selection against the test fold)."""
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
        # Penultimate embedding: swap the backbone's final output Linear for Identity, run forward,
        # then restore. This yields the [CLS]-token representation (d_block-dim) with no edit to rtdl.
        saved_output = model.backbone.output
        model.backbone.output = torch.nn.Identity()
        try:
            emb = model(xc_t[te_t], xq_t[te_t]).cpu().numpy()
        finally:
            model.backbone.output = saved_output
    # background_X source: the standardized TRAIN-fold inputs (numeric ⊕ categorical codes), the
    # feature space SHAP KernelExplainer perturbs. Standardized numeric matches the model's input.
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
    """FIX-1 (flag-gated; call site opts in). Export FROZEN OUT-OF-FOLD clinical FT-Transformer
    embeddings for the fusion arm + the SHAP background/feature-names the XAI arm needs.

    Per seed, runs the SAME patient-grouped StratifiedGroupKFold(5) CV as `cross_val_auroc` and, for
    each held-out fold, captures the penultimate [CLS] embedding of the model that never trained on
    that patient. Writes, per seed:
        out_dir/clinical_embed_s{SEED}.npz   pids:[str], emb:[N, d_block]   (fusion input, OUT-OF-FOLD)
    and once (seed-independent, LOCK-2 leakage-safe by construction):
        out_dir/clinical_background.npz      background_X:[K, n_feat], feature_names:[str]  (SHAP)

    `feature_names` is FEATURES (numeric ⊕ categorical), asserted disjoint from FORBIDDEN before write
    (belt-and-suspenders over `_assert_leak_free`). `background_X` is a `background_n`-row sample of
    the standardized dev-cohort feature matrix — the space KernelExplainer perturbs. Returns a small
    provenance dict (files written, dims, n)."""
    from sklearn.model_selection import StratifiedGroupKFold

    _assert_leak_free()  # never export a background whose columns include a label-defining field
    feature_names = list(FEATURES)
    leaked = set(feature_names) & set(FORBIDDEN_FEATURES)
    if leaked:
        raise LeakageError(f"forbidden fields in exported clinical feature_names: {sorted(leaked)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    y = list(y)
    n = len(y)
    d_block = None
    bg_pool: list[np.ndarray] = []  # standardized train-fold rows, pooled for a background sample
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
        pids_ordered = [str(groups[j]) for j in order]  # patient == group in this arm
        p = out_dir / f"clinical_embed_s{seed}.npz"
        np.savez(p, pids=np.array(pids_ordered, dtype=object), emb=emb_mat)
        written.append(str(p))

    # SHAP background: sample the pooled standardized train rows (seed-0 folds) to `background_n`.
    bg_all = np.concatenate(bg_pool) if bg_pool else np.zeros((0, len(feature_names)))
    rng = np.random.default_rng(0)
    k = min(background_n, bg_all.shape[0])
    bg_sample = bg_all[rng.choice(bg_all.shape[0], size=k, replace=False)] if k > 0 else bg_all
    # TICKET-016: write the TRUE cat_cardinalities (`cards`) into the background npz. XAI's loader
    # PREFERS this `cards` array over inferring max(code)+1 from the ≤100-row background sample (which
    # can UNDERSHOOT the checkpoint's embedding-table size when a rare level is absent from the sample,
    # silently loading a mis-sized model). This is the authoritative cardinality the encoder trained on
    # (already includes the reserved OOV slot from load_xy).
    bg_p = out_dir / "clinical_background.npz"
    np.savez(bg_p, background_X=bg_sample.astype(float),
             feature_names=np.array(feature_names, dtype=object),
             cards=np.array(list(cards), dtype=int))
    written.append(str(bg_p))
    return {"files": written, "d_block": d_block, "n_dev": n,
            "background_n": int(bg_sample.shape[0]), "feature_names": feature_names,
            "cards": [int(c) for c in cards]}


def fit_on_full_dev(x_num, x_cat, y, cards, seed, epochs=80):
    """G5 leg-1: train ONE clinical FT-Transformer on the ENTIRE dev cohort (all 624 patients).

    The per-fold `_fit_eval_fold` produces out-of-fold predictions for the internal CV number; it
    does NOT give a single model to apply to an external cohort. External validation needs exactly
    one model trained on ALL of Duke-dev, then scored ONCE on ISPY2. This function is that fit.

    The recipe is byte-identical to `_fit_eval_fold` — same seed→construct order (`set_seed` → build
    FTTransformer(n_blocks=2) → AdamW(lr=1e-3, wd=1e-4) → fixed 80-epoch full-batch BCE-with-pos_weight
    loop), same numeric median-impute+standardize (fit here on the FULL dev cohort, which is the whole
    training set for this model), same categorical cards. The ONLY difference from a CV fold is the
    train index = every dev row (no held-out fold), because for external eval the dev cohort is the
    training set in full. So "the published checkpoint IS this fit" (TICKET-010 semantics) carries
    over: re-running this with the same seed reproduces the exact external prediction.

    Returns a state bundle: {state_dict, mu, sd, med, cards, seed, epochs} — the CPU-tensor weights
    plus the TRAIN-only (== full-dev) impute/standardize stats needed to transform an external table
    the SAME way before inference. LOCK-2: mu/sd/med are fit on dev ONLY and are the ONLY stats that
    may touch the external table (never fit anything on ISPY2)."""
    import torch
    from rtdl_revisiting_models import FTTransformer

    from pinksight.seed import set_seed

    set_seed(seed)
    # numeric: median-impute + standardize, fit on the FULL dev cohort (the training set here) — LOCK-2.
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
    for _ in range(epochs):  # full-batch GD over all 624 dev rows — trivial size
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
    """Apply a `fit_on_full_dev` state bundle to an external feature matrix → TNBC probabilities.

    LOCK-2 CRITICAL: the external numeric matrix is imputed with the state's `med` and standardized
    with the state's `mu`/`sd` — the DUKE-TRAIN statistics carried in the bundle. NO statistic is
    (re)computed from the external rows here. Categorical codes must already be in the Duke code space
    (see g5_external_eval's parity crosswalk); `state["cards"]` is the embedding-table size, and any
    external code >= its card is clamped to the reserved OOV slot (0) so an unseen level cannot index
    out of range. Returns a 1-D array of P(TNBC) aligned to the external rows."""
    import torch
    from rtdl_revisiting_models import FTTransformer

    med = np.asarray(state["med"], float)
    mu = np.asarray(state["mu"], float)
    sd = np.asarray(state["sd"], float)
    cards = list(state["cards"])

    xn = np.where(np.isnan(x_num_ext), med, x_num_ext)  # Duke-train median impute (LOCK-2)
    xn = (xn - mu) / sd  # Duke-train standardize (LOCK-2)

    xc = np.asarray(x_cat_ext, dtype=int).copy()
    for j, card in enumerate(cards):
        # any external code outside the trained embedding table -> reserved OOV slot 0 (never OOB).
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


# =====================================================================================================
# LogReg estimator path (G5 re-run 25-07-26) — the H6 COALITION estimator, so the external drop is
# measured LogReg-internal -> LogReg-external against the 0.708/0.719 clinical-subtype ANCHOR (the
# ablation-ladder / H6 modality-audit estimator), NOT the FT-Transformer. The prior G5 leg used the FTT
# (cross_val_auroc, internal ~0.634); that sits under a LogReg headline -> apples-to-oranges. These
# functions reproduce fva.shuffle_sentinel.coalition_oof's estimator EXACTLY:
#   raw LOCK-3 features (incl. Nottingham grade) -> OneHotEncoder(handle_unknown="ignore") on the
#   categorical codes -> hstack([x_num, one-hot]) -> SimpleImputer(median) -> StandardScaler ->
#   LogisticRegression(C=1.0, max_iter=1000).
# The FTT path above (fit_on_full_dev / predict_with_state / cross_val_auroc) is left byte-for-byte
# intact for backward compatibility; nothing else in the repo changes estimator.


def _logreg_oh_encoder(x_cat_dev: np.ndarray):
    """Fit the OneHotEncoder on the FULL Duke-dev categorical codes (handle_unknown='ignore').

    Matches h6_modality_audit.load_raw_clinical EXACTLY: `OneHotEncoder(handle_unknown="ignore",
    sparse_output=False).fit(x_cat)`. Fitting on dev means every external categorical level UNSEEN in
    Duke-dev becomes an all-zero one-hot row (handle_unknown='ignore') — the LogReg analogue of the FTT
    OOV-slot clamp, and the ONLY leakage-safe encoding (the encoder never sees ISPY2). LOCK-2."""
    from sklearn.preprocessing import OneHotEncoder

    return OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(x_cat_dev)


def fit_logreg_on_full_dev(x_num, x_cat, y, cards, seed):
    """G5 re-run: fit ONE H6-estimator LogReg on the ENTIRE Duke-dev cohort, for external application.

    The estimator is byte-identical to fva.shuffle_sentinel.coalition_oof (the pre-reg's LOCKED
    coalition estimator that produces the 0.708/0.719 clinical anchor): one-hot the categorical codes,
    hstack with the raw numeric columns, median-impute + standardize (fit on FULL dev here — the whole
    training set for external eval), then LogisticRegression(C=1.0, max_iter=1000).

    LOCK-2: the OneHotEncoder, the SimpleImputer(median), and the StandardScaler are ALL fit on Duke-dev
    ONLY — they are the ONLY transforms that may touch the external table (never fit anything on ISPY2).
    `seed` is accepted for signature parity with the FTT path and to seed LogReg's solver determinism;
    LogReg on this problem is effectively deterministic across seeds (liblinear/lbfgs converge), so the
    3-seed spread collapses to a near-constant — reported honestly rather than hidden.

    Returns a state bundle: {oh, imputer, scaler, clf, cards, seed} — the fitted sklearn objects needed
    to transform an external table the SAME way before inference."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    _assert_leak_free()
    oh = _logreg_oh_encoder(np.asarray(x_cat, dtype=int))
    X = np.hstack([np.asarray(x_num, float), oh.transform(np.asarray(x_cat, dtype=int))])
    imputer = SimpleImputer(strategy="median").fit(X)  # Duke-dev median (LOCK-2)
    Xi = imputer.transform(X)
    scaler = StandardScaler().fit(Xi)  # Duke-dev mean/std (LOCK-2)
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
    """Apply a `fit_logreg_on_full_dev` state bundle to an external feature matrix -> TNBC probs.

    LOCK-2 CRITICAL: the external one-hot / impute / standardize all use the Duke-dev-FITTED sklearn
    objects carried in the bundle. NO statistic is (re)computed from the external rows here. Unseen
    external categorical levels are dropped to an all-zero one-hot row by the encoder's
    handle_unknown='ignore' (the LogReg analogue of the FTT OOV clamp). Returns P(TNBC) aligned to the
    external rows."""
    oh, imputer, scaler, clf = state["oh"], state["imputer"], state["scaler"], state["clf"]
    X = np.hstack([np.asarray(x_num_ext, float), oh.transform(np.asarray(x_cat_ext, dtype=int))])
    Xs = scaler.transform(imputer.transform(X))
    pos = list(clf.classes_).index(1)
    return clf.predict_proba(Xs)[:, pos]


def logreg_oof(x_num, x_cat, y, groups, cards, seed):
    """One-seed pooled-OOF P(TNBC) for the H6-estimator LogReg on Duke-dev, patient-grouped 5-fold.

    Mirrors fva.shuffle_sentinel.coalition_oof for the single clinical stream (raw_clinical_lock3):
    per fold, fit OneHot(dev-train-fold) + SimpleImputer(median) + StandardScaler + LogReg on the TRAIN
    fold only, predict the held-out fold. Each dev patient scored by a model that never trained on them
    -> the internal number that Delta compares to the pooled external AUROC (estimator-consistent)."""
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
        # one-hot fit on TRAIN fold (handle_unknown='ignore' covers a level unseen in this fold).
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
    """Multi-seed pooled-OOF AUROC (+ mean DeLong CI, ECE) for the H6-estimator LogReg on Duke-dev.

    The LogReg counterpart of cross_val_auroc. Returns pooled-OOF AUROC per seed and the seed mean,
    plus a mean DeLong CI and mean ECE. This is the LogReg INTERNAL number on FULL Duke-dev (the same
    cohort the external re-fit trains on) — the estimator-consistent Delta comparator. It is close to
    but NOT identical to the H6 anchor 0.719, which is on the N~613 3-way imaging intersection; the N
    difference is reported by the caller."""
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
    """G5 leg-1 / E1: persist the DUKE-TRAIN imputation+encoding statistics artifact.

    The G5 external eval must impute missing external features with Duke-TRAIN stats ONLY (LOCK-2 —
    never fit a statistic on ISPY2). This writes that artifact once, from the full dev cohort, so the
    external script LOADS it rather than deriving anything from the external table. Contents:
        med  : per-numeric-feature median (dev cohort)          -> impute missing external numerics
        mu,sd: per-numeric-feature mean/std (dev cohort)        -> standardize external numerics
        cards: categorical embedding-table sizes (incl. OOV)    -> clamp external codes safely
        feature_names_num / feature_names_cat : provenance
    Numeric stats are the full-dev equivalents of `_fit_eval_fold`'s train-fold stats. Written as a
    pickle at `out_path`. Returns the stats dict (also usable in-process)."""
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
    """Patient-grouped, class-stratified CV AUROC across SEEDS. Returns a metrics dict."""
    from sklearn.model_selection import StratifiedGroupKFold

    y = list(y)
    yv = np.asarray(y)
    per_seed = {}  # fold-mean AUROC (original P05 definition — keeps the 0.626 comparable)
    pooled, ci, ece_seed = {}, {}, {}  # per-seed pooled-OOF AUROC, DeLong CI, calibration
    for seed in SEEDS:
        cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        fold_aucs = []
        oof = np.full(len(y), np.nan)  # each patient-row lands in exactly one test fold this seed
        for tr, te in cv.split(x_num, y, groups):
            if not set(np.asarray(groups)[tr]).isdisjoint(np.asarray(groups)[te]):
                raise LeakageError("patient leaked across a CV fold — LOCK-2 violation")
            fold_auc, te_idx, probs = _fit_eval_fold(x_num, x_cat, y, tr, te, cards, seed)
            fold_aucs.append(fold_auc)
            oof[te_idx] = probs
        if np.isnan(oof).any():
            raise RuntimeError("OOF preds incomplete — a patient-row was never in a test fold")
        per_seed[seed] = float(np.mean(fold_aucs))
        auc, lo, hi = delong_ci(yv, oof)  # CI per seed (preds independent within a seed)
        pooled[seed], ci[seed] = auc, (lo, hi)
        ece_seed[seed] = ece(yv, oof)
    means = np.array(list(per_seed.values()))
    pooled_v = np.array(list(pooled.values()))
    return {
        "auroc_mean": float(means.mean()),  # mean of per-fold AUROC across seeds (P05 original)
        "auroc_std_across_seeds": float(means.std()),
        "auroc_min": float(means.min()),
        "auroc_max": float(means.max()),
        "per_seed_mean_auroc": {str(k): round(v, 4) for k, v in per_seed.items()},
        "auroc_pooled_oof_mean": float(pooled_v.mean()),  # AUROC on pooled OOF preds (DeLong base)
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
            "DeLong 95% CI + ECE (CLAUDE.md eval rule)."
        ),
        gate="G2",
    )
    exp = _next_exp_dir(reports)
    (exp / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return {"exp_dir": str(exp), **metrics}


def selfcheck() -> int:
    """No data, no torch: the leak guard is the load-bearing invariant — assert it bites."""
    _assert_leak_free()  # raises LeakageError if FEATURES ∩ FORBIDDEN ≠ ∅
    assert len(FEATURES) == len(set(FEATURES)) == 9, "feature set drifted"
    print("selfcheck OK — FEATURES (9) disjoint from FORBIDDEN; leak guard armed")  # noqa: T201
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
        print(f"Not found: {args.clinical} (data is git-ignored / not downloaded).")  # noqa: T201
        return 2
    out = run(args.clinical, args.split, args.reports)
    print(json.dumps(out, indent=2))  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
