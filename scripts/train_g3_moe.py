"""G3 #7 — Biology-gated MoE on top of the #4 hierarchical fused representation (patient-level CV).

Deterministic biology gating (NOT a learned router — banned at G3 scale). Rides on the #4
HierarchicalStagedFusion: per fold, fit a hierarchical model on the frozen embeddings, extract its
`fused` stage-3 representation, then train the BiologyGatedMoE head on those reps with an INTEGER
expert-routing index. Reports the H-G3-B comparison inputs + the mandatory train/val gap guard +
per-expert subtype class balance (leakage/collapse inspection).

Two modes:
  --smoke-only : 1 seed, few folds/epochs, $0 CPU validity check (no gate threshold). Step 7.4.
  (default)    : full multi-seed run (Step 7.5, GPU) — H-G3-B gate number.

Routing methods (--strata-method), both LOCK-2 safe (integer index, NEVER a feature):
  grade_band : Option B (plan §Component #7) — Nottingham grade routes experts (NHG1 -> Expert 0,
               NHG3 -> Expert 1; NHG2/missing routed TARGET-BLIND by a stable pid-hash parity split).
               Grade is read from the Duke clinical xlsx as a LABEL only. This is the CORRECT full-run
               method: HR-status routing LEAKS in this strict LumA-vs-TNBC cohort (ER/PR is near-
               collinear with the subtype target, so hard-gated HR routing launders the label ->
               spurious AUROC 1.0). Grade only partially correlates with subtype, so the leakage_flag
               (any expert sees >95% one class) must still be checked before reporting any lift.
  hr_status  : $0-smoke placeholder ONLY — a target-blind median split of MRI-embedding dim 0. Kept
               for the Step 7.4 architecture smoke; NOT a valid full-run routing (see grade_band).

BiologyGatedMoE.forward asserts strata_labels is integer-typed so a float label can never be used as
a routing signal. Claim-ledger: subtype characterisation at diagnosis; no growth-rate / kinetics.

    # $0 architecture smoke (target-blind):
    PYTHONPATH=src .venv/bin/python scripts/train_g3_moe.py --smoke-only \
        --seed 0 --n-folds 2 --n-epochs 5 \
        --hierarchical-oof reports/G3_fusion_arch_bundle/smoke_hierarchical_s0.json \
        --strata-method hr_status --split configs/split_v2.yaml \
        --out reports/G3_fusion_arch_bundle/smoke_moe_s0.json

    # Full run (Step 7.5) — grade-band routing (Option B), NEVER hr_status:
    PYTHONPATH=src .venv/bin/python scripts/train_g3_moe.py --seeds 0 1 2 --n-folds 5 --n-epochs 30 \
        --hierarchical-oof reports/G3_fusion_arch_bundle/hierarchical_oof.json \
        --strata-method grade_band --split configs/split_v2.yaml \
        --out reports/G3_fusion_arch_bundle/moe_gradeband_oof.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from pinksight.metrics import delong_ci
from pinksight.models.fusion import BiologyGatedMoE, HierarchicalStagedFusion

# Reuse the #4 data alignment so both components share ONE frozen pid set + label source.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_g3_hierarchical import build_aligned  # noqa: E402  (scripts/ on sys.path)

ROOT = Path(__file__).resolve().parents[1]
DUKE_CLINICAL_XLSX = ROOT / "data/raw/Clinical_and_Other_Features.xlsx"
# Nottingham grade column in the Duke clinical xlsx (3-row header): 1=low 2=intermediate 3=high.
_NHG_COL = ("Tumor Characteristics", "Nottingham grade", "1=low 2=intermediate 3=high\n")


def make_gap_entry(train_auroc_per_fold, val_auroc_per_fold) -> dict:
    """Execute-agent instruction E-4: train_val_gap = mean(train_auroc) - mean(val_auroc).

    Single source of truth for the gap formula (imported by tests/test_g3_moe.py). Clamped to [0,1]
    for the overfit guard (a negative raw gap means val >= train -> no overfit -> report 0.0)."""
    gap = float(np.mean(train_auroc_per_fold) - np.mean(val_auroc_per_fold))
    return {"value": max(0.0, min(1.0, gap)),
            "raw": round(gap, 4),
            "mean_train_auroc": round(float(np.mean(train_auroc_per_fold)), 4),
            "mean_val_auroc": round(float(np.mean(val_auroc_per_fold)), 4),
            "note": "E-4: mean(train_auroc)-mean(val_auroc); gate FAILS if > 0.15 (expert overfit)."}


def _smoke_strata_target_blind(feats: dict) -> np.ndarray:
    """TARGET-BLIND integer routing strata for the $0 smoke (a placeholder for real HR-status).

    IMPORTANT (leakage): the routing index must NOT be the subtype target itself. In this strict
    LumA-vs-TNBC cohort, HR-status (ER/PR) is near-collinear with subtype ([1.2-R] flag), so routing
    on subtype (or a near-identical HR proxy) would LAUNDER the label through the hard gate and give a
    spuriously perfect OOF AUROC — that is target leakage, not a working model. For the $0 architecture
    smoke we therefore route on a fixed, target-blind partition: the median split of MRI-embedding
    dimension 0. This exercises the exact MoE code path (gate_min clamp, 2 experts, soft mixing) and
    the train/val-gap contract WITHOUT any target information reaching the router. It is an integer
    index, never a feature (LOCK-2).

    The FULL run (Step 7.5, GPU — out of this $0 leg's scope) derives HR-status from the ER/PR LABEL
    columns of the clinical xlsx as integer routing indices (still never features). The subtype-lift
    verdict (H-G3-B) is only meaningful there, not at this smoke scale."""
    d0 = feats["mri"][:, 0]
    return (d0 >= np.median(d0)).astype(np.int64)


def _grade_labels() -> pd.Series:
    """Patient-level Nottingham grade (NHG 1/2/3) from the Duke clinical xlsx (a LABEL, never a feature).

    The grade column is only ever read to derive an INTEGER expert-routing index for the MoE gate
    (Option B, plan §Component #7). It is NEVER passed to a classifier head, so it does not touch the
    FORBIDDEN_FEATURES leakage surface. Returns a Series indexed by patient_id with float NHG in
    {1.0, 2.0, 3.0} or NaN when grade is unrecorded."""
    df = pd.read_excel(DUKE_CLINICAL_XLSX, header=[0, 1, 2])
    pid = df.iloc[:, 0].astype(str)
    grade = pd.to_numeric(df[_NHG_COL], errors="coerce")
    return pd.Series(grade.to_numpy(), index=pid.to_numpy())


def _grade_band_strata(groups: np.ndarray) -> np.ndarray:
    """Option-B grade-band routing (plan §Component #7): NHG1 -> Expert 0, NHG3 -> Expert 1.

    NHG2 (intermediate) and any patient with unrecorded grade are routed TARGET-BLIND — a fixed
    parity split on a stable hash of the patient_id — so no target information reaches the router
    (grade partially correlates with subtype; a target-derived NHG2 split would launder the label
    through the hard gate exactly as HR-status did, producing a spurious AUROC). The returned array
    is an integer expert index (dtype=int64), NEVER a feature: BiologyGatedMoE.forward asserts the
    strata are integer-typed (LOCK-2). NHG grade is read only from the clinical xlsx as a label.

    Determinism (reproducibility hygiene, 12-08-26 md5 fix): the NHG2/missing parity split MUST use
    an md5 digest of the patient_id, NOT Python's builtin ``hash()``. CPython salts string ``hash()``
    per-process via ``PYTHONHASHSEED`` (fixed once at interpreter start, never logged at the frozen
    run), so the routing — and hence the reported AUROC — silently varied run-to-run. A 20-salt sweep
    (moe7_reproducibility_investigation_12-08-26.md) measured that hidden variance at AUROC
    0.650 ± 0.018 [0.615, 0.689], all 20/20 draws below the 0.708 clinical anchor. md5 is process-
    stable (independent of PYTHONHASHSEED), so the split — target-blind exactly as before (md5 of an
    id carries no subtype information) — is now bit-identical on every re-run. This is a pure
    reproducibility hygiene change: it does not re-freeze a new headline point (the reported #7
    magnitude remains the disclosed salt-distribution), it only stops future re-runs drawing a fresh
    unrecorded salt."""
    grade = _grade_labels()
    strata = np.empty(len(groups), dtype=np.int64)
    for i, pid in enumerate(groups):
        g = grade.get(str(pid), np.nan)
        if g == 1.0:
            strata[i] = 0            # NHG1 -> Expert 0
        elif g == 3.0:
            strata[i] = 1            # NHG3 -> Expert 1
        else:
            # NHG2 or missing: target-blind parity split on a DETERMINISTIC md5 digest of the
            # patient id (NOT builtin hash() — that is PYTHONHASHSEED-salted and non-reproducible).
            # md5 is target-blind (no subtype info in an id hash) and process-stable (12-08-26 fix).
            strata[i] = int(hashlib.md5(str(pid).encode("utf-8")).hexdigest(), 16) & 1
    return strata


def _strata_for(method: str, feats: dict, groups: np.ndarray) -> np.ndarray:
    """Dispatch integer routing strata by method (both are LOCK-2 safe integer indices, never features).

    - grade_band: NHG-grade routing (Option B) — grade is a LABEL, NHG2/missing routed target-blind.
    - hr_status : $0-smoke placeholder — target-blind median split of MRI dim 0 (the real ER/PR
      HR-status derivation stays out of scope until the clinical xlsx ER/PR columns are wired)."""
    if method == "grade_band":
        return _grade_band_strata(groups)
    return _smoke_strata_target_blind(feats)


def _fused_reps(model: HierarchicalStagedFusion, feats: dict) -> np.ndarray:
    model.eval()
    xt = {k: torch.from_numpy(v).float() for k, v in feats.items()}
    with torch.no_grad():
        return model(xt, drop=set())["fused"].cpu().numpy()


def _fit_hierarchical(feats_tr, y_tr, seed, n_epochs, lr=1e-3) -> HierarchicalStagedFusion:
    torch.manual_seed(seed)
    model = HierarchicalStagedFusion(
        {"mri": feats_tr["mri"].shape[1], "clinical": feats_tr["clinical"].shape[1]}, fused_dim=128,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    xt = {k: torch.from_numpy(v).float() for k, v in feats_tr.items()}
    yt = torch.from_numpy(y_tr).float()
    model.train()
    for _ in range(n_epochs):
        opt.zero_grad()
        out = model(xt)
        total, _ = model.joint_loss(out, yt)
        if not torch.isfinite(total):
            raise SystemExit("NaN/inf loss in hierarchical backbone — STOP")
        total.backward()
        opt.step()
    return model


def _fit_moe(rep_tr, y_tr, strata_tr, seed, n_epochs, lr=1e-3) -> BiologyGatedMoE:
    torch.manual_seed(seed + 777)
    moe = BiologyGatedMoE(fused_dim=rep_tr.shape[1], n_experts=2, strata_init_method="hr_status")
    opt = torch.optim.AdamW(moe.parameters(), lr=lr, weight_decay=1e-4)
    rt = torch.from_numpy(rep_tr).float()
    st = torch.from_numpy(strata_tr).long()
    yt = torch.from_numpy(y_tr).float()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    moe.train()
    for _ in range(n_epochs):
        opt.zero_grad()
        out = moe(rt, st)
        loss = loss_fn(out["subtype_logit"].reshape(-1), yt)
        if not torch.isfinite(loss):
            raise SystemExit("NaN/inf loss in MoE head — STOP")
        loss.backward()
        opt.step()
    return moe


def _moe_prob(moe: BiologyGatedMoE, rep: np.ndarray, strata: np.ndarray) -> np.ndarray:
    moe.eval()
    with torch.no_grad():
        logit = moe(torch.from_numpy(rep).float(), torch.from_numpy(strata).long())["subtype_logit"]
    return torch.sigmoid(logit.reshape(-1)).cpu().numpy()


def run_seed(feats, y, groups, seed, n_folds, n_epochs, strata_method="hr_status"):
    """Pooled-OOF MoE probs + per-fold train/val AUROC for the gap guard.

    strata_method selects the integer expert-routing source (both LOCK-2 safe, never features):
    grade_band routes on Nottingham grade (Option B); hr_status uses the target-blind smoke split."""
    oof = np.full(len(y), np.nan, dtype=float)
    train_aucs, val_aucs = [], []
    # Full-cohort integer routing strata (NHG grade for grade_band; target-blind for hr_status).
    strata_all = _strata_for(strata_method, feats, groups)
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr, va in sgkf.split(feats["mri"], y, groups):
        assert not (set(groups[tr]) & set(groups[va])), "patient overlap across folds — STOP"
        feats_tr = {k: v[tr] for k, v in feats.items()}
        feats_va = {k: v[va] for k, v in feats.items()}
        hier = _fit_hierarchical(feats_tr, y[tr], seed, n_epochs)
        rep_tr, rep_va = _fused_reps(hier, feats_tr), _fused_reps(hier, feats_va)
        # Integer expert-routing index (grade label OR target-blind split) — NEVER a feature (LOCK-2).
        st_tr, st_va = strata_all[tr], strata_all[va]
        moe = _fit_moe(rep_tr, y[tr], st_tr, seed, n_epochs)
        p_tr = _moe_prob(moe, rep_tr, st_tr)
        p_va = _moe_prob(moe, rep_va, st_va)
        oof[va] = p_va
        train_aucs.append(float(roc_auc_score(y[tr], p_tr)) if len(np.unique(y[tr])) > 1 else 0.5)
        val_aucs.append(float(roc_auc_score(y[va], p_va)) if len(np.unique(y[va])) > 1 else 0.5)
    assert not np.isnan(oof).any(), "un-scored patient in MoE OOF — STOP"
    return oof, train_aucs, val_aucs


def _expert_class_balance(strata: np.ndarray, y: np.ndarray, n_experts: int = 2) -> dict:
    """Per-expert routed-patient count + subtype class balance (leakage/collapse inspection).

    The leakage watch (plan #7): if any expert routes >95% of ITS patients to one subtype class, the
    routing has laundered the target and any 'lift' is spurious. Also flags expert collapse (>95% of
    ALL patients funnelled to one expert)."""
    n = len(y)
    out = {}
    max_share = 0.0
    for e in range(n_experts):
        mask = strata == e
        cnt = int(mask.sum())
        share = cnt / n if n else 0.0
        max_share = max(max_share, share)
        if cnt:
            pos = float(np.mean(y[mask]))          # fraction TNBC (class 1) among this expert's patients
            purity = max(pos, 1.0 - pos)           # dominant-class share within the expert
        else:
            pos, purity = float("nan"), float("nan")
        out[f"expert_{e}"] = {"n": cnt, "share": round(share, 4),
                              "tnbc_frac": round(pos, 4) if cnt else None,
                              "class_purity": round(purity, 4) if cnt else None}
    purities = [v["class_purity"] for v in out.values() if v["class_purity"] is not None]
    out["leakage_flag"] = bool(purities and max(purities) > 0.95)     # expert sees ~one class -> laundered
    out["collapse_flag"] = bool(max_share > 0.95)                     # ~all patients on one expert
    return out


def run(seeds, embeddings_dir, split_yaml, n_folds, n_epochs, out_path, smoke_only,
        strata_method="hr_status", save_oof_dir=None):
    per_seed_auc, oof_by_seed, y_by_seed = {}, {}, {}
    all_train, all_val = [], []
    balance_by_seed = {}
    if save_oof_dir is not None:
        save_oof_dir = Path(save_oof_dir)
        save_oof_dir.mkdir(parents=True, exist_ok=True)
    for s in seeds:
        feats, y, groups = build_aligned(embeddings_dir, split_yaml, s)
        oof, tr_aucs, va_aucs = run_seed(feats, y, groups, s, n_folds, n_epochs, strata_method)
        per_seed_auc[s] = float(roc_auc_score(y, oof))
        oof_by_seed[s] = oof
        y_by_seed[s] = y
        all_train.extend(tr_aucs)
        all_val.extend(va_aucs)
        balance_by_seed[str(s)] = _expert_class_balance(
            _strata_for(strata_method, feats, groups), y)
        # Item 2 (additive): persist per-sample OOF probs + patient IDs for the downstream paired
        # DeLong vs the clinical anchor (item 3). Guarded by --save-oof-dir; when the flag is
        # omitted this block is skipped and behavior is byte-identical to the frozen version.
        # pids/y/oof are index-aligned (groups order from build_aligned == oof/y order).
        if save_oof_dir is not None:
            np.savez(save_oof_dir / f"moe_oof_s{s}.npz", pids=groups, y=y, oof=oof)

    mean_auc = float(np.mean(list(per_seed_auc.values())))
    auc0, lo, hi = delong_ci(y_by_seed[seeds[0]], oof_by_seed[seeds[0]])
    gap_entry = make_gap_entry(all_train, all_val)
    any_leakage = any(b["leakage_flag"] for b in balance_by_seed.values())
    any_collapse = any(b["collapse_flag"] for b in balance_by_seed.values())
    doc = {
        "gate": f"G3 #7 biology-gated MoE ({strata_method}) (H-G3-B)",
        "smoke_only": smoke_only,
        "strata_method": strata_method,
        "subtype": {
            "auroc": {"value": round(mean_auc, 4), "ci95": [round(lo, 4), round(hi, 4)],
                      "ci_method": "delong", "ci_seed": int(seeds[0])},
            "per_seed_auroc": {str(k): round(v, 4) for k, v in per_seed_auc.items()},
            "train_val_gap": gap_entry,
        },
        "expert_class_balance": balance_by_seed,
        "leakage_flag": any_leakage,
        "collapse_flag": any_collapse,
        "n": int(len(y_by_seed[seeds[0]])),
        "seeds": [int(s) for s in seeds],
        "n_folds": n_folds,
        "n_epochs": n_epochs,
        "note": ("$0 CPU smoke — MoE architecture validity + train/val gap KEY presence only, NOT a "
                 "gate number (statistical validity of the gap is only meaningful at full-run scale). "
                 "Routing strata are TARGET-BLIND (median split of MRI dim 0) to avoid laundering the "
                 "near-collinear subtype label through the hard gate; an integer index, never a feature "
                 "(LOCK-2). Full-run HR-status routing from ER/PR label columns is Step 7.5 (GPU) scope."
                 if smoke_only else
                 f"H-G3-B gate ({strata_method} routing): MoE subtype AUROC vs #4 hierarchical "
                 "single-head + train/val gap < 0.15. Routing index is an integer LABEL "
                 "(NHG grade for grade_band; NHG2/missing routed target-blind), never a feature "
                 "(LOCK-2). leakage_flag=True means an expert saw ~one subtype class -> spurious; "
                 "collapse_flag=True means ~all patients on one expert -> DEGENERATE."),
        "claim_ledger": "subtype characterisation at diagnosis; ER/PR/HER2/Ki-67 never a feature (LOCK-2)",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2) + "\n")
    flags = f" LEAKAGE_FLAG={any_leakage} COLLAPSE_FLAG={any_collapse}"
    print(f"[G3 #7] MoE ({strata_method}) subtype AUROC {mean_auc:.4f} [{lo:.4f},{hi:.4f}]  "
          f"train/val gap {gap_entry['value']:.4f} (raw {gap_entry['raw']})  N={doc['n']}{flags}")
    print(f"wrote {out_path}")
    return doc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke-only", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--n-epochs", type=int, default=30)
    ap.add_argument("--hierarchical-oof", type=Path, default=None,
                    help="path to the #4 hierarchical OOF JSON (recorded for provenance)")
    ap.add_argument("--strata-method", default="hr_status", choices=["hr_status", "grade_band"])
    ap.add_argument("--embeddings-dir", type=Path, default=ROOT / "reports/G2_imaging/embeddings")
    ap.add_argument("--split", type=Path, default=ROOT / "configs/split_v2.yaml")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--save-oof-dir", type=Path, default=None,
                    help="if set, write per-seed moe_oof_s{seed}.npz (pids/y/oof) for the downstream "
                         "paired DeLong vs the clinical anchor (item 2; additive — has no effect on "
                         "any existing output or on behavior when the flag is omitted)")
    args = ap.parse_args()

    seeds = [args.seed] if args.smoke_only else (args.seeds or [0, 1, 2, 3, 4])
    run(seeds, args.embeddings_dir, args.split, args.n_folds, args.n_epochs, args.out,
        args.smoke_only, args.strata_method, save_oof_dir=args.save_oof_dir)


if __name__ == "__main__":
    main()
