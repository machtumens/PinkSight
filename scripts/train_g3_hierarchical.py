"""G3 #4 — train HierarchicalStagedFusion on the FROZEN G2 OOF embeddings (patient-level CV).

Clinical enters LATE and STRONG (H6 firewall). Trains ONLY the fusion head + its diagnosis heads on
the pre-computed frozen MRI (512-d) + clinical (128-d) embeddings — no 3D conv in the graph, so the
whole run is CPU-cheap. Patient-level StratifiedGroupKFold on configs/split_v2.yaml (frozen dev set);
the sealed Avanto holdout is NEVER touched here.

Two modes:
  --smoke-only : 1 seed, few folds/epochs, $0 CPU validity check (no gate threshold). Step 4.4.
  (default)    : full multi-seed run (Step 4.5, GPU) — H-G3-A gate number.

Output JSON schema (matches the Step 4.6 gate script + ablation harness):
  {"subtype": {"auroc": {"value", "ci95"}, "shuffle_auroc", "per_seed_auroc"},
   "grade": {...}, "n", "seeds", "n_folds", "n_epochs", "smoke_only", ...}

Claim-ledger: subtype characterisation + grade characterisation AT DIAGNOSIS. Never growth-rate.
Leakage (LOCK-2): ER/PR/HER2/Ki-67/... never enter — the model's forward asserts FORBIDDEN_FEATURES
absent from the modality dict. Grade labels (when present) are TARGETS, never inputs.

    PYTHONPATH=src .venv/bin/python scripts/train_g3_hierarchical.py --smoke-only \
        --seed 0 --n-folds 2 --n-epochs 5 \
        --embeddings-dir reports/G2_imaging/embeddings --split configs/split_v2.yaml \
        --out reports/G3_fusion_arch_bundle/smoke_hierarchical_s0.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from pinksight.metrics import delong_ci
from pinksight.models.fusion import HierarchicalStagedFusion

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/manifest_v1.csv"


def _subtype_labels() -> pd.Series:
    """Patient-level subtype labels from the manifest (luminal_like=0, tnbc=1)."""
    man = pd.read_csv(MANIFEST).set_index("patient_id")
    return man["subtype"].map({"luminal_like": 0, "tnbc": 1})


def _load_embed(path: Path) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    pids = np.array([str(p) for p in z["pids"]])
    return z["emb"].astype(np.float32), pids


def _dev_pids(split_yaml: Path) -> set[str]:
    s = yaml.safe_load(open(split_yaml))
    return set(str(p) for p in s.get("dev", []))


def build_aligned(embeddings_dir: Path, split_yaml: Path, seed: int):
    """Align MRI + clinical embeddings to the (mri ∩ clinical ∩ dev ∩ labeled) pid set.

    seed selects the per-seed embedding file (mri_embed_s{seed}.npz / clinical_embed_s{seed}.npz).
    Returns (feats_by_pid, y, groups) where feats stacks are index-aligned to `groups`."""
    xm, pm = _load_embed(embeddings_dir / f"mri_embed_s{seed}.npz")
    xc, pc = _load_embed(embeddings_dir / f"clinical_embed_s{seed}.npz")
    lab = _subtype_labels()
    dev = _dev_pids(split_yaml)

    inter = sorted(set(pm) & set(pc) & dev & set(lab.dropna().index))
    inter = [p for p in inter if not pd.isna(lab.get(p))]
    if not inter:
        raise SystemExit("empty mri ∩ clinical ∩ dev ∩ labeled pid set — STOP")

    om = {p: i for i, p in enumerate(pm)}
    oc = {p: i for i, p in enumerate(pc)}
    mri = np.stack([xm[om[p]] for p in inter])
    clin = np.stack([xc[oc[p]] for p in inter])
    y = lab.loc[inter].to_numpy(int)
    groups = np.array(inter)
    return {"mri": mri, "clinical": clin}, y, groups


def _train_one_fold(feats_tr, y_tr, feats_va, seed, n_epochs, lr=1e-3):
    """Train HierarchicalStagedFusion on one fold; return val subtype probabilities."""
    torch.manual_seed(seed)
    model = HierarchicalStagedFusion(
        {"mri": feats_tr["mri"].shape[1], "clinical": feats_tr["clinical"].shape[1]},
        fused_dim=128, p_modality_dropout=0.25,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    xt = {k: torch.from_numpy(v).float() for k, v in feats_tr.items()}
    yt = torch.from_numpy(y_tr).float()
    model.train()
    for _ in range(n_epochs):
        opt.zero_grad()
        out = model(xt)  # random modality dropout active in train()
        total, _ = model.joint_loss(out, yt)  # subtype-only (grade labels absent -> masked no-op)
        if not torch.isfinite(total):
            raise SystemExit(f"NaN/inf loss at seed {seed} — STOP (pipeline bug)")
        total.backward()
        opt.step()
    model.eval()
    xv = {k: torch.from_numpy(v).float() for k, v in feats_va.items()}
    with torch.no_grad():
        logit = model(xv, drop=set())["subtype_logit"].reshape(-1)
    return torch.sigmoid(logit).cpu().numpy()


def pooled_oof(feats, y, groups, seed, n_folds, n_epochs, shuffle_labels=False):
    """StratifiedGroupKFold pooled-OOF subtype probabilities (patient-disjoint folds)."""
    y_used = y.copy()
    if shuffle_labels:
        rng = np.random.default_rng(1234 + seed)
        y_used = rng.permutation(y_used)
    oof = np.full(len(y_used), np.nan, dtype=float)
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr, va in sgkf.split(feats["mri"], y_used, groups):
        # patient-disjoint assertion (LOCK-2)
        assert not (set(groups[tr]) & set(groups[va])), "patient overlap across folds — STOP"
        feats_tr = {k: v[tr] for k, v in feats.items()}
        feats_va = {k: v[va] for k, v in feats.items()}
        oof[va] = _train_one_fold(feats_tr, y_used[tr], feats_va, seed, n_epochs)
    assert not np.isnan(oof).any(), "un-scored patient in OOF — STOP"
    return oof, y_used


def run(seeds, embeddings_dir, split_yaml, n_folds, n_epochs, out_path, smoke_only,
        save_oof_dir=None):
    per_seed_auc, oof_by_seed, y_by_seed, shuffle_auc = {}, {}, {}, {}
    if save_oof_dir is not None:
        save_oof_dir = Path(save_oof_dir)
        save_oof_dir.mkdir(parents=True, exist_ok=True)
    for s in seeds:
        feats, y, groups = build_aligned(embeddings_dir, split_yaml, s)
        oof, y_used = pooled_oof(feats, y, groups, s, n_folds, n_epochs)
        per_seed_auc[s] = float(roc_auc_score(y_used, oof))
        oof_by_seed[s] = oof
        y_by_seed[s] = y_used
        shuf, y_shuf = pooled_oof(feats, y, groups, s, n_folds, n_epochs, shuffle_labels=True)
        shuffle_auc[s] = float(roc_auc_score(y_shuf, shuf))
        # Item 2 (additive): persist per-sample OOF probs + patient IDs for the downstream paired
        # DeLong vs the clinical anchor (item 3). Guarded by --save-oof-dir; when the flag is
        # omitted this block is skipped and behavior is byte-identical to the frozen version.
        # pids/y/oof are index-aligned (groups order from build_aligned == oof/y_used order).
        if save_oof_dir is not None:
            np.savez(save_oof_dir / f"hierarchical_oof_s{s}.npz",
                     pids=groups, y=y_used, oof=oof)

    # Pool across seeds for a single AUROC + DeLong CI (seed 0 CI as representative, mean value).
    mean_auc = float(np.mean(list(per_seed_auc.values())))
    auc0, lo, hi = delong_ci(y_by_seed[seeds[0]], oof_by_seed[seeds[0]])
    doc = {
        "gate": "G3 #4 hierarchical staged fusion (H-G3-A)",
        "smoke_only": smoke_only,
        "subtype": {
            "auroc": {"value": round(mean_auc, 4), "ci95": [round(lo, 4), round(hi, 4)],
                      "ci_method": "delong", "ci_seed": int(seeds[0])},
            "shuffle_auroc": round(float(np.mean(list(shuffle_auc.values()))), 4),
            "per_seed_auroc": {str(k): round(v, 4) for k, v in per_seed_auc.items()},
            "per_seed_shuffle": {str(k): round(v, 4) for k, v in shuffle_auc.items()},
        },
        "n": int(len(y_by_seed[seeds[0]])),
        "seeds": [int(s) for s in seeds],
        "n_folds": n_folds,
        "n_epochs": n_epochs,
        "note": ("$0 CPU smoke — architecture validity only, NOT a gate number. Clinical enters LATE "
                 "and STRONG (H6 firewall). Patient-level StratifiedGroupKFold on frozen split_v2 dev; "
                 "Avanto holdout untouched. Subtype characterisation at diagnosis; no growth-rate."
                 if smoke_only else
                 "H-G3-A gate: hierarchical fused subtype AUROC vs clinical-alone 0.708 (H6 anchor)."),
        "claim_ledger": "subtype + grade characterisation at diagnosis; ER/PR/HER2/Ki-67 excluded (LOCK-2)",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"[G3 #4] subtype AUROC {mean_auc:.4f} [{lo:.4f},{hi:.4f}] "
          f"shuffle {doc['subtype']['shuffle_auroc']:.4f}  N={doc['n']}  seeds={seeds}")
    print(f"wrote {out_path}")
    return doc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke-only", action="store_true", help="1-seed $0 CPU validity check")
    ap.add_argument("--seed", type=int, default=0, help="single seed for --smoke-only")
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="multi-seed (full run)")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--n-epochs", type=int, default=50)
    ap.add_argument("--embeddings-dir", type=Path, default=ROOT / "reports/G2_imaging/embeddings")
    ap.add_argument("--split", type=Path, default=ROOT / "configs/split_v2.yaml")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--save-oof-dir", type=Path, default=None,
                    help="if set, write per-seed hierarchical_oof_s{seed}.npz (pids/y/oof) for the "
                         "downstream paired DeLong vs the clinical anchor (item 2; additive — has no "
                         "effect on any existing output or on behavior when the flag is omitted)")
    args = ap.parse_args()

    seeds = [args.seed] if args.smoke_only else (args.seeds or [0, 1, 2, 3, 4])
    run(seeds, args.embeddings_dir, args.split, args.n_folds, args.n_epochs, args.out,
        args.smoke_only, save_oof_dir=args.save_oof_dir)


if __name__ == "__main__":
    main()
