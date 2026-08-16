
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn

from pinksight.metrics import delong_ci, ece
from pinksight.trackb import GatedAttentionMIL, TrackBSubtypeHead, assert_gate_open

DEFAULT_BAGS_DIR = Path("data/pathology/bags/uni")  
DEFAULT_OUT = Path("reports/trackb/mil_cv_uni.json")  
LABELS_MANIFEST = Path("data/pathology/features/tcga_brca_titan_manifest.csv")
_LEGACY_TITAN_PKL = Path("data/pathology/features/TCGA_TITAN_features.pkl")

_N_SPLITS = 5
_N_EPOCHS = 20
_LR = 1e-3
_IN_DIM = 1536
_SEEDS_DEFAULT = "0,1,2"  
_SHUFFLE_SENTINEL_MAX = 0.60
_FRAMING = "methods-rigour result only — NOT headline claim"


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _read_labels(manifest: Path) -> dict[str, int]:
    if not manifest.exists():
        raise FileNotFoundError(f"labels manifest not found: {manifest}")
    df = pd.read_csv(manifest)
    for col in ("patient_id", "label_binary"):
        if col not in df.columns:
            raise ValueError(
                f"labels manifest {manifest} missing column '{col}'; has {list(df.columns)}"
            )
    return {str(p): int(g["label_binary"].iloc[0]) for p, g in df.groupby("patient_id")}


def _read_bag_h5(path: Path) -> np.ndarray:
    import h5py

    with h5py.File(Path(path), "r") as fh:
        return np.asarray(fh["features"][:], dtype=np.float32)


def load_bags(
    bags_dir: Path = DEFAULT_BAGS_DIR,
    manifest: Path = LABELS_MANIFEST,
    *,
    legacy_titan: bool = False,
) -> tuple[list[str], list[int], dict[str, dict]]:
    if legacy_titan:
        return _load_bags_titan_legacy(manifest)

    labels_map = _read_labels(manifest)
    bag_files = sorted(Path(bags_dir).glob("*.h5"))
    if not bag_files:
        raise FileNotFoundError(
            f"no .h5 bags in {bags_dir} — run scripts/encode_bags_uni.py first (or pass --legacy-titan)"
        )
    bags: dict[str, dict] = {}
    for h5 in bag_files:
        pid = h5.stem  
        if pid not in labels_map:
            continue  
        bags[pid] = {"emb": _read_bag_h5(h5), "label": labels_map[pid]}
    if not bags:
        raise AssertionError(
            f"no .h5 bag in {bags_dir} matched a label in {manifest} — check patient_id keys"
        )
    patients = list(bags.keys())
    labels = [bags[p]["label"] for p in patients]
    return patients, labels, bags


def _load_bags_titan_legacy(manifest: Path) -> tuple[list[str], list[int], dict[str, dict]]:
    if not manifest.exists():
        raise FileNotFoundError(
            f"manifest not found: {manifest} — run scripts/trackb_label_align.py first"
        )
    df = pd.read_csv(manifest)
    with _LEGACY_TITAN_PKL.open("rb") as fh:
        d = pickle.load(fh)  
    E = d["embeddings"]  

    bags: dict[str, dict] = {}
    for p, grp in df.groupby("patient_id"):
        rows = grp["tile_row_index"].to_numpy()
        label = int(grp["label_binary"].iloc[0])
        bags[str(p)] = {"emb": E[rows], "label": label}
    patients = list(bags.keys())
    labels = [bags[p]["label"] for p in patients]
    return patients, labels, bags


def _run_one_seed(
    patients: list[str],
    labels_arr: np.ndarray,
    bags: dict[str, dict],
    seed: int,
    in_dim: int,
    pos_weight: float,
) -> np.ndarray:
    _set_seed(seed)
    n = len(patients)
    groups = patients  
    sgkf = StratifiedGroupKFold(n_splits=_N_SPLITS, shuffle=True, random_state=seed)
    oof_scores = np.full(n, np.nan, dtype=float)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(patients, labels_arr, groups)):
        train_pat = {patients[i] for i in train_idx}
        val_pat = {patients[i] for i in val_idx}
        assert train_pat.isdisjoint(val_pat), (
            f"seed {seed} fold {fold}: train/val patient overlap — LOCK-2 violation: "
            f"{train_pat & val_pat}"
        )

        model = TrackBSubtypeHead(GatedAttentionMIL(in_dim=in_dim))
        optimizer = torch.optim.Adam(model.parameters(), lr=_LR)

        model.train()
        for _epoch in range(_N_EPOCHS):
            for i in train_idx:
                p = patients[i]
                bag = torch.tensor(bags[p]["emb"], dtype=torch.float32)
                y = torch.tensor([[bags[p]["label"]]], dtype=torch.float32)
                logit, _attn = model(bag)  
                loss = criterion(logit, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            for i in val_idx:
                bag = torch.tensor(bags[patients[i]]["emb"], dtype=torch.float32)
                logit, _attn = model(bag)
                oof_scores[i] = torch.sigmoid(logit).item()

    assert not np.isnan(oof_scores).any(), (
        f"seed {seed}: some patients received no OOF prediction — fold coverage gap"
    )
    return oof_scores


def run_cv_multiseed(
    bags_dir: Path = DEFAULT_BAGS_DIR,
    manifest: Path = LABELS_MANIFEST,
    *,
    seeds: tuple[int, ...] | list[int] = (0, 1, 2),
    shuffle: bool = True,
    legacy_titan: bool = False,
) -> dict:
    assert_gate_open()  
    seeds = list(seeds)
    if len(seeds) < 3:
        raise ValueError(
            f"[BOUND-F3] >=3 seeds required (docs/CLAIM_LEDGER.md eval standard: 3 min / 5 target); got {seeds}"
        )

    patients, labels, bags = load_bags(bags_dir, manifest, legacy_titan=legacy_titan)
    n = len(patients)
    labels_arr = np.asarray(labels, int)
    if labels_arr.min() == labels_arr.max():
        raise AssertionError("only one class present — AUROC undefined; check the labels manifest")
    in_dim = int(bags[patients[0]]["emb"].shape[1])  
    n_luma = int((labels_arr == 0).sum())
    n_basal = int((labels_arr == 1).sum())
    pos_weight = n_luma / max(n_basal, 1)  

    print(
        f"[trackb_mil_cv] subtype characterisation, TCGA-BRCA cohort — N={n} "
        f"(LumA={n_luma}, Basal={n_basal}); in_dim={in_dim}; seeds={seeds}; "
        f"source={'legacy-TITAN(768d)' if legacy_titan else 'UNI2-h .h5(1536d)'}"
    )

    per_seed_aurocs: list[float] = []
    oof_stack: list[np.ndarray] = []
    for seed in seeds:
        oof = _run_one_seed(patients, labels_arr, bags, seed, in_dim, pos_weight)
        per_seed_aurocs.append(float(roc_auc_score(labels_arr, oof)))
        oof_stack.append(oof)
        print(f"[trackb_mil_cv] seed {seed}: pooled-OOF AUROC {per_seed_aurocs[-1]:.4f}")

    pooled_oof = np.vstack(oof_stack).mean(axis=0)
    auc = float(roc_auc_score(labels_arr, pooled_oof))
    auc_delong, ci_lo, ci_hi = delong_ci(labels_arr, pooled_oof)
    ece_val = float(ece(labels_arr, pooled_oof))

    shuffle_auroc: float | None = None
    if shuffle:
        shuffled = np.random.default_rng(seed=0).permutation(labels_arr)
        shuffle_auroc = float(roc_auc_score(shuffled, pooled_oof))

    return {
        "auroc": auc,  
        "auc_delong": float(auc_delong),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "ece": ece_val,
        "shuffle_auroc": shuffle_auroc,
        "per_seed_aurocs": per_seed_aurocs,  
        "seed_mean": float(np.mean(per_seed_aurocs)),
        "seed_std": float(np.std(per_seed_aurocs)),
        "seeds": seeds,
        "n": n,
        "n_luma": n_luma,
        "n_basal": n_basal,
        "in_dim": in_dim,
        "source": "legacy-titan-768d" if legacy_titan else "uni2h-h5-1536d",
        "framing": _FRAMING,
    }


def _write_json(metrics: dict, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Track B multi-seed MIL CV — subtype characterisation, TCGA-BRCA (methods-rigour)"
    )
    ap.add_argument("--bags-dir", type=str, default=str(DEFAULT_BAGS_DIR),
                    help="dir of per-patient {patient_id}.h5 UNI2-h bags (default real-bag source)")
    ap.add_argument("--manifest", type=str, default=str(LABELS_MANIFEST),
                    help="labels CSV (needs patient_id,label_binary) — subtype labels source")
    ap.add_argument("--seeds", type=str, default=_SEEDS_DEFAULT,
                    help="comma-separated seeds; [BOUND-F3] >=3 required (e.g. 0,1,2)")
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT), help="output JSON result path")
    ap.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True,
                    help="run the label-shuffle leakage sentinel (default on; --no-shuffle to skip)")
    ap.add_argument("--legacy-titan", action="store_true",
                    help="fallback: read the legacy degenerate 768-D TITAN pkl instead of .h5 bags")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]

    m = run_cv_multiseed(
        bags_dir=Path(args.bags_dir),
        manifest=Path(args.manifest),
        seeds=seeds,
        shuffle=args.shuffle,
        legacy_titan=args.legacy_titan,
    )
    _write_json(m, Path(args.out))  

    print("[trackb_mil_cv] === RESULT — subtype characterisation, TCGA-BRCA cohort ===")
    print(f"[trackb_mil_cv] Pooled-OOF AUROC: {m['auroc']:.4f} "
          f"[{m['ci_lo']:.4f}, {m['ci_hi']:.4f}] (DeLong 95% CI)")
    print(f"[trackb_mil_cv] Per-seed AUROCs: {[round(a, 4) for a in m['per_seed_aurocs']]} "
          f"(mean {m['seed_mean']:.4f} +/- {m['seed_std']:.4f}) [BOUND-F3]")
    print(f"[trackb_mil_cv] ECE: {m['ece']:.4f}")
    if m["shuffle_auroc"] is not None:
        ok = m["shuffle_auroc"] < _SHUFFLE_SENTINEL_MAX
        print(f"[trackb_mil_cv] Shuffle AUROC: {m['shuffle_auroc']:.4f} "
              f"(< {_SHUFFLE_SENTINEL_MAX} {'OK' if ok else 'FAIL'})")
    print(f"[trackb_mil_cv] N={m['n']} (LumA={m['n_luma']} / Basal={m['n_basal']}), source={m['source']}")
    print(f"[trackb_mil_cv] framing: {m['framing']}")
    print(f"[trackb_mil_cv] wrote {args.out}")

    if m["shuffle_auroc"] is not None and not (m["shuffle_auroc"] < _SHUFFLE_SENTINEL_MAX):
        raise AssertionError(
            f"LEAKAGE SENTINEL: shuffle AUROC {m['shuffle_auroc']:.4f} >= {_SHUFFLE_SENTINEL_MAX} — "
            f"a leak path is present; STOP and investigate (JSON written to {args.out} for forensics)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
