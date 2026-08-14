"""Track B MIL cross-validation — subtype characterisation, TCGA-BRCA cohort (decisions.md [5.2]).

Patient-level 5-fold StratifiedGroupKFold CV over frozen tile-embedding bags using
GatedAttentionMIL -> TrackBSubtypeHead with a class-weighted BCE loss (~2.9:1 LumA:Basal). Runs
across >=3 seeds ([BOUND-F3], CLAUDE.md eval standard: 3 min / 5 target), pools the per-seed OOF
scores, and reports pooled-OOF AUROC + 95% DeLong CI + ECE + per-seed spread + a label-shuffle
leakage sentinel. Writes a JSON result artifact.

DEFAULT source is the real per-patient UNI2-h `.h5` bags under data/pathology/bags/uni/ (dataset
`features`, shape (n_tiles, 1536) — [BOUND-F7]). The legacy degenerate 768-D TITAN pkl (mean 1.06
tiles/patient) is kept ONLY as a clearly-labelled `--legacy-titan` fallback for the old baseline.
in_dim is inferred from the loaded bags, so the .h5 path uses 1536 and the legacy path uses 768
without any code change.

CLAIM LEDGER (hard): this is intra-TCGA-BRCA subtype characterisation ONLY. NEVER a
cross-institution generalisation claim. NEVER growth-rate / kinetics / early-detection framing.
AUROC is reported ONLY with its DeLong CI — never a bare number. This is a METHODS-RIGOUR result,
NOT a headline scientific claim.

FRAMING GUARD (LOCK-1): this script handles Track B numbers ONLY and NEVER juxtaposes them against
Track A (Duke) numbers in any output, table, or log line — side-by-side tabulation is a soft
cross-institution comparison and is forbidden. The result is reported regardless of sign (a negative
finding — real bags below the TITAN baseline — is documented honestly, never suppressed).

Integrity guarantees enforced here:
  * assert_gate_open() gates the real-data path (TRACKB_GATE_OPEN=True, ratified 2026-07-11).
  * Patient-disjoint folds — an explicit train/val patient-overlap assertion per fold.
  * Label-shuffle sentinel — shuffled-label OOF AUROC must be < 0.60 or the run STOPS (leak path).

Corrected imports (validate-contract E1/E2): metrics live in pinksight.metrics —
`from pinksight.metrics import delong_ci, ece`  (delong_ci -> (auc, lo, hi); ece(y, p) -> float).

$0 compute — CPU only. Real 1536-D bags with ~2000 tiles/patient still run on CPU (forward-only,
no gradients through the frozen encoder); the MIL head is tiny.

Run (real UNI2-h bags):  uv run --extra ml python scripts/trackb_mil_cv.py \
                           --bags-dir data/pathology/bags/uni --seeds 0,1,2 --out reports/trackb/mil_cv_uni.json
Run (legacy TITAN pkl):  uv run --extra ml python scripts/trackb_mil_cv.py --legacy-titan --seeds 0,1,2
"""

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

# --- Paths / contract constants ------------------------------------------------------------------
DEFAULT_BAGS_DIR = Path("data/pathology/bags/uni")  # real UNI2-h .h5 bags (DEFAULT source)
DEFAULT_OUT = Path("reports/trackb/mil_cv_uni.json")  # multi-seed pooled-OOF result artifact
# Labels source (patient_id -> label_binary). The TITAN manifest already carries this mapping; it is
# the same manifest the legacy path reads, so it serves both the .h5 and legacy modes.
LABELS_MANIFEST = Path("data/pathology/features/tcga_brca_titan_manifest.csv")
# LEGACY degenerate 768-D TITAN pkl (mean 1.06 tiles/patient) — fallback only, via --legacy-titan.
_LEGACY_TITAN_PKL = Path("data/pathology/features/TCGA_TITAN_features.pkl")

_N_SPLITS = 5
_N_EPOCHS = 20
_LR = 1e-3
# [BOUND-F7] real UNI2-h bags are 1536-D (was 768 for the legacy degenerate TITAN pkl). in_dim is
# INFERRED from the loaded bags at runtime, so this constant is the documented expectation for the
# real-bag run — the .h5 path resolves to 1536 and the legacy path to 768 automatically.
_IN_DIM = 1536
_SEEDS_DEFAULT = "0,1,2"  # [BOUND-F3] >=3 seeds required (CLAUDE.md: DeLong CI + ECE + multi-seed)
_SHUFFLE_SENTINEL_MAX = 0.60
_FRAMING = "methods-rigour result only — NOT headline claim"


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------- labels + bag IO
def _read_labels(manifest: Path) -> dict[str, int]:
    """patient_id -> binary subtype label (0=LumA, 1=Basal) from the labels manifest CSV.

    The TITAN manifest has one row per tile; collapse to a single label per patient. Never fabricate
    a label — a patient with no manifest row is simply absent from the map (and its bag is skipped).
    """
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
    """Read a per-patient bag .h5 -> (n_tiles, D) float32 (dataset `features`), as written by
    scripts/encode_bags_uni.write_bag_h5."""
    import h5py

    with h5py.File(Path(path), "r") as fh:
        return np.asarray(fh["features"][:], dtype=np.float32)


def load_bags(
    bags_dir: Path = DEFAULT_BAGS_DIR,
    manifest: Path = LABELS_MANIFEST,
    *,
    legacy_titan: bool = False,
) -> tuple[list[str], list[int], dict[str, dict]]:
    """Return (patients, labels, bags) where bags[p] = {'emb': (n_tiles, D), 'label': int}.

    DEFAULT: real per-patient UNI2-h `.h5` bags under `bags_dir` (dataset `features`, (n_tiles, 1536)
    per [BOUND-F7]), joined to subtype labels from `manifest`. `legacy_titan=True` instead reads the
    degenerate 768-D TITAN pkl — kept only as a labelled fallback for the old single-tile baseline.
    """
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
        pid = h5.stem  # bag files are named {patient_id}.h5 (12-char TCGA barcode)
        if pid not in labels_map:
            continue  # bag with no matching label -> skip (never fabricate a label)
        bags[pid] = {"emb": _read_bag_h5(h5), "label": labels_map[pid]}
    if not bags:
        raise AssertionError(
            f"no .h5 bag in {bags_dir} matched a label in {manifest} — check patient_id keys"
        )
    patients = list(bags.keys())
    labels = [bags[p]["label"] for p in patients]
    return patients, labels, bags


def _load_bags_titan_legacy(manifest: Path) -> tuple[list[str], list[int], dict[str, dict]]:
    """LEGACY 768-D TITAN pkl path (degenerate 1-tile baseline, mean 1.06 tiles/patient). Retained
    as a clearly-labelled fallback only — the real-bag run uses the .h5 path in load_bags()."""
    if not manifest.exists():
        raise FileNotFoundError(
            f"manifest not found: {manifest} — run scripts/trackb_label_align.py first"
        )
    df = pd.read_csv(manifest)
    with _LEGACY_TITAN_PKL.open("rb") as fh:
        d = pickle.load(fh)  # noqa: S301 — trusted local project artifact
    E = d["embeddings"]  # (11658, 768) float32

    bags: dict[str, dict] = {}
    for p, grp in df.groupby("patient_id"):
        rows = grp["tile_row_index"].to_numpy()
        label = int(grp["label_binary"].iloc[0])
        bags[str(p)] = {"emb": E[rows], "label": label}
    patients = list(bags.keys())
    labels = [bags[p]["label"] for p in patients]
    return patients, labels, bags


# ---------------------------------------------------------------- CV
def _run_one_seed(
    patients: list[str],
    labels_arr: np.ndarray,
    bags: dict[str, dict],
    seed: int,
    in_dim: int,
    pos_weight: float,
) -> np.ndarray:
    """One full patient-level 5-fold StratifiedGroupKFold pass at `seed`; returns the OOF score
    vector (sigmoid probability per patient). Folds are patient-disjoint (LOCK-2 assertion per fold).
    """
    _set_seed(seed)
    n = len(patients)
    groups = patients  # group == patient (patient-disjoint GroupKFold)
    sgkf = StratifiedGroupKFold(n_splits=_N_SPLITS, shuffle=True, random_state=seed)
    oof_scores = np.full(n, np.nan, dtype=float)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(patients, labels_arr, groups)):
        # --- patient-disjoint assertion (LOCK-2): train and val never share a patient ------------
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
                logit, _attn = model(bag)  # logit is [1, 1]
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
    """Multi-seed patient-level CV ([BOUND-F3]). Runs one full 5-fold CV per seed, pools the per-seed
    OOF scores (seed-average per patient), then reports pooled-OOF AUROC + 95% DeLong CI + ECE, the
    per-seed AUROC spread (mean/std), and a label-shuffle leakage sentinel on the pooled OOF.
    """
    assert_gate_open()  # real-data gate — raises TrackBGateClosedError if closed
    seeds = list(seeds)
    if len(seeds) < 3:
        raise ValueError(
            f"[BOUND-F3] >=3 seeds required (CLAUDE.md eval standard: 3 min / 5 target); got {seeds}"
        )

    patients, labels, bags = load_bags(bags_dir, manifest, legacy_titan=legacy_titan)
    n = len(patients)
    labels_arr = np.asarray(labels, int)
    if labels_arr.min() == labels_arr.max():
        raise AssertionError("only one class present — AUROC undefined; check the labels manifest")
    in_dim = int(bags[patients[0]]["emb"].shape[1])  # inferred: 1536 (.h5) or 768 (legacy TITAN)
    n_luma = int((labels_arr == 0).sum())
    n_basal = int((labels_arr == 1).sum())
    pos_weight = n_luma / max(n_basal, 1)  # class-weighted BCE (real cohort ~2.9:1 LumA:Basal)

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

    # --- pool per-seed OOF (seed-average per patient), then AUROC + DeLong CI + ECE on the pool ---
    pooled_oof = np.vstack(oof_stack).mean(axis=0)
    auc = float(roc_auc_score(labels_arr, pooled_oof))
    auc_delong, ci_lo, ci_hi = delong_ci(labels_arr, pooled_oof)
    ece_val = float(ece(labels_arr, pooled_oof))

    # --- label-shuffle sentinel: permuted labels vs the SAME pooled OOF must collapse to chance ---
    shuffle_auroc: float | None = None
    if shuffle:
        shuffled = np.random.default_rng(seed=0).permutation(labels_arr)
        shuffle_auroc = float(roc_auc_score(shuffled, pooled_oof))

    return {
        "auroc": auc,  # pooled-OOF AUROC (primary — reported with DeLong CI below, never bare)
        "auc_delong": float(auc_delong),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "ece": ece_val,
        "shuffle_auroc": shuffle_auroc,
        "per_seed_aurocs": per_seed_aurocs,  # [BOUND-F3] one pooled-OOF AUROC per seed (>=3)
        "seed_mean": float(np.mean(per_seed_aurocs)),
        "seed_std": float(np.std(per_seed_aurocs)),
        "seeds": seeds,
        "n": n,
        "n_luma": n_luma,
        "n_basal": n_basal,
        "in_dim": in_dim,
        "source": "legacy-titan-768d" if legacy_titan else "uni2h-h5-1536d",
        # Methods-rigour framing (Rule 4 / LOCK-1): NOT a headline claim; NEVER juxtaposed vs Track A.
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
    _write_json(m, Path(args.out))  # write the artifact FIRST so it survives a sentinel abort

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

    # leakage sentinel enforcement (AFTER the JSON is written): shuffle AUROC >= 0.60 means a leak.
    if m["shuffle_auroc"] is not None and not (m["shuffle_auroc"] < _SHUFFLE_SENTINEL_MAX):
        raise AssertionError(
            f"LEAKAGE SENTINEL: shuffle AUROC {m['shuffle_auroc']:.4f} >= {_SHUFFLE_SENTINEL_MAX} — "
            f"a leak path is present; STOP and investigate (JSON written to {args.out} for forensics)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
