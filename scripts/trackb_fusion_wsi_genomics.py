"""Piece B — intra-TCGA-BRCA WSI (frozen TITAN OOF) + mRNA-genomics late-fusion subtype model,
with a STRUCTURALLY SEPARATE PAM50 proliferation-index Ridge co-target (the LOCK-2 firewall).

Asks: does adding a pre-registered 6-gene mRNA panel to the frozen-TITAN WSI logit beat WSI-alone by
ΔAUC on the LumA-vs-Basal contrast, N=640 TCGA-BRCA? WSI-alone is already near-ceiling (arm3 0.9646 /
UNI2-h 0.9675), so a null/marginal/negative ΔAUC is the EXPECTED, acceptable, honestly-reported
outcome — this is NOT gated on beating a bar (see the pre-registered decision rule in the plan).

HONEST FRAMING (non-negotiable — decisions.md claim ledger, E4):
  This is LOGISTIC-REGRESSION LATE-FUSION STACKING of [frozen TITAN OOF logit, scaled mRNA panel].
  It is NOT cross-attention MIL fusion (same honest-framing rationale as trackb_fusion_deltaauc.py:
  the PoC bags are degenerate at ~1 tile/patient). Report ONLY as "linear-probe TITAN + late-fusion
  mRNA-genomics bundle" — NEVER "cross-attention fusion achieves X".

  Intra-TCGA-BRCA ONLY. NOT a cross-institution generalisation claim. NOT growth-rate / kinetics /
  early-detection. AUROC / ΔAUC are ALWAYS reported with CI + p-value + a shuffle sentinel.

  NON-INDEPENDENCE CAVEAT (must stay in the report): Piece B reuses the SAME frozen TITAN embeddings
  as arm3 and arm5. It is therefore NOT independent evidence of a WSI subtype signal — the three are
  readouts of one embedding set on one 640-patient cohort.

FIREWALL (LOCK-2 — the load-bearing integrity guarantee):
  The auxiliary PAM50 proliferation-index co-target is fit by a STRUCTURALLY SEPARATE sklearn
  estimator (Ridge), sharing ZERO trainable parameters with the subtype LogisticRegression. The
  proliferation score is a molecular SNAPSHOT at diagnosis (a Ki-67 surrogate index; arm5's target,
  reused verbatim) — permitted here ONLY as a TARGET, never as a subtype-model input feature, and
  never concatenated into the subtype design matrix. `run(compute_proliferation=False)` and
  `run(compute_proliferation=True)` produce BIT-IDENTICAL subtype OOF scores by construction: the
  subtype path is a pure function of [TITAN logit, mRNA panel, labels, folds] with no dependence on
  whether the auxiliary head runs. tests/test_leakage.py::test_pam50_proliferation_never_feeds_subtype_head
  proves this both structurally (synthetic) and empirically (real-data bit-for-bit equivalence).

$0 compute — CPU only. The mRNA read is a single streaming file scan; the cached TITAN OOF logits are
reused, so the whole run is well under a minute on CPU. No GPU / RunPod.

Run:  uv run --extra ml python scripts/trackb_fusion_wsi_genomics.py
      uv run --extra ml python scripts/trackb_fusion_wsi_genomics.py --selfcheck
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from pinksight import FORBIDDEN_FEATURES
from pinksight.metrics import delong_ci, delong_paired, ece
from pinksight.trackb import assert_gate_open

# Sibling imports — reuse the canonical Track B fold split / WSI baseline (trackb_fusion_deltaauc)
# and arm5's proliferation-target machinery, verbatim (plan Touchpoints: READ-ONLY reuse via import).
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "novel_heads"))

from arm5_proliferation_snapshot import (  # noqa: E402
    PROLIF_GENES,
    _bootstrap_r_ci,
    _fit_predict_ridge,
    _pam50_proliferation_score,
)
from trackb_fusion_deltaauc import (  # noqa: E402
    _FORBIDDEN_HISTOLOGY,
    _FORBIDDEN_SURVIVAL,
    _fold_splitter,
    _get_unimodal_oof,
    _load_manifest,
)

# --- Contract constants ---------------------------------------------------------------------------
# Pre-registered 6-gene mRNA panel (E1: locked BEFORE any run — do NOT substitute after seeing a
# result). Extends the 4 genes already vetted in the old synthetic leakage fixture
# {TP53, CDH1, PIK3CA, GATA3} with two more common, non-PAM50-centroid TCGA-BRCA driver genes.
MODALITY_C_GENES = ["TP53", "PIK3CA", "GATA3", "CDH1", "MAP3K1", "PTEN"]

# GATA3-circularity probe (PI-approved 04-08-26): the ORIGINAL 6-gene panel minus GATA3, the one
# PAM50 luminal-lineage classifier gene. Removing it isolates whether the fusion ΔAUC over WSI-alone
# is genuine non-circular driver-gene signal or a GATA3 subtype-correlate artefact. The original
# panel above is NOT deleted — this is an additive comparison code path (`--compare-ablation`).
MODALITY_C_GENES_ABLATED = ["TP53", "PIK3CA", "CDH1", "MAP3K1", "PTEN"]

EXPR = Path("data/genomics/tcga/brca_tcga_pan_can_atlas_2018/data_mrna_seq_v2_rsem.txt")
GENOMICS_OOF_NPY = Path("data/pathology/features/tcga_brca_fusion_genomics_oof.npy")
GENOMICS_OOF_NPY_ABLATED = Path("data/pathology/features/tcga_brca_fusion_genomics_ablated_oof.npy")

SEED = 42
SHUFFLE_SENTINEL_MAX = 0.60  # fusion-subtype shuffle AUC must be < this (leak sentinel)
PROLIF_SHUFFLE_ABS_MAX = 0.25  # |proliferation shuffle r| must collapse below this
N_EXPECTED = 640

# Forbidden-gene union (E7: PROLIF_GENES is a dict[str, tuple[str,...]] -> use set(PROLIF_GENES) for
# its keys; a bare `... | PROLIF_GENES` raises TypeError). Panel must be disjoint from ALL of these:
# the leakage ledger (incl. the ESR1/PGR/ERBB2/MKI67 gene-symbol mirror), the reused histology /
# survival column sets, AND the auxiliary target's own 11 proliferation genes (so "independent"
# firewall is not undermined by input-level collinearity).
_ALL_FORBIDDEN_GENES = (
    FORBIDDEN_FEATURES | _FORBIDDEN_HISTOLOGY | _FORBIDDEN_SURVIVAL | set(PROLIF_GENES)
)
_leaked_at_import = set(MODALITY_C_GENES) & _ALL_FORBIDDEN_GENES
assert not _leaked_at_import, (
    f"LEAKAGE (import-time fail-fast): MODALITY_C_GENES intersect a forbidden/proliferation set: "
    f"{_leaked_at_import}"
)


def _assemble_modality_c(patients: list[str], genes: list[str] | None = None) -> np.ndarray:
    """Streaming-filter read of the mRNA matrix for `genes` -> (640, len(genes)), 640-patient order.

    `genes` defaults to the pre-registered MODALITY_C_GENES (6). The GATA3-ablation probe passes
    MODALITY_C_GENES_ABLATED (5). Everything else — file read, log2(x+1), 12-char barcode grouping,
    alignment, EV1 NaN transparency, leakage-safe constant 0-fill — is identical across panels.
    Mirrors arm5._pam50_proliferation_score's file-reading pattern exactly (never a full 155 MB read).
    """
    genes = genes if genes is not None else MODALITY_C_GENES
    if not EXPR.exists():
        raise FileNotFoundError(f"Piece B: mRNA matrix absent: {EXPR}")

    wanted = set(genes)
    rows: list[list[str]] = []
    with EXPR.open() as fh:
        header = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            if line.split("\t", 1)[0] in wanted:
                rows.append(line.rstrip("\n").split("\t"))

    found = {r[0] for r in rows}
    missing = [g for g in genes if g not in found]
    if missing:
        # E1: a locked-panel gene absent from the real file is a data-availability gap, not a licence
        # to silently substitute a different gene.
        raise FileNotFoundError(
            f"Piece B (E1): panel gene(s) absent from the real mRNA matrix: {missing}. "
            "STOP and report a data-availability gap — do NOT substitute genes."
        )

    samples = header[2:]  # cols 0,1 = Hugo_Symbol, Entrez_Gene_Id
    by_gene = {
        r[0]: np.asarray(
            [float(v) if v not in ("", "NA") else np.nan for v in r[2:]], dtype=float
        )
        for r in rows
    }
    expr = pd.DataFrame(by_gene, index=samples)  # samples x genes
    expr = np.log2(expr + 1.0)
    expr.index = [s[:12] for s in expr.index]  # TCGA-XX-XXXX-01 -> 12-char patient barcode
    expr = expr.groupby(level=0).mean()  # average repeat samples per patient

    mat = expr.reindex(index=list(patients), columns=genes)
    nan_counts = mat.isna().sum(axis=0)
    print(f"  [modality-C] per-gene NaN counts BEFORE fill ({len(genes)}-gene panel, EV1 transparency):")
    for g in genes:
        print(f"    {g}: {int(nan_counts[g])}")

    mc = np.nan_to_num(mat.to_numpy(dtype=float), nan=0.0)  # constant 0-fill uses NO fold info (leak-safe)
    assert mc.shape == (len(patients), len(genes)), (
        f"expected ({len(patients)},{len(genes)}) got {mc.shape}"
    )
    return mc


def _subtype_fusion_oof(
    oof_logits_unimodal: np.ndarray, modality_c: np.ndarray, oof_labels: np.ndarray, folds
) -> np.ndarray:
    """STEP B2 — per-fold LR stacking of [TITAN logit, TRAIN-scaled mRNA panel] -> subtype OOF probs.

    PURE function of its inputs with ZERO dependence on the proliferation co-target. Each fold uses
    its OWN StandardScaler instance (never shared/mutated across the firewall). The design matrix has
    EXACTLY 1 + len(MODALITY_C_GENES) columns — the proliferation score is never a column here.
    """
    n = len(oof_labels)
    oof = np.zeros(n)
    for fold, (tr, va) in enumerate(folds):
        scaler = StandardScaler()
        mc_tr = scaler.fit_transform(modality_c[tr])  # scaler fit on TRAIN only
        mc_va = scaler.transform(modality_c[va])
        x_tr = np.hstack([oof_logits_unimodal[tr].reshape(-1, 1), mc_tr])
        x_va = np.hstack([oof_logits_unimodal[va].reshape(-1, 1), mc_va])
        assert x_tr.shape[1] == 1 + modality_c.shape[1], (
            f"subtype design matrix width {x_tr.shape[1]} != 1 + {modality_c.shape[1]} "
            "(proliferation score must NEVER be a subtype column — LOCK-2 firewall)"
        )
        lr = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
        lr.fit(x_tr, oof_labels[tr])
        oof[va] = lr.predict_proba(x_va)[:, 1]
        print(f"  [fusion] fold {fold}: train={len(tr)} val={len(va)} stacked (X cols={x_tr.shape[1]})")
    return oof


def _proliferation_oof(
    oof_logits_unimodal: np.ndarray, modality_c: np.ndarray, target: np.ndarray, folds
) -> np.ndarray:
    """STEP B3 — auxiliary PAM50 proliferation-index Ridge on the SAME inputs + SAME folds, but a
    DIFFERENT, independent estimator instance (arm5._fit_predict_ridge makes its own StandardScaler +
    Ridge(alpha=1.0)). ZERO shared parameters with _subtype_fusion_oof; the target is never fed into
    the subtype path. This is the structural firewall.
    """
    n = len(target)
    oof = np.full(n, np.nan)
    for _fold, (tr, va) in enumerate(folds):
        x_tr = np.hstack([oof_logits_unimodal[tr].reshape(-1, 1), modality_c[tr]])
        x_va = np.hstack([oof_logits_unimodal[va].reshape(-1, 1), modality_c[va]])
        oof[va] = _fit_predict_ridge(x_tr, target[tr], x_va)  # arm5's own scaler+Ridge, TRAIN-only fit
    assert not np.isnan(oof).any(), "proliferation OOF coverage gap — some patient got no prediction"
    return oof


def _proliferation_target(patients: list[str]) -> np.ndarray:
    """PAM50 proliferation index per patient (arm5._pam50_proliferation_score, reused verbatim),
    aligned to the 640-patient order. Used ONLY as a Ridge TARGET — never a subtype input.
    """
    score = _pam50_proliferation_score()  # pd.Series indexed by 12-char barcode
    aligned = score.reindex(list(patients))
    n_missing = int(aligned.isna().sum())
    if n_missing:
        # E1: cannot fabricate a regression target. arm5's probe confirmed 640/640 coverage; a miss
        # is a data-availability gap, reported honestly rather than 0-filled (0-filling a TARGET
        # invents a label).
        raise FileNotFoundError(
            f"Piece B (E1): {n_missing} patient(s) lack a PAM50 proliferation score — cannot fabricate "
            "a regression target. STOP and report a data-availability gap."
        )
    return aligned.to_numpy(dtype=float)


def _persist_oof(oof_scores_fusion: np.ndarray, path: Path = GENOMICS_OOF_NPY) -> None:
    """Persist the subtype-fusion OOF (gitignored; mirrors OOF_NPY precedent). Always regenerated —
    never loaded as an input cache — so the E3 stale-trust risk does not apply to this artifact. The
    ablated panel persists to a SEPARATE path so it never overwrites the 6-gene OOF.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, oof_scores_fusion)
    print(f"  [fusion] saved subtype-fusion OOF ({len(oof_scores_fusion)}) to {path} (gitignored)")


def run(compute_proliferation: bool = True, panel: list[str] | None = None) -> dict:
    """Run Piece B. Returns the trackb_fusion_deltaauc-shaped dict PLUS `oof_scores_fusion` (for the
    firewall equivalence test) and, when compute_proliferation=True, a `proliferation` sub-dict.

    `panel` defaults to the pre-registered MODALITY_C_GENES (6). The GATA3-ablation probe passes
    MODALITY_C_GENES_ABLATED (5). WSI TITAN OOF, seed, folds, N, and the proliferation firewall are
    IDENTICAL across panels — only the mRNA feature columns change — so the two ΔAUCs are comparable.

    compute_proliferation=False computes the subtype path ONLY — and the returned subtype OOF is
    BIT-IDENTICAL to the compute_proliferation=True run for the SAME panel (the firewall).
    """
    assert_gate_open()  # Track B real-data hard gate (LOCK-6) — raises if closed

    panel = panel if panel is not None else MODALITY_C_GENES
    ablated = set(panel) != set(MODALITY_C_GENES)
    oof_path = GENOMICS_OOF_NPY_ABLATED if ablated else GENOMICS_OOF_NPY

    leaked = set(panel) & _ALL_FORBIDDEN_GENES  # belt-and-suspenders (mirrors import assert)
    assert not leaked, f"LEAKAGE: panel contains forbidden gene(s): {leaked}"

    patients, oof_labels, manifest_df = _load_manifest()
    folds = _fold_splitter(patients, oof_labels)  # SINGLE canonical split, reused for both heads (E4)

    print("[trackb_fusion_wsi_genomics] === STEP B0: unimodal TITAN OOF (WSI-alone, raw logits) ===")
    oof_logits_unimodal = _get_unimodal_oof(patients, oof_labels, manifest_df, folds)

    print(f"[trackb_fusion_wsi_genomics] === STEP B1: assemble mRNA panel ({len(panel)} genes) ===")
    modality_c = _assemble_modality_c(patients, panel)

    print("[trackb_fusion_wsi_genomics] === STEP B2: subtype fusion (LR stacking) ===")
    oof_scores_fusion = _subtype_fusion_oof(oof_logits_unimodal, modality_c, oof_labels, folds)

    print("[trackb_fusion_wsi_genomics] === STEP B4: subtype metrics ===")
    auroc_u = roc_auc_score(oof_labels, oof_logits_unimodal)
    auroc_f = roc_auc_score(oof_labels, oof_scores_fusion)
    _, u_lo, u_hi = delong_ci(oof_labels, oof_logits_unimodal)
    _, f_lo, f_hi = delong_ci(oof_labels, oof_scores_fusion)
    paired = delong_paired(oof_labels, oof_logits_unimodal, oof_scores_fusion)  # delta = fusion - WSI
    ece_u = ece(oof_labels, 1.0 / (1.0 + np.exp(-oof_logits_unimodal)))  # logits -> prob for a fair ECE
    ece_f = ece(oof_labels, oof_scores_fusion)

    rng = np.random.default_rng(seed=0)
    shuffle_auc = roc_auc_score(rng.permutation(oof_labels), oof_scores_fusion)
    if not (shuffle_auc < SHUFFLE_SENTINEL_MAX):
        raise AssertionError(
            f"LEAKAGE SENTINEL: fusion shuffle AUC {shuffle_auc:.4f} >= {SHUFFLE_SENTINEL_MAX} — "
            "a leak path is present; STOP and investigate."
        )

    _persist_oof(oof_scores_fusion, oof_path)

    result: dict = {
        "auroc_u": float(auroc_u), "u_lo": float(u_lo), "u_hi": float(u_hi), "ece_u": float(ece_u),
        "auroc_f": float(auroc_f), "f_lo": float(f_lo), "f_hi": float(f_hi), "ece_f": float(ece_f),
        "delta": float(paired["delta"]),
        "ci95": [float(paired["ci95"][0]), float(paired["ci95"][1])],
        "z": float(paired["z"]), "p": float(paired["p"]), "shuffle_auc": float(shuffle_auc),
        "n": int(len(oof_labels)),
        "panel": list(panel), "n_genes": len(panel),
        "oof_scores_fusion": oof_scores_fusion,  # firewall equivalence-test surface (E2)
    }

    if compute_proliferation:
        print("[trackb_fusion_wsi_genomics] === STEP B3: auxiliary PAM50 proliferation Ridge (SEPARATE) ===")
        target = _proliferation_target(patients)
        oof_prolif = _proliferation_oof(oof_logits_unimodal, modality_c, target, folds)
        pearson_r = float(np.corrcoef(target, oof_prolif)[0, 1])
        lo, hi = _bootstrap_r_ci(target, oof_prolif, SEED)

        rng2 = np.random.default_rng(seed=12345)
        shuf_target = rng2.permutation(target)
        oof_prolif_shuf = _proliferation_oof(oof_logits_unimodal, modality_c, shuf_target, folds)
        shuffle_r = float(np.corrcoef(shuf_target, oof_prolif_shuf)[0, 1])
        if not (abs(shuffle_r) < PROLIF_SHUFFLE_ABS_MAX):
            raise AssertionError(
                f"PROLIFERATION SHUFFLE SENTINEL: |r| {abs(shuffle_r):.4f} >= {PROLIF_SHUFFLE_ABS_MAX} "
                "— the auxiliary target leaks; STOP and investigate."
            )
        result["proliferation"] = {
            "pearson_r": pearson_r,
            "ci95": [float(lo), float(hi)],
            "shuffle_r": shuffle_r,
            "n": int(len(target)),
        }

    return result


def _selfcheck_firewall_structural() -> dict:
    """Synthetic-fixture structural firewall proof (no real data). Confirms the subtype design matrix
    has exactly 1 + len(MODALITY_C_GENES) columns and that running the proliferation Ridge does NOT
    perturb the subtype OOF (bit-identical). Backs test_pam50_proliferation_never_feeds_subtype_head (a).
    """
    rng = np.random.default_rng(0)
    n, k = 12, len(MODALITY_C_GENES)
    logits = rng.normal(size=n)
    mc = rng.normal(size=(n, k))
    labels = np.array([0, 1] * (n // 2))
    target = rng.normal(size=n)
    folds = [
        (np.arange(n // 2, n), np.arange(0, n // 2)),
        (np.arange(0, n // 2), np.arange(n // 2, n)),
    ]

    # Real column count from the exact construction the subtype path uses.
    tr = folds[0][0]
    scaler = StandardScaler().fit(mc[tr])
    x_tr = np.hstack([logits[tr].reshape(-1, 1), scaler.transform(mc[tr])])
    subtype_x_cols = int(x_tr.shape[1])

    sub_before = _subtype_fusion_oof(logits, mc, labels, folds)
    _ = _proliferation_oof(logits, mc, target, folds)  # separate estimator — must not touch subtype
    sub_after = _subtype_fusion_oof(logits, mc, labels, folds)

    return {"subtype_x_cols": subtype_x_cols, "bit_identical": bool(np.array_equal(sub_before, sub_after))}


def selfcheck() -> int:
    """Assert the panel disjointness and the structural firewall — with NO data on disk."""
    leaked = set(MODALITY_C_GENES) & _ALL_FORBIDDEN_GENES
    assert not leaked, f"selfcheck: MODALITY_C_GENES leak forbidden/proliferation genes: {leaked}"

    s = _selfcheck_firewall_structural()
    assert s["subtype_x_cols"] == 1 + len(MODALITY_C_GENES), (
        f"selfcheck: subtype X has {s['subtype_x_cols']} cols, expected {1 + len(MODALITY_C_GENES)}"
    )
    assert s["bit_identical"], "selfcheck: synthetic firewall breach — proliferation perturbed subtype"

    print(
        f"[piece-b] selfcheck OK — {len(MODALITY_C_GENES)}-gene panel disjoint from all forbidden "
        f"sets (E7 set(PROLIF_GENES)); firewall structural (subtype X = {s['subtype_x_cols']} cols "
        f"= 1 + {len(MODALITY_C_GENES)}); proliferation on/off leaves subtype OOF bit-identical."
    )
    return 0


def _print_report(m: dict) -> None:
    print("[trackb_fusion_wsi_genomics] === STEP B5: report ===")
    print("--- Piece B: linear-probe TITAN + late-fusion mRNA-genomics bundle (intra-TCGA-BRCA) ---")
    print("IMPORTANT: logistic-regression late-fusion stacking of [frozen TITAN OOF logit, mRNA panel].")
    print("  NOT cross-attention MIL fusion (same honest-framing rationale as trackb_fusion_deltaauc).")
    print(f"WSI-alone (TITAN):  AUROC {m['auroc_u']:.4f} [{m['u_lo']:.4f}, {m['u_hi']:.4f}]  ECE {m['ece_u']:.4f}")
    print(f"Fusion (WSI+mRNA):  AUROC {m['auroc_f']:.4f} [{m['f_lo']:.4f}, {m['f_hi']:.4f}]  ECE {m['ece_f']:.4f}")
    print(f"ΔAUC (fusion - WSI-alone): {m['delta']:+.4f}  95% CI [{m['ci95'][0]:+.4f}, {m['ci95'][1]:+.4f}]  (paired DeLong)")
    print(f"Paired DeLong z: {m['z']:.3f}   p = {m['p']:.4f}   ({'significant p<0.05' if m['p'] < 0.05 else 'not significant p>=0.05'})")
    sentinel_ok = m["shuffle_auc"] < SHUFFLE_SENTINEL_MAX
    print(f"Fusion shuffle AUC: {m['shuffle_auc']:.4f}  (must be < {SHUFFLE_SENTINEL_MAX}: {'OK' if sentinel_ok else 'FAIL'})")
    print(f"N={m['n']} (LumA=475 / Basal=165)")
    print(f"mRNA panel ({m['n_genes']} genes): {' / '.join(m['panel'])}")
    if "proliferation" in m:
        pr = m["proliferation"]
        print("--- Auxiliary co-target: PAM50 proliferation index AT DIAGNOSIS (STRUCTURALLY SEPARATE Ridge) ---")
        print("  A molecular snapshot at diagnosis (Ki-67 surrogate index), NOT fused into the subtype")
        print("  decision and NOT a rate-like readout. Reported separately; never a subtype input feature.")
        print(f"  Pearson r: {pr['pearson_r']:+.4f}  bootstrap 95% CI [{pr['ci95'][0]:+.4f}, {pr['ci95'][1]:+.4f}]  (N={pr['n']})")
        prolif_ok = abs(pr["shuffle_r"]) < PROLIF_SHUFFLE_ABS_MAX
        print(f"  proliferation shuffle |r|: {abs(pr['shuffle_r']):.4f}  (must be < {PROLIF_SHUFFLE_ABS_MAX}: {'OK' if prolif_ok else 'FAIL'})")
    print("FRAMING (LOCK-1): intra-TCGA-BRCA subtype characterisation ONLY. NOT a cross-institution")
    print("  generalisation claim. NOT growth-rate / kinetics / early-detection. WSI-alone is near-ceiling")
    print("  (arm3 0.9646 / UNI2-h 0.9675), so a null/marginal/negative ΔAUC is the EXPECTED, acceptable,")
    print("  honestly-reported outcome — Piece B is NOT gated on beating a bar (pre-registered rule).")
    print("  NON-INDEPENDENCE CAVEAT: Piece B reuses the SAME TITAN embeddings as arm3/arm5 — it is NOT")
    print("  independent evidence of a WSI subtype signal (three readouts of one embedding set, one cohort).")


def _run_ablation_comparison() -> dict:
    """GATA3-circularity probe (PI-approved 04-08-26): run the full pipeline with the 6-gene panel AND
    the 5-gene (GATA3-ablated) panel under IDENTICAL conditions (same WSI TITAN OOF, seed 42, folds,
    N=640, firewall), verify the firewall still holds for the ablated panel, then print the ΔAUC
    comparison + a plain verdict. Reports whatever the number is — a collapse to null is the expected,
    acceptable outcome that confirms circularity. If the ablated firewall fails, STOP (raise).
    """
    print("########## GATA3-CIRCULARITY PROBE — FULL (6-gene) vs ABLATED (5-gene, GATA3 removed) ##########")
    print("\n>>> FULL 6-gene panel <<<")
    full = run(compute_proliferation=True, panel=MODALITY_C_GENES)
    print("\n>>> ABLATED 5-gene panel (GATA3 REMOVED) <<<")
    abl = run(compute_proliferation=True, panel=MODALITY_C_GENES_ABLATED)

    # Firewall MUST still hold for the ablated panel: subtype OOF bit-identical with/without prolif.
    abl_off = run(compute_proliferation=False, panel=MODALITY_C_GENES_ABLATED)
    firewall_ok = bool(
        np.array_equal(
            np.asarray(abl["oof_scores_fusion"], dtype=float),
            np.asarray(abl_off["oof_scores_fusion"], dtype=float),
        )
    )
    if not firewall_ok:
        raise AssertionError(
            "ABLATED FIREWALL BREACH: subtype OOF differs with vs without the proliferation head on the "
            "5-gene panel — LOCK-2 violation; STOP and report, do NOT work around."
        )

    abl_lo, _abl_hi = abl["ci95"]
    drop = full["delta"] - abl["delta"]
    gata3_share = (drop / full["delta"]) if full["delta"] != 0 else float("nan")
    if abl_lo <= 0.0:
        verdict = (
            "CIRCULARITY CONFIRMED — the GATA3-ablated ΔAUC 95% CI crosses 0, so removing GATA3 collapses "
            "the fusion advantage to a non-significant/null result. The original 6-gene win was "
            "substantially GATA3-driven (a luminal-lineage subtype-correlate). The honest null stands."
        )
    else:
        verdict = (
            f"NON-CIRCULAR SIGNAL SURVIVES — the GATA3-ablated ΔAUC stays > 0 with a 95% CI excluding 0, so "
            f"a real driver-gene signal survives GATA3 removal. GATA3 accounts for ~{gata3_share * 100:.0f}% "
            f"of the original 6-gene ΔAUC magnitude (not a leakage finding)."
        )

    print("\n########## COMPARISON (the deliverable) ##########")
    print(f"WSI-alone (TITAN) AUROC:        {full['auroc_u']:.4f} [{full['u_lo']:.4f}, {full['u_hi']:.4f}]  "
          f"(ablated run: {abl['auroc_u']:.4f} — identical TITAN OOF)")
    print(f"6-gene fusion AUROC:            {full['auroc_f']:.4f} [{full['f_lo']:.4f}, {full['f_hi']:.4f}]  ECE {full['ece_f']:.4f}")
    print(f"5-gene (GATA3-ablated) AUROC:   {abl['auroc_f']:.4f} [{abl['f_lo']:.4f}, {abl['f_hi']:.4f}]  ECE {abl['ece_f']:.4f}")
    print(f"6-gene ΔAUC (fusion - WSI):     {full['delta']:+.4f}  95% CI [{full['ci95'][0]:+.4f}, {full['ci95'][1]:+.4f}]  DeLong p={full['p']:.4f}")
    print(f"5-gene ΔAUC (GATA3-ablated):    {abl['delta']:+.4f}  95% CI [{abl['ci95'][0]:+.4f}, {abl['ci95'][1]:+.4f}]  DeLong p={abl['p']:.4f}")
    print(f"GATA3 share of 6-gene ΔAUC:     ~{gata3_share * 100:.0f}%  (Δ drop {drop:+.4f})")
    sent_ok = abl["shuffle_auc"] < SHUFFLE_SENTINEL_MAX
    print(f"5-gene fusion shuffle AUC:      {abl['shuffle_auc']:.4f}  (< {SHUFFLE_SENTINEL_MAX}: {'OK' if sent_ok else 'FAIL'})")
    print(f"5-gene firewall bit-identical:  {firewall_ok}")
    if "proliferation" in abl:
        pr_ok = abs(abl["proliferation"]["shuffle_r"]) < PROLIF_SHUFFLE_ABS_MAX
        print(f"5-gene proliferation shuffle |r|: {abs(abl['proliferation']['shuffle_r']):.4f}  (< {PROLIF_SHUFFLE_ABS_MAX}: {'OK' if pr_ok else 'FAIL'})")
    print(f"\nVERDICT: {verdict}")
    print("FRAMING (LOCK-1): intra-TCGA-BRCA only; whatever the number, it is reported faithfully. A")
    print("  collapse to null on ablation is the expected, acceptable outcome. NOT a cross-institution claim.")
    return {"full": full, "ablated": abl, "firewall_ablated_bit_identical": firewall_ok, "verdict": verdict}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Piece B — WSI(TITAN)+mRNA-genomics late-fusion + PAM50-proliferation firewall"
    )
    ap.add_argument(
        "--selfcheck", action="store_true",
        help="assert panel disjointness + structural firewall with NO data on disk",
    )
    ap.add_argument(
        "--compare-ablation", action="store_true",
        help="GATA3-circularity probe: run full 6-gene vs 5-gene(GATA3-ablated) + print ΔAUC comparison",
    )
    args = ap.parse_args(argv)

    if args.selfcheck:
        return selfcheck()

    if args.compare_ablation:
        _run_ablation_comparison()
        return 0

    m = run(compute_proliferation=True)
    _print_report(m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
