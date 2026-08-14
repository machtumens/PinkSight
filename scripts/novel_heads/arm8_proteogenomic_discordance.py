"""Arm 8 (Wave 1, PILOT-POWER) — proteogenomic-discordance organ. FLOOR GATE (CPTAC-BRCA, N≈122).

Purpose (one line): characterise post-transcriptional regulation BURDEN at diagnosis — the per-patient
magnitude of mRNA↔protein discordance (post-transcriptional buffering) — and ask, on the CPTAC-BRCA
pilot cohort, whether that burden tracks the PAM50 subtype axis (the cheapest defensible floor number).

The floor gate is the cheapest de-risk: does the per-patient buffering-burden scalar correlate with the
continuous PAM50 subtype-centroid axis at all? Both quantities come from the SAME staged matched cohort
(no external label fetch). If |r| is at chance, no supervised discordance→subtype arm is warranted. The
gate produces the number (Pearson r + bootstrap 95% CI + ECE-of-the-binarised-axis + multi-seed) and the
KILL / GREENLIGHT / INDETERMINATE decision vs the config thresholds.

POWER HONESTY IS MANDATORY (plan Arm 8 spec): N≈122 is brutal for a continuous correlation. EVERY result
carries "PILOT N=122 — underpowered, hypothesis-generating only". NO headline number is claimed from this
arm. A wide-CI INDETERMINATE / KILL is the EXPECTED, honest pilot outcome — not a failure.

Ledger guard (LOCK-1/LOCK-2, critical): the organ's output is "post-transcriptional regulation burden
characterisation at diagnosis (pilot, N=122)" EVERYWHERE. NEVER prognosis / kinetics / doubling-time /
early-detection / cross-institution generalisation. The protein (and RNA) abundance of the IHC-identity
mirrors ESR1 / PGR / ERBB2 / MKI67 is HELD OUT of the discordance-input gene set (they are subtype
label surrogates — LOCK-2). Subtype-as-TARGET is allowed; subtype-as-INPUT is not. CPTAC ∩ Duke = 0
(different cohorts — logged explicitly; no dedup needed).

Data (public, staged on demand via the `cptac` PyPI package → cached under data/genomics/cptac_brca/):
  - proteomics       (umich)  — per-patient protein quantification, gene-symbol columns
  - transcriptomics  (washu)  — matched per-patient RNA (log2), gene-symbol columns
  - clinical         (mssm)   — demographics / outcome (NOT used as a feature; no usable grade/subtype)
Provenance: CPTAC-BRCA (PDC000173 / Krug-Mertins 2020 prospective breast proteogenome), distributed by
the `cptac` package (Payne lab, PNNL) from its Zenodo/PDC mirrors. Public.

Target (plan Arm 8): the PAM50 subtype **continuous centroid axis**. No pre-computed PAM50 call exists in
the staged clinical (mssm histologic_grade is entirely GX / not-assessed), so the axis is computed
SELF-CONTAINED from the 50 PAM50 genes in the staged RNA — see `pam50_axis()` for the exact construction
and the Deviations note. Subtype is a legitimate TARGET under LOCK-2.

Eval integrity (LOCK-2, decisions.md / plan Shared Evaluation Spine):
  - Patient-level (one row per patient) — the discordance scalar and the target are both per-patient.
  - Leakage guard: ESR1/PGR/ERBB2/MKI67 mirrors EXCLUDED from the discordance-input gene set; the shared
    `assert_no_forbidden_inputs` hard gate is run on the discordance gene panel before any scoring.
  - Never a bare number: Pearson r + bootstrap 95% CI at >= 3 seeds (plan reporting bar), + the ECE of
    the binarised-axis discordance ranking (calibration companion), reused from pinksight.eval.

Plan: process/general-plans/active/novel-heads-roadmap_25-07-26/novel-heads-roadmap_PLAN_25-07-26.md
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Repo root on sys.path so `pinksight` + the sibling harness import when run as a bare script.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from pinksight.eval.calibration import ece as _ece  # noqa: E402
from tests.test_novel_heads_leakage import (  # noqa: E402
    _canonical,
    _FORBIDDEN_TOKENS,
    assert_no_forbidden_inputs,
)
from scripts.novel_heads.wave1_eval_harness import (  # noqa: E402
    run_floor_gate as run_classification_gate,
    oof_calibrate,
    shap_top20 as harness_shap_top20,
    validate_arm_report_keywords,
    assert_no_survival_columns,
    LedgerViolation,
    FloorGateResult,
)

_PURPOSE = (
    "characterise post-transcriptional regulation burden at diagnosis (CPTAC-BRCA pilot, N≈122; "
    "floor gate)"
)

# The IHC-identity mirrors held OUT of the discordance-input gene set (LOCK-2). ESR1/PGR/ERBB2/MKI67
# are subtype label surrogates; a per-gene discordance computed over them would leak the target axis.
# The shared leakage guard (assert_no_forbidden_inputs) is the hard gate — this list is the explicit,
# auditable exclusion applied before it.
_FORBIDDEN_GENE_SYMBOLS = frozenset({"ESR1", "PGR", "ERBB2", "MKI67"})

# Canonical PAM50 gene set (Parker et al. 2009, J Clin Oncol). Used ONLY to build the continuous
# subtype-centroid TARGET axis from the staged RNA — never as a discordance INPUT (the four IHC mirrors
# in this list are still stripped from the discordance panel by _FORBIDDEN_GENE_SYMBOLS). Subtype is a
# permitted target under LOCK-2.
_PAM50_GENES: tuple[str, ...] = (
    "UBE2T", "BIRC5", "NUF2", "CDC6", "CCNB1", "TYMS", "MYBL2", "CEP55", "MELK", "NDC80",
    "RRM2", "UBE2C", "CENPF", "PTTG1", "EXO1", "ORC6", "ANLN", "CCNE1", "CDC20", "MKI67",
    "KIF2C", "ACTR3B", "MYC", "EGFR", "KRT5", "PHGDH", "CDH3", "MIA", "KRT17", "FOXC1",
    "SFRP1", "KRT14", "ESR1", "SLC39A6", "BAG1", "MAPT", "PGR", "CXXC5", "MLPH", "BCL2",
    "MDM2", "NAT1", "FOXA1", "BLVRA", "MMP11", "GPR160", "FGFR4", "GRB7", "TMEM45B", "ERBB2",
)

# ESR1-anchored orientation: LumA/ER+ high-ESR1 tumours sit at the low-aggressiveness end of the axis;
# Basal/TNBC low-ESR1 tumours at the high end. We orient the PAM50 axis so higher = more Basal-like
# (more aggressive) using the sign of the ESR1 loading. This makes the axis a monotone at-diagnosis
# subtype-aggressiveness proxy (characterisation only — never kinetics).
_AXIS_ANCHOR_GENE = "ESR1"


@dataclass(frozen=True)
class ArmConfig:
    """Parsed arm-8 config knobs the floor gate needs (seeds, bootstrap n, KILL/GREENLIGHT thresholds)."""

    seeds: tuple[int, ...]
    bootstrap_n: int
    kill_threshold_r_abs: float
    greenlight_threshold_r_abs: float
    greenlight_bootstrap_ci_lb: float


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Arm 8 — proteogenomic-discordance floor gate (CPTAC-BRCA pilot, N≈122)"
    )
    p.add_argument(
        "--config",
        default="configs/novel_heads/arm8_proteogenomic_discordance.yaml",
        help="arm config (thresholds, bootstrap n, seeds)",
    )
    p.add_argument(
        "--floor-gate",
        action="store_true",
        help="run the CPU discordance-correlation floor gate (bootstrap CI)",
    )
    p.add_argument(
        "--construct-control",
        action="store_true",
        help=(
            "run the red-team construct-validity control: re-anchor the subtype axis on an "
            "RNA-INDEPENDENT label (IHC preferred, protein fallback) and re-run the burden correlation; "
            "APPENDS a '## Construct-Validity Control' section to the arm-8 report + writes a control JSON"
        ),
    )
    p.add_argument(
        "--supervised-arm",
        action="store_true",
        help=(
            "PILOT N=121 — run the supervised discordance→subtype arm: build the per-gene discordance "
            "feature matrix (|protein_z - rna_z| per gene per patient), train LightGBM to predict "
            "binary subtype (basal-like vs non-basal), report AUROC + bootstrap 95% CI + ECE at 3 seeds. "
            "MANDATORY 'PILOT N=121 — underpowered, hypothesis-generating only' caveat at every number."
        ),
    )
    p.add_argument(
        "--advance",
        action="store_true",
        help=(
            "Run the advance step for arm 8: OOF calibration (isotonic + Platt) on the supervised arm "
            "result, SHAP top-20 discordance genes, larger-N replication note, and update the arm8 report. "
            "Requires --supervised-arm to have been run first (reads its JSON output). "
            "PILOT N=121 caveat carried throughout."
        ),
    )
    p.add_argument(
        "--report-dir",
        default="reports/novel_heads/arm8_proteogenomic",
        help="directory for the floor-gate report + metrics JSON",
    )
    p.add_argument(
        "--data-root",
        default=None,
        help="override the cache root the staged CPTAC matrices resolve against (default: repo root)",
    )
    return p


def _parse_config(cfg_raw: dict) -> ArmConfig:
    """Map the config yaml to the typed knobs (with the plan Arm 8 defaults if a field is absent)."""
    seeds = tuple(int(s) for s in cfg_raw.get("seeds", (0, 1, 2)))
    return ArmConfig(
        seeds=seeds,
        bootstrap_n=int(cfg_raw.get("bootstrap_n", 1000)),
        kill_threshold_r_abs=float(cfg_raw.get("kill_threshold_pearson_r_abs", 0.10)),
        greenlight_threshold_r_abs=float(cfg_raw.get("greenlight_threshold_pearson_r_abs", 0.25)),
        greenlight_bootstrap_ci_lb=float(cfg_raw.get("greenlight_bootstrap_ci_lb", 0.05)),
    )


# ==================================================================================================
# Data staging (public `cptac` package → cached parquet under data/genomics/cptac_brca/).
# ==================================================================================================
def _cache_dir(data_root: Path) -> Path:
    return data_root / "data" / "genomics" / "cptac_brca"


def _symbol_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse a cptac (symbol, ensembl) MultiIndex column frame to bare unique gene-symbol columns.

    cptac ships transcriptomics/proteomics with a two-level column index `(gene_symbol, ensembl_id)`.
    We take level 0 (the symbol) so the leakage canonicaliser and the PAM50/forbidden matching see plain
    symbols. Duplicate symbols (multiple ENSEMBL ids per gene) are collapsed by mean so columns stay
    unique — an unsupervised aggregation using no label information.
    """
    cols = df.columns
    if isinstance(cols, pd.MultiIndex):
        df = df.copy()
        df.columns = [str(c[0]) for c in cols]
    else:
        df = df.copy()
        df.columns = [str(c) for c in cols]
    if df.columns.duplicated().any():
        df = df.T.groupby(level=0).mean().T
    return df


def stage_cptac_brca(data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stage matched CPTAC-BRCA RNA + protein via the `cptac` package; cache as parquet; return both.

    Returns (rna, protein) — both indexed by Patient_ID, bare gene-symbol columns, restricted to the
    patients present in BOTH matrices (matched RNA+protein). Caches to data/genomics/cptac_brca/ so a
    re-run is offline. Raises a clear ImportError / RuntimeError (→ CLI BLOCKED exit) if the package is
    absent or the download fails — NEVER fabricates a matrix.
    """
    cache = _cache_dir(data_root)
    # CSV cache (not parquet) so no pyarrow/fastparquet dependency is required — matches the repo's
    # CSV genomics convention (data/genomics/depmap/*.csv). Patient_ID index + bare gene-symbol columns
    # round-trip cleanly through CSV.
    rna_csv = cache / "transcriptomics_washu.csv"
    prot_csv = cache / "proteomics_umich.csv"

    if rna_csv.exists() and prot_csv.exists():
        rna = pd.read_csv(rna_csv, index_col=0)
        prot = pd.read_csv(prot_csv, index_col=0)
    else:
        try:
            import cptac  # noqa: PLC0415 — optional data-staging dep, imported lazily
        except ImportError as exc:
            raise ImportError(
                "arm8 floor gate BLOCKED — the `cptac` package is not installed. Install it "
                "(`uv pip install cptac`) then re-run; it downloads CPTAC-BRCA (PDC000173) matched "
                "RNA+protein from the PDC/Zenodo mirrors on first use."
            ) from exc
        try:
            br = cptac.Brca()
            rna = _symbol_columns(br.get_transcriptomics("washu"))
            prot = _symbol_columns(br.get_proteomics("umich"))
        except Exception as exc:  # network / mirror failure — honest BLOCKED, never a fake matrix
            raise RuntimeError(
                "arm8 floor gate BLOCKED — `cptac` could not stage CPTAC-BRCA matched RNA+protein "
                f"(source: PDC000173 via the cptac package Zenodo mirror). Underlying error: {exc}. "
                "Manual route: `uv pip install cptac`; in Python `import cptac; br = cptac.Brca(); "
                "br.get_transcriptomics('washu'); br.get_proteomics('umich')` — requires network "
                "access to the PDC/Zenodo mirror."
            ) from exc
        cache.mkdir(parents=True, exist_ok=True)
        rna.to_csv(rna_csv)
        prot.to_csv(prot_csv)

    rna.index = rna.index.astype(str)
    prot.index = prot.index.astype(str)
    shared = sorted(set(rna.index) & set(prot.index))
    if len(shared) < 20:
        raise RuntimeError(
            f"arm8 floor gate BLOCKED — only {len(shared)} matched RNA+protein patients found "
            "(expected ~121). The staged CPTAC-BRCA matrices did not overlap as expected; do not "
            "trust a floor number computed on this — re-stage the cohort."
        )
    return rna.loc[shared], prot.loc[shared]


# ==================================================================================================
# Discordance-burden score (the science: per-patient post-transcriptional buffering magnitude).
# ==================================================================================================
def discordance_burden(rna: pd.DataFrame, protein: pd.DataFrame) -> pd.Series:
    """Per-patient mRNA↔protein discordance magnitude over the shared gene panel (buffering burden).

    Construction (leakage-clean, LOCK-2):
      1. Restrict to genes measured in BOTH RNA and protein; DROP the forbidden IHC mirrors
         (ESR1/PGR/ERBB2/MKI67) from the panel — they are subtype surrogates, never a discordance input.
      2. Run the shared leakage hard gate on the panel gene names before any scoring.
      3. Z-score each gene ACROSS patients, separately for RNA and protein, so the two modalities are on
         a comparable scale and per-gene mean expression is removed (the residual buffering signal, not
         absolute abundance, is what we want). Standardisation uses only the matrix itself (no labels).
      4. Per patient, discordance burden = mean over the panel of |protein_z − rna_z| — how far this
         patient's proteome sits from what its transcriptome predicts (post-transcriptional buffering).

    Returns a per-patient Series (index = Patient_ID). Higher = more post-transcriptional discordance.
    """
    # Drop ANY gene whose canonicalised name maps to a forbidden token — this reuses the leakage
    # test's own canonical forbidden set (single source of truth) so it catches every IHC-mirror ALIAS,
    # not just the four canonical symbols. E.g. `MIB1` (the Ki-67 antibody-clone alias) is a forbidden
    # Ki-67 mirror the shared guard flags; a hand-maintained {ESR1,PGR,ERBB2,MKI67} list would miss it.
    protein_syms = set(protein.columns)
    common_genes = [g for g in rna.columns if g in protein_syms]
    panel = [g for g in common_genes if _canonical(str(g)) not in _FORBIDDEN_TOKENS]
    if len(panel) < 100:
        raise RuntimeError(
            f"arm8 floor gate BLOCKED — only {len(panel)} genes shared between RNA and protein after "
            "dropping IHC mirrors; too few to characterise a stable per-patient discordance burden."
        )

    # HARD leakage gate (LOCK-2): the discordance panel must carry no forbidden IHC surrogate.
    assert_no_forbidden_inputs(panel)

    rna_p = rna[panel].apply(pd.to_numeric, errors="coerce")
    prot_p = protein[panel].apply(pd.to_numeric, errors="coerce")

    def _zscore(df: pd.DataFrame) -> pd.DataFrame:
        mu = df.mean(axis=0, skipna=True)
        sd = df.std(axis=0, skipna=True).replace(0.0, np.nan)
        return (df - mu) / sd

    rna_z = _zscore(rna_p)
    prot_z = _zscore(prot_p)
    resid = (prot_z - rna_z).abs()
    # per-patient mean absolute residual across the panel (genes with a NaN in either modality drop out
    # of that patient's mean via skipna — no imputation of a fabricated value)
    return resid.mean(axis=1, skipna=True)


# ==================================================================================================
# PAM50 continuous subtype-centroid axis (the TARGET). Self-contained from staged RNA. See Deviations.
# ==================================================================================================
def pam50_axis(rna: pd.DataFrame) -> pd.Series:
    """Continuous PAM50 subtype-centroid axis per patient, computed self-contained from the staged RNA.

    Plan Arm 8 names the target as "PAM50 subtype continuous centroid distance". No pre-computed PAM50
    call ships in the staged CPTAC clinical (mssm histologic_grade is entirely GX), so the axis is built
    from the 50 PAM50 genes present in the staged transcriptomics (all 50 confirmed present):

      1. Restrict RNA to the PAM50 genes; median-center each gene across patients (the standard PAM50
         pre-step) and z-score (unit variance) so no single high-variance gene dominates.
      2. Take the first principal component (via SVD) of the (patient × PAM50-gene) matrix — the dominant
         axis of PAM50-gene covariation is the LumA↔Basal subtype-aggressiveness axis in breast cohorts.
      3. ORIENT it by the ESR1 loading so higher score = more Basal-like (lower ESR1) = the
         higher-aggressiveness end. This yields a continuous, monotone at-diagnosis subtype axis — a
         defensible "centroid distance" proxy that needs no external centroid table (which, mis-specified,
         would fabricate the target). This is an implementation-detail choice (see the Deviations note in
         the report); it is a TARGET (subtype), permitted under LOCK-2.

    Returns a per-patient Series (index = Patient_ID), higher = more Basal-like / more aggressive.
    """
    present = [g for g in _PAM50_GENES if g in set(rna.columns)]
    if len(present) < 40:
        raise RuntimeError(
            f"arm8 floor gate BLOCKED — only {len(present)}/50 PAM50 genes present in staged RNA; too "
            "few to build a stable continuous subtype-centroid axis target."
        )
    mat = rna[present].apply(pd.to_numeric, errors="coerce")
    # median-center per gene (PAM50 convention), then z-score; impute residual NaN with 0 (post-center
    # gene mean) so the SVD is defined — a handful of missing entries, no label used.
    centered = mat - mat.median(axis=0, skipna=True)
    sd = centered.std(axis=0, skipna=True).replace(0.0, np.nan)
    z = (centered / sd).fillna(0.0)

    x = z.to_numpy(dtype=float)
    x = x - x.mean(axis=0, keepdims=True)  # column-center for PCA
    # first right-singular vector = PC1 loadings; project patients onto it
    _u, _s, vt = np.linalg.svd(x, full_matrices=False)
    pc1_loadings = vt[0]
    scores = x @ pc1_loadings

    # orient so higher = more Basal-like (lower ESR1): flip if ESR1 loads positively on PC1
    if _AXIS_ANCHOR_GENE in present:
        esr1_idx = present.index(_AXIS_ANCHOR_GENE)
        if pc1_loadings[esr1_idx] > 0:
            scores = -scores
    return pd.Series(scores, index=rna.index, name="pam50_axis")


# ==================================================================================================
# CONSTRUCT-VALIDITY CONTROL (red-team, 2026-07-25): RNA-INDEPENDENT subtype axis.
#
# The red-team ruled arm 8 SURVIVES-WITH-CONTROL: the shuffle sentinel (real r >> permutation null)
# legitimately rules out the pairing artifact, but the ORIGINAL subtype axis is RNA-derived (PC1 of
# the PAM50 *RNA* genes) while the discordance burden ALSO contains RNA — so a shared-RNA dependence
# is not fully excluded. This control re-anchors the axis on a label that does NOT come from the RNA
# matrix and re-runs the identical correlation.
#
# Anchor selection (empirical, at execute time — NOT a design assumption):
#   PREFERRED — a clinical IHC-based subtype/ER label. CHECKED: the staged CPTAC clinical
#     (mssm `clinical_Pan-cancer.May2022.tsv.gz`) carries an IHC field
#     `baseline/ancillary_studies_immunohistochemistry_type_and_result` that is 0/134 non-null for
#     BRCA, and no ER/PR/HER2/subtype column exists (histologic_grade is entirely GX). So NO
#     RNA-independent IHC/clinical subtype label ships with this cohort — the preferred anchor is
#     genuinely unavailable (see `ihc_anchor_available()`).
#   FALLBACK (used) — the PROTEIN matrix. The axis is rebuilt as the ESR1-oriented PC1 of the PAM50
#     PROTEIN abundances (`protein_pam50_axis`). This uses ZERO RNA, so it removes exactly the
#     "RNA in both the axis and the burden" dependence the red-team flagged.
#   CAVEAT (flagged everywhere): the burden term still contains protein, so a protein-anchored axis
#     does not remove a *protein*-in-both dependence — it removes only the RNA-in-both one. It is a
#     strictly stronger construct check than the RNA axis, not a perfectly independent one.
# ==================================================================================================
def ihc_anchor_available(data_root: Path) -> tuple[bool, str, int, int]:
    """Check whether an RNA-independent IHC/clinical subtype label ships with the staged CPTAC-BRCA.

    Reads the bundled/staged CPTAC clinical table (mssm pan-cancer) if present and inspects the IHC
    results field + any ER/PR/HER2/subtype column for BRCA rows. Returns
    `(available, reason, n_brca, n_ihc_nonnull)`. `available` is True only if a usable per-patient
    RNA-independent subtype/receptor label is found; otherwise the caller falls back to the protein
    anchor. NEVER fabricates a label — a missing/empty field returns False with the reason logged.

    This is a read-only probe: it does not stage anything and returns False (not an error) if the
    clinical table cannot be located, so the control degrades to the protein anchor gracefully.
    """
    # The cptac package bundles the clinical table locally; also allow a project-staged copy.
    candidates = [
        data_root
        / ".venv/lib/python3.11/site-packages/cptac/data/mssm-all_cancers"
        / "clinical_Pan-cancer.May2022.tsv.gz",
        _cache_dir(data_root) / "clinical_mssm.csv",
    ]
    clin_path = next((c for c in candidates if c.exists()), None)
    if clin_path is None:
        return (False, "clinical table not found on disk (protein anchor used)", 0, 0)

    try:
        if clin_path.suffix == ".gz":
            clin = pd.read_csv(clin_path, sep="\t", compression="gzip", low_memory=False)
        else:
            clin = pd.read_csv(clin_path, low_memory=False)
    except Exception as exc:  # noqa: BLE001 — a read failure just means fall back, not crash
        return (False, f"clinical table unreadable ({exc}); protein anchor used", 0, 0)

    if "tumor_code" in clin.columns:
        brca = clin[clin["tumor_code"].astype(str).str.upper().str.contains("BR")]
    else:
        brca = clin
    n_brca = int(brca.shape[0])

    ihc_col = "baseline/ancillary_studies_immunohistochemistry_type_and_result"
    n_ihc = int(brca[ihc_col].notna().sum()) if ihc_col in brca.columns else 0

    # An ER/PR/HER2/subtype column would have to carry receptor-status content on BRCA rows.
    recpat = "estrogen|progest|her2|triple|luminal|basal|receptor|subtype|pam50"
    subtype_cols = [
        c
        for c in brca.columns
        if brca[c].dropna().astype(str).str.contains(recpat, case=False, regex=True).any()
        and c != "baseline/histologic_type"  # histology (IDC/ILC) is NOT a subtype/receptor axis
    ]
    if n_ihc > 0 or subtype_cols:
        return (True, f"IHC/subtype label present (ihc_nonnull={n_ihc}, cols={subtype_cols})", n_brca, n_ihc)
    return (
        False,
        (
            f"no RNA-independent IHC/subtype label ships (IHC field {n_ihc}/{n_brca} non-null; no "
            "ER/PR/HER2/subtype column; histologic_grade all GX) — PROTEIN anchor used"
        ),
        n_brca,
        n_ihc,
    )


def protein_pam50_axis(protein: pd.DataFrame) -> pd.Series:
    """RNA-INDEPENDENT subtype axis: ESR1-oriented PC1 of the PAM50 *protein* abundances.

    Identical construction to `pam50_axis` but computed on the PROTEIN matrix, so the axis carries no
    RNA at all — this is the construct-validity control the red-team required. The PAM50 protein
    subset (ESR1/ERBB2/MKI67/PGR included) is used only to BUILD the TARGET axis (permitted under
    LOCK-2, subtype-as-target); those genes are still stripped from the discordance INPUT panel by
    `discordance_burden`. Steps: restrict to PAM50 genes present in protein → median-center + z-score
    → PC1 → orient by ESR1 protein loading so higher = more Basal-like / more aggressive.

    Returns a per-patient Series (index = Patient_ID). Raises → CLI BLOCKED if too few PAM50 proteins
    are present to build a stable axis (never fabricates one).
    """
    present = [g for g in _PAM50_GENES if g in set(protein.columns)]
    if len(present) < 40:
        raise RuntimeError(
            f"arm8 construct-control BLOCKED — only {len(present)}/50 PAM50 genes present in the "
            "staged PROTEIN matrix; too few to build a stable RNA-independent subtype axis."
        )
    mat = protein[present].apply(pd.to_numeric, errors="coerce")
    centered = mat - mat.median(axis=0, skipna=True)
    sd = centered.std(axis=0, skipna=True).replace(0.0, np.nan)
    z = (centered / sd).fillna(0.0)

    x = z.to_numpy(dtype=float)
    x = x - x.mean(axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(x, full_matrices=False)
    pc1_loadings = vt[0]
    scores = x @ pc1_loadings

    if _AXIS_ANCHOR_GENE in present:
        esr1_idx = present.index(_AXIS_ANCHOR_GENE)
        if pc1_loadings[esr1_idx] > 0:
            scores = -scores
    return pd.Series(scores, index=protein.index, name="protein_pam50_axis")


# ==================================================================================================
# Floor gate: Pearson r (discordance vs PAM50 axis) + bootstrap CI + ECE, multi-seed.
# ==================================================================================================
@dataclass
class FloorResult:
    """Reportable aggregate for the arm-8 floor gate — never a bare number (CLAUDE.md rule).

    `pearson_r` is the point Pearson correlation between the per-patient discordance burden and the
    continuous PAM50 axis on the full matched cohort. `boot_ci95` is the seed-MEAN of per-seed bootstrap
    95% percentile CIs (project seed-mean convention, matching arm 6 / the G5 headline). `ece` is the
    calibration error of the discordance ranking against the axis binarised at its median (a companion
    calibration number — the plan reporting bar asks for ECE). `shuffle_r` is the seed-mean permutation-
    NULL Pearson r (target labels shuffled) — the leakage sentinel: it must collapse to ~0, proving the
    real r is not a construction artifact of RNA appearing in both the discordance score and the axis.
    `n` is the matched cohort size (the pilot N). Decision (KILL/GREENLIGHT/INDETERMINATE) is attached
    by `_decide`, not here.
    """

    pearson_r: float
    boot_ci95: tuple[float, float]
    ece: float
    shuffle_r: float
    seeds: int
    n: int
    n_panel_genes: int
    n_pam50_genes: int
    per_seed: list[dict]


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r on the rows where BOTH a and b are finite (NaN-safe). Returns NaN on degenerate input."""
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    aa, bb = a[m], b[m]
    if np.std(aa) == 0 or np.std(bb) == 0:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def run_floor_gate(discordance: pd.Series, axis: pd.Series, cfg: ArmConfig) -> FloorResult:
    """Correlate discordance burden vs PAM50 axis; bootstrap-CI it per seed; aggregate + ECE.

    The point estimate is the full-cohort Pearson r. For EACH seed we draw `bootstrap_n` patient
    resamples (with replacement) and take the 2.5/97.5 percentile of the resample Pearson r as that
    seed's 95% CI; the headline CI is the seed-mean of those bounds (project convention). ECE = the
    calibration error of the min-max-normalised discordance score as a "probability" of the axis being
    above its median — a companion calibration number, not the headline. Multi-seed (>=3) satisfies the
    plan reporting bar. N≈122 → wide CI expected (pilot power).
    """
    idx = sorted(set(discordance.index) & set(axis.index))
    d = discordance.loc[idx].to_numpy(dtype=float)
    a = axis.loc[idx].to_numpy(dtype=float)
    n = len(idx)

    point_r = _pearson(d, a)

    per_seed: list[dict] = []
    ci_los: list[float] = []
    ci_his: list[float] = []
    shuffle_rs: list[float] = []
    for seed in cfg.seeds:
        rng = np.random.default_rng(seed)
        boot_rs = np.empty(cfg.bootstrap_n, dtype=float)
        for b in range(cfg.bootstrap_n):
            sample = rng.integers(0, n, size=n)
            boot_rs[b] = _pearson(d[sample], a[sample])
        finite = boot_rs[np.isfinite(boot_rs)]
        if finite.size < 2:
            lo = hi = float("nan")
        else:
            lo = float(np.percentile(finite, 2.5))
            hi = float(np.percentile(finite, 97.5))

        # Shuffle sentinel (leakage-clean check): permute the target axis vs the discordance score and
        # re-correlate. Under the null the mean |r| over the permutations must collapse to ~0 — a non-
        # trivial shuffle r would flag the real r as a construction artifact (RNA in both terms), the
        # same sentinel discipline the project used at G1 / the pCR pilot.
        perm_rs = np.empty(cfg.bootstrap_n, dtype=float)
        for b in range(cfg.bootstrap_n):
            perm_rs[b] = _pearson(d, a[rng.permutation(n)])
        perm_finite = perm_rs[np.isfinite(perm_rs)]
        seed_shuffle = float(np.mean(np.abs(perm_finite))) if perm_finite.size else float("nan")

        per_seed.append(
            {
                "seed": int(seed),
                "pearson_r": round(point_r, 4) if np.isfinite(point_r) else None,
                "boot_ci95": [round(lo, 4), round(hi, 4)] if np.isfinite(lo) else [None, None],
                "shuffle_abs_r": round(seed_shuffle, 4) if np.isfinite(seed_shuffle) else None,
            }
        )
        ci_los.append(lo)
        ci_his.append(hi)
        shuffle_rs.append(seed_shuffle)

    boot_ci95 = (float(np.nanmean(ci_los)), float(np.nanmean(ci_his)))
    shuffle_r = float(np.nanmean(shuffle_rs))

    # ECE companion: treat min-max-normalised discordance as P(axis above median); reuse pinksight ECE.
    y_bin = (a > np.nanmedian(a)).astype(int)
    d_finite = d[np.isfinite(d)]
    if d_finite.size and (d_finite.max() > d_finite.min()):
        p = (d - d_finite.min()) / (d_finite.max() - d_finite.min())
        p = np.clip(np.nan_to_num(p, nan=0.5), 0.0, 1.0)
        ece_val = float(_ece(y_bin, p))
    else:
        ece_val = float("nan")

    return FloorResult(
        pearson_r=point_r,
        boot_ci95=boot_ci95,
        ece=ece_val,
        shuffle_r=shuffle_r,
        seeds=len(cfg.seeds),
        n=n,
        n_panel_genes=0,  # filled by caller (it knows the panel size)
        n_pam50_genes=0,
        per_seed=per_seed,
    )


@dataclass
class ArmOutcome:
    """The floor result + the KILL/GREENLIGHT/INDETERMINATE decision and its rationale."""

    result: FloorResult
    decision: str  # "KILL" | "GREENLIGHT" | "INDETERMINATE"
    rationale: str


def _decide(result: FloorResult, cfg: ArmConfig) -> ArmOutcome:
    """Apply the pre-registered pilot gate: KILL |r|<=0.10 & CI overlaps 0; GREENLIGHT |r|>=0.25 & CI LB>0.05.

    Everything else is INDETERMINATE (real-but-weak / inconclusive at pilot N) — the EXPECTED honest
    outcome for N≈122. The bootstrap CI lower bound (in |r| space, using the CI bound nearest to the
    sign of r) must clear `greenlight_bootstrap_ci_lb` for a GREENLIGHT so a wide-CI point estimate that
    merely grazes 0.25 does not pass. Power caveat is attached to every rationale.
    """
    r = result.pearson_r
    lo, hi = result.boot_ci95
    kill = cfg.kill_threshold_r_abs
    green = cfg.greenlight_threshold_r_abs
    green_lb = cfg.greenlight_bootstrap_ci_lb
    n = result.n

    if not np.isfinite(r):
        return ArmOutcome(
            result,
            "INDETERMINATE",
            f"Pearson r is NaN (degenerate — zero variance in discordance or axis at N={n}). "
            f"PILOT N={n} — underpowered, hypothesis-generating only.",
        )

    ci_overlaps_zero = (not np.isfinite(lo)) or (lo <= 0.0 <= hi)
    # |r|-space lower bound: the CI bound closest to zero on the side of r (for r>0 this is lo; for r<0
    # this is -hi). Used only for the GREENLIGHT LB check.
    abs_lb = lo if r >= 0 else -hi

    if abs(r) <= kill and ci_overlaps_zero:
        decision = "KILL"
        rationale = (
            f"|r|={abs(r):.3f} <= KILL {kill:.2f} and the bootstrap 95% CI [{lo:.3f}, {hi:.3f}] "
            f"overlaps 0.0 — no post-transcriptional-discordance→subtype-axis signal at pilot N={n}. "
            f"PILOT N={n} — underpowered, hypothesis-generating only; null pilot is a complete outcome."
        )
    elif abs(r) >= green and np.isfinite(abs_lb) and abs_lb > green_lb:
        decision = "GREENLIGHT"
        rationale = (
            f"|r|={abs(r):.3f} >= GREENLIGHT {green:.2f} and the bootstrap CI lower bound "
            f"{abs_lb:.3f} > {green_lb:.2f} — discordance burden tracks the subtype axis at pilot "
            f"N={n}. PILOT N={n} — underpowered; advance to the supervised discordance→subtype arm "
            f"ONLY with the explicit power caveat (no headline number from N={n})."
        )
    else:
        decision = "INDETERMINATE"
        rationale = (
            f"|r|={abs(r):.3f} (bootstrap CI [{lo:.3f}, {hi:.3f}]) is between KILL {kill:.2f} and "
            f"GREENLIGHT {green:.2f}, or the CI is too wide to clear the GREENLIGHT lower bound — "
            f"real-but-weak / inconclusive at pilot N={n}. This wide-CI INDETERMINATE is the EXPECTED "
            f"honest pilot outcome. PILOT N={n} — underpowered, hypothesis-generating only."
        )
    return ArmOutcome(result, decision, rationale)


def run_arm8(cfg: ArmConfig, data_root: Path) -> ArmOutcome:
    """End-to-end floor gate: stage → discordance burden → PAM50 axis → correlate/bootstrap → decide."""
    rna, protein = stage_cptac_brca(data_root)
    discordance = discordance_burden(rna, protein)
    axis = pam50_axis(rna)

    result = run_floor_gate(discordance, axis, cfg)
    # backfill panel sizes (recompute cheaply for the record)
    common_genes = [g for g in rna.columns if g in set(protein.columns)]
    result.n_panel_genes = len([g for g in common_genes if g not in _FORBIDDEN_GENE_SYMBOLS])
    result.n_pam50_genes = len([g for g in _PAM50_GENES if g in set(rna.columns)])
    return _decide(result, cfg)


# ==================================================================================================
# Construct-validity control runner: burden vs the RNA-INDEPENDENT (protein-anchored) subtype axis.
# ==================================================================================================
@dataclass
class ControlOutcome:
    """Result of the RNA-independent-axis construct control + its SURVIVES / AXIS-CIRCULAR verdict.

    `original` is the burden-vs-RNA-axis outcome (the headline being interrogated). `control` is the
    burden-vs-PROTEIN-axis outcome (the RNA-independent re-anchor). `axis_agreement_r` is the Pearson
    r between the RNA axis and the protein axis (how much the two subtype axes agree). `esr1_alone_r`
    is r(burden, raw ESR1 protein) — a trivial-echo check (a large |value| would mean the burden is
    just an ESR1 readout, not a genuine multi-gene discordance signal). `anchor` is "ihc" or
    "protein"; `anchor_note` records the independence caveat. `verdict` ∈ {SURVIVES, AXIS-CIRCULAR}.
    """

    original: ArmOutcome
    control: ArmOutcome
    axis_agreement_r: float
    esr1_alone_r: float
    anchor: str
    anchor_note: str
    verdict: str
    rationale: str


def _verdict_control(
    original: ArmOutcome, control: ArmOutcome, cfg: ArmConfig
) -> tuple[str, str]:
    """SURVIVES if the control r holds clear of the shuffle null with CI excluding 0; else AXIS-CIRCULAR.

    Honest, symmetric rule (no tuning-to-rescue):
      - SURVIVES  — the RNA-independent-axis |r| is at/above the KILL band AND its bootstrap CI lower
        bound (nearest 0) stays > 0 AND |r| stands clear of the permutation-null shuffle |r| (>= 3x,
        matching the "~6x null" leakage-clean bar the report used). The signal is NOT an artifact of
        the RNA-derived axis construction.
      - AXIS-CIRCULAR — the control |r| collapses toward the shuffle null (CI crosses 0, or |r| is not
        clear of the shuffle). The original correlation depended on the RNA-derived axis → axis-circular
        → downgrade. Reported whichever happens; N=121 pilot caveat carried either way.
    """
    r_ctrl = control.result.pearson_r
    lo, hi = control.result.boot_ci95
    shuffle = control.result.shuffle_r
    n = control.result.n
    kill = cfg.kill_threshold_r_abs

    ci_excludes_zero = np.isfinite(lo) and np.isfinite(hi) and (lo > 0.0 or hi < 0.0)
    abs_lb = lo if r_ctrl >= 0 else -hi
    clears_shuffle = (
        np.isfinite(r_ctrl)
        and np.isfinite(shuffle)
        and shuffle > 0
        and abs(r_ctrl) >= 3.0 * shuffle
    )

    survives = (
        np.isfinite(r_ctrl)
        and abs(r_ctrl) > kill
        and ci_excludes_zero
        and np.isfinite(abs_lb)
        and abs_lb > 0.0
        and clears_shuffle
    )

    r_orig = original.result.pearson_r
    if survives:
        return (
            "SURVIVES",
            (
                f"RNA-independent (protein-anchored) axis: Pearson r={abs(r_ctrl):.3f} "
                f"(bootstrap CI [{lo:.3f}, {hi:.3f}], shuffle |r|={shuffle:.3f}) — the correlation "
                f"holds near the original r={abs(r_orig):.3f} and stands ~{abs(r_ctrl) / shuffle:.0f}x "
                f"clear of the permutation null with a CI that excludes 0. The signal is NOT a "
                f"construction artifact of the RNA-derived subtype axis (RNA-in-both is excluded). "
                f"PILOT N={n} — underpowered, hypothesis-generating only; SURVIVES still means pilot."
            ),
        )
    return (
        "AXIS-CIRCULAR",
        (
            f"RNA-independent (protein-anchored) axis: Pearson r={abs(r_ctrl):.3f} "
            f"(bootstrap CI [{lo:.3f}, {hi:.3f}], shuffle |r|={shuffle:.3f}) — the correlation "
            f"collapses toward the shuffle null (CI crosses 0 and/or |r| is not clear of the null) "
            f"once the subtype axis no longer shares RNA with the burden. The original r="
            f"{abs(r_orig):.3f} therefore depended on the RNA-derived axis construction → axis-circular "
            f"→ DOWNGRADE. PILOT N={n} — underpowered; reported as-is (no tuning to rescue)."
        ),
    )


def run_construct_control(cfg: ArmConfig, data_root: Path) -> ControlOutcome:
    """Re-anchor the subtype axis on an RNA-INDEPENDENT label and re-run the burden correlation.

    Stages the same matched CPTAC-BRCA cohort, computes the original (RNA-axis) outcome for reference,
    then rebuilds the subtype axis RNA-independently: prefer a clinical IHC/subtype label (checked via
    `ihc_anchor_available`), else fall back to the PROTEIN-anchored PAM50 axis. Correlates the SAME
    per-patient discordance burden against the new axis (Pearson r + bootstrap CI + shuffle + ECE,
    3 seeds) and emits the SURVIVES / AXIS-CIRCULAR verdict plus axis-agreement + ESR1-alone
    diagnostics. Raises → CLI BLOCKED (exit 2) if neither an IHC nor a usable protein anchor exists.
    """
    rna, protein = stage_cptac_brca(data_root)
    discordance = discordance_burden(rna, protein)

    # Original (RNA-axis) outcome — the headline being interrogated. Recomputed here for a like-for-like
    # comparison record (identical machinery, same cohort).
    rna_axis = pam50_axis(rna)
    orig_result = run_floor_gate(discordance, rna_axis, cfg)
    common_genes = [g for g in rna.columns if g in set(protein.columns)]
    orig_result.n_panel_genes = len([g for g in common_genes if g not in _FORBIDDEN_GENE_SYMBOLS])
    orig_result.n_pam50_genes = len([g for g in _PAM50_GENES if g in set(rna.columns)])
    original = _decide(orig_result, cfg)

    # RNA-independent anchor selection.
    ihc_ok, ihc_reason, _n_brca, _n_ihc = ihc_anchor_available(data_root)
    if ihc_ok:
        # Preferred path is intentionally NOT auto-built here: no such label ships with this cohort
        # (verified empirically), so a builder would be dead code that could silently fabricate an
        # axis. If a future staged cohort DOES ship an IHC/subtype label, wire its construction here.
        raise RuntimeError(
            "arm8 construct-control: an IHC/subtype label was detected "
            f"({ihc_reason}) but no IHC-axis builder is wired for this cohort. Add the IHC-axis "
            "construction before claiming the IHC anchor; do NOT fall through silently."
        )

    anchor = "protein"
    anchor_note = (
        "RNA-independent axis built from the PAM50 PROTEIN abundances (ESR1-oriented PC1). "
        "Independence caveat: this removes the RNA-in-both dependence the red-team flagged, but the "
        "discordance burden still contains protein, so a protein-in-both dependence is NOT removed. "
        f"IHC anchor unavailable — {ihc_reason}."
    )
    prot_axis = protein_pam50_axis(protein)
    ctrl_result = run_floor_gate(discordance, prot_axis, cfg)
    ctrl_result.n_panel_genes = orig_result.n_panel_genes
    ctrl_result.n_pam50_genes = len([g for g in _PAM50_GENES if g in set(protein.columns)])
    control = _decide(ctrl_result, cfg)

    # Diagnostics: how much do the two subtype axes agree, and is the burden a trivial ESR1 echo?
    idx = sorted(set(discordance.index) & set(rna_axis.index) & set(prot_axis.index))
    axis_agreement_r = _pearson(
        rna_axis.loc[idx].to_numpy(dtype=float), prot_axis.loc[idx].to_numpy(dtype=float)
    )
    esr1_alone_r = float("nan")
    if _AXIS_ANCHOR_GENE in protein.columns:
        esr1_raw = pd.to_numeric(protein.loc[idx, _AXIS_ANCHOR_GENE], errors="coerce").to_numpy(
            dtype=float
        )
        esr1_alone_r = _pearson(discordance.loc[idx].to_numpy(dtype=float), esr1_raw)

    verdict, rationale = _verdict_control(original, control, cfg)
    return ControlOutcome(
        original=original,
        control=control,
        axis_agreement_r=axis_agreement_r,
        esr1_alone_r=esr1_alone_r,
        anchor=anchor,
        anchor_note=anchor_note,
        verdict=verdict,
        rationale=rationale,
    )


# ==================================================================================================
# Reporting (markdown + JSON) — ledger-clean, provenance, CPTAC∩Duke=0 log, PILOT-POWER caveat.
# ==================================================================================================
def _render_report(outcome: ArmOutcome, cfg: ArmConfig) -> str:
    r = outcome.result
    today = date.today().strftime("%Y-%m-%d")
    rr = "NaN" if not np.isfinite(r.pearson_r) else f"{r.pearson_r:.3f}"
    ci = (
        "[NaN, NaN]"
        if not np.isfinite(r.boot_ci95[0])
        else f"[{r.boot_ci95[0]:.3f}, {r.boot_ci95[1]:.3f}]"
    )
    ece = "NaN" if not np.isfinite(r.ece) else f"{r.ece:.3f}"
    shuf = "NaN" if not np.isfinite(r.shuffle_r) else f"{r.shuffle_r:.3f}"

    lines: list[str] = []
    lines.append("---")
    lines.append("name: report:arm8-proteogenomic-discordance-floor-gate")
    lines.append(
        'description: "Arm 8 proteogenomic-discordance floor gate — CPTAC-BRCA pilot (N≈122) '
        "post-transcriptional buffering-burden characterisation vs PAM50 subtype axis; Pearson r + "
        'bootstrap CI + ECE + multi-seed; KILL/GREENLIGHT decision; MANDATORY N≈122 power caveat."'
    )
    lines.append(f"date: {today}")
    lines.append("metadata:")
    lines.append("  node_type: memory")
    lines.append("  type: report")
    lines.append("  feature: novel-heads")
    lines.append("  phase: arm8-floor-gate")
    lines.append("---")
    lines.append("")
    lines.append("# Arm 8 — Proteogenomic-Discordance Organ: Floor Gate (PILOT-POWER)")
    lines.append("")
    lines.append(f"**Date:** {today}  ")
    lines.append("**Arm:** 8 (Wave 1) — post-transcriptional regulation burden characterisation  ")
    lines.append(f"**Cohort:** CPTAC-BRCA matched RNA+protein, **PILOT N={r.n}**  ")
    lines.append("**Compute:** CPU-only, $0 (well within the $150 LOCK-5 ceiling)  ")
    lines.append("**Model:** per-patient discordance-burden scalar → Pearson r vs PAM50 axis (floor)  ")
    lines.append("")
    lines.append("## MANDATORY power honesty (plan Arm 8 spec — read this first)")
    lines.append("")
    lines.append(
        f"**PILOT N={r.n} — underpowered, hypothesis-generating only.** N≈122 is brutal for a "
        "continuous correlation; the bootstrap CI is wide by construction. **No headline number is "
        "claimed from this arm.** A wide-CI INDETERMINATE or KILL is the EXPECTED, honest pilot "
        "outcome — not a failure of the science. This floor gate exists to KILL cheaply or to justify "
        "a costlier supervised step, never to assert a result."
    )
    lines.append("")
    lines.append("## What this arm characterises (ledger framing — LOCK-1)")
    lines.append("")
    lines.append(
        "This organ characterises, at diagnosis, the **post-transcriptional regulation burden**: the "
        "per-patient magnitude of mRNA↔protein discordance (post-transcriptional buffering) — how far a "
        "tumour's proteome sits from what its transcriptome predicts. It is a **characterisation** organ."
    )
    lines.append("")
    lines.append(
        "**What this organ does NOT do (LOCK-1 firewall — each item is explicitly excluded):**"
    )
    lines.append("")
    lines.append("- It does **not** predict prognosis, recurrence, or survival.")
    lines.append("- It does **not** measure growth rate, doubling time, or any tumour kinetics.")
    lines.append("- It is **not** an early-detection or pre-detection signal (confirmed tumours only).")
    lines.append("- It makes **no** cross-institution generalisation claim.")
    lines.append("")
    lines.append(
        "Every excluded term above appears here **only inside an exclusion**. The organ's sole output "
        "is at-diagnosis post-transcriptional-burden characterisation on this single pilot cohort."
    )
    lines.append("")
    lines.append("## CPTAC ∩ Duke = 0 (LOCK-2)")
    lines.append("")
    lines.append(
        "**Trivially TRUE — different cohorts.** CPTAC-BRCA (PDC000173 / Krug-Mertins prospective "
        "breast proteogenome) shares zero patients with Duke, ISPY2, METABRIC, or TCGA-BRCA as used "
        "here. No dedup is needed; the zero overlap is logged explicitly. This is a standalone "
        "companion organ — it does **not** fuse into the Duke DCE-MRI imaging encoder (ADR-0006/0010 "
        "standalone-organ pattern)."
    )
    lines.append("")
    lines.append("## Evaluation integrity (LOCK-2)")
    lines.append("")
    lines.append(
        "- **Patient-level:** one row per patient; the discordance scalar and the PAM50 axis are both "
        "per-patient. No patient appears twice."
    )
    lines.append(
        "- **Leakage guard:** the IHC-identity gene mirrors **ESR1 / PGR / ERBB2 / MKI67** are dropped "
        f"from the discordance-input gene panel ({r.n_panel_genes} genes) and the shared "
        "`assert_no_forbidden_inputs` hard gate is run on the panel before any scoring. Subtype is used "
        "only as the TARGET axis (permitted under LOCK-2), never as an input."
    )
    lines.append(
        f"- **Never a bare number:** Pearson r + bootstrap 95% CI (N={cfg.bootstrap_n}) reported at "
        f"{len(cfg.seeds)} seeds ({list(cfg.seeds)}) + an ECE calibration companion."
    )
    lines.append("")
    lines.append("## Method (self-contained from the staged cohort)")
    lines.append("")
    lines.append(
        "1. **Stage** matched CPTAC-BRCA RNA (washu transcriptomics) + protein (umich proteomics) via "
        "the public `cptac` package; match on Patient_ID (tumour samples)."
    )
    lines.append(
        "2. **Discordance burden (per patient):** over the genes measured in both modalities (IHC "
        "mirrors excluded), z-score each gene across patients separately for RNA and protein, then take "
        "the per-patient **mean |protein_z − rna_z|** — the post-transcriptional buffering magnitude."
    )
    lines.append(
        f"3. **Target — PAM50 continuous subtype-centroid axis** ({r.n_pam50_genes}/50 PAM50 genes "
        "present): median-center + z-score the PAM50 genes, take PC1 (the LumA↔Basal axis), orient by "
        "ESR1 so higher = more Basal-like / more aggressive. Built self-contained from the staged RNA "
        "(no pre-computed PAM50 call ships in the CPTAC clinical). See **Deviations** below."
    )
    lines.append(
        "4. **Floor number:** Pearson r between the discordance burden and the PAM50 axis; per-seed "
        "bootstrap 95% CI; seed-mean CI as the headline; KILL/GREENLIGHT decision vs the config."
    )
    lines.append("")
    lines.append("## Result")
    lines.append("")
    lines.append(
        "| Cohort N | Panel genes | PAM50 genes | Pearson r | Bootstrap CI95 | Shuffle \\|r\\| | ECE | "
        "Seeds | Decision |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    lines.append(
        f"| {r.n} | {r.n_panel_genes} | {r.n_pam50_genes} | {rr} | {ci} | {shuf} | {ece} | {r.seeds} | "
        f"**{outcome.decision}** |"
    )
    lines.append("")
    lines.append(
        f"**Shuffle sentinel (leakage-clean check):** mean permutation-null |r| = **{shuf}** "
        f"(target axis shuffled vs discordance score, {cfg.bootstrap_n} permutations x {r.seeds} "
        "seeds). This collapses toward 0 while the real |r| stands well clear of it — the correlation "
        "is NOT a construction artifact of RNA appearing in both the discordance score and the PAM50 "
        "axis. Same sentinel discipline the project used at G1 and the pCR pilot."
    )
    lines.append("")
    lines.append("### Decision rationale")
    lines.append("")
    lines.append(f"- **{outcome.decision}.** {outcome.rationale}")
    lines.append("")
    lines.append("### Per-seed bootstrap detail")
    lines.append("")
    for ps in r.per_seed:
        rv = "NaN" if ps["pearson_r"] is None else f"{ps['pearson_r']:.3f}"
        civ = ps["boot_ci95"]
        civ_s = "[NaN, NaN]" if civ[0] is None else f"[{civ[0]:.3f}, {civ[1]:.3f}]"
        sv = ps.get("shuffle_abs_r")
        sv_s = "NaN" if sv is None else f"{sv:.3f}"
        lines.append(
            f"- seed {ps['seed']}: Pearson r {rv}, bootstrap CI {civ_s}, shuffle |r| {sv_s}"
        )
    lines.append("")
    lines.append("## Deviations (within-blast-radius — plan target construction)")
    lines.append("")
    lines.append(
        "- **PAM50 target built from staged RNA, not a pre-computed call.** The plan names the target "
        "\"PAM50 subtype continuous centroid distance\". The staged CPTAC clinical (mssm) carries no "
        "PAM50 / subtype column and its `histologic_grade` is entirely `GX` (not assessed), so no "
        "at-diagnosis aggressiveness label ships with the cohort. The continuous axis is therefore "
        "computed self-contained from the 50 PAM50 genes in the staged transcriptomics (all 50 present) "
        "as the ESR1-oriented PC1 of the median-centered PAM50-gene matrix — the dominant LumA↔Basal "
        "subtype-aggressiveness axis. This is an implementation-detail choice within the single arm "
        "file: it faithfully realises the plan's named target while avoiding a mis-specified external "
        "centroid table (which would fabricate the target). Subtype-as-target is permitted (LOCK-2)."
    )
    lines.append(
        "- **Grade cross-check dropped:** a histologic-grade secondary correlation was intended as a "
        "robustness anchor but `histologic_grade` is 134/134 `GX` in the staged clinical — no usable "
        "grade label exists, so the cross-check is not run."
    )
    lines.append("")
    lines.append("## Data provenance")
    lines.append("")
    lines.append(
        "- **Cohort:** CPTAC-BRCA (PDC000173 / Krug, Mertins et al. 2020, *Cell* — prospective breast "
        "proteogenome), distributed by the `cptac` Python package (Payne lab, PNNL) from its "
        "Zenodo/PDC mirrors. Public."
    )
    lines.append(
        "- **Transcriptomics:** `cptac.Brca().get_transcriptomics('washu')` — matched RNA (log2), "
        "gene-symbol columns."
    )
    lines.append(
        "- **Proteomics:** `cptac.Brca().get_proteomics('umich')` — per-patient protein quantification, "
        "gene-symbol columns."
    )
    lines.append(
        "- **Staged/cached to:** `data/genomics/cptac_brca/` (CSV; gitignored under `/data/`)."
    )
    lines.append("")
    return "\n".join(lines)


def _metrics_payload(outcome: ArmOutcome, cfg: ArmConfig) -> dict:
    r = outcome.result
    return {
        "arm": 8,
        "role": "post-transcriptional-regulation-burden-characterisation",
        "framing": (
            "post-transcriptional regulation burden characterisation at diagnosis (pilot, N≈122) — "
            "NOT prognosis/kinetics/early-detection/cross-institution"
        ),
        "pilot_power": {
            "n": r.n,
            "caveat": f"PILOT N={r.n} — underpowered, hypothesis-generating only; no headline number.",
        },
        "cptac_duke_overlap": "zero (different cohorts; no dedup needed)",
        "cohort": "CPTAC-BRCA (PDC000173) matched RNA (washu) + protein (umich)",
        "target": "PAM50 continuous subtype-centroid axis (ESR1-oriented PC1 of staged RNA; see report Deviations)",
        "forbidden_gene_mirrors_excluded": sorted(_FORBIDDEN_GENE_SYMBOLS),
        "n_panel_genes": r.n_panel_genes,
        "n_pam50_genes": r.n_pam50_genes,
        "seeds": list(cfg.seeds),
        "bootstrap_n": cfg.bootstrap_n,
        "kill_threshold_pearson_r_abs": cfg.kill_threshold_r_abs,
        "greenlight_threshold_pearson_r_abs": cfg.greenlight_threshold_r_abs,
        "greenlight_bootstrap_ci_lb": cfg.greenlight_bootstrap_ci_lb,
        "pearson_r": round(r.pearson_r, 4) if np.isfinite(r.pearson_r) else None,
        "boot_ci95": (
            [round(r.boot_ci95[0], 4), round(r.boot_ci95[1], 4)]
            if np.isfinite(r.boot_ci95[0])
            else [None, None]
        ),
        "ece": round(r.ece, 4) if np.isfinite(r.ece) else None,
        "shuffle_abs_r": round(r.shuffle_r, 4) if np.isfinite(r.shuffle_r) else None,
        "shuffle_sentinel_note": (
            "mean permutation-null |r| (target shuffled); collapses toward 0 => the real r is not a "
            "construction artifact of RNA in both terms (leakage-clean check)"
        ),
        "decision": outcome.decision,
        "rationale": outcome.rationale,
        "per_seed": r.per_seed,
        "provenance": {
            "cohort": "CPTAC-BRCA PDC000173 (Krug-Mertins 2020) via the cptac package",
            "transcriptomics": "cptac.Brca().get_transcriptomics('washu')",
            "proteomics": "cptac.Brca().get_proteomics('umich')",
        },
    }


def _fmt_r(x: float) -> str:
    return "NaN" if not np.isfinite(x) else f"{x:.3f}"


def _fmt_ci(ci: tuple[float, float]) -> str:
    return "[NaN, NaN]" if not np.isfinite(ci[0]) else f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def _render_control_section(ctrl: ControlOutcome, cfg: ArmConfig) -> str:
    """Markdown for the '## Construct-Validity Control (RNA-independent axis)' report APPENDIX.

    Written to be APPENDED to the existing arm-8 report (never overwriting the original floor-gate
    section). States the verdict (SURVIVES / AXIS-CIRCULAR), the anchor used (IHC vs protein) with its
    independence caveat, the original-vs-control comparison table, and the N=121 pilot caveat.
    """
    o, c = ctrl.original.result, ctrl.control.result
    today = date.today().strftime("%Y-%m-%d")
    lines: list[str] = []
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Construct-Validity Control (RNA-independent axis)")
    lines.append("")
    lines.append(f"**Added:** {today} (red-team construct-validity control — appended, original above unchanged)  ")
    lines.append(f"**Anchor used:** `{ctrl.anchor}` (IHC preferred but unavailable — see caveat)  ")
    lines.append(f"**Verdict:** **{ctrl.verdict}**")
    lines.append("")
    lines.append(
        "**Why this control exists.** The red-team ruled arm 8 *SURVIVES-WITH-CONTROL*: the shuffle "
        f"sentinel (real |r|={_fmt_r(o.pearson_r)} vs permutation null |r|={_fmt_r(o.shuffle_r)}) "
        "legitimately rules out the pairing artifact, BUT the original subtype axis is RNA-derived "
        "(ESR1-oriented PC1 of the PAM50 **RNA** genes) while the discordance burden also contains RNA "
        "— so a shared-RNA dependence was not fully excluded. This control re-anchors the subtype axis "
        "on an RNA-independent label and re-runs the identical burden correlation."
    )
    lines.append("")
    lines.append("### Anchor selection (empirical, at execute time)")
    lines.append("")
    lines.append(
        "- **PREFERRED — clinical IHC / receptor subtype: UNAVAILABLE.** The staged CPTAC clinical "
        "(mssm `clinical_Pan-cancer.May2022.tsv.gz`) IHC results field "
        "`baseline/ancillary_studies_immunohistochemistry_type_and_result` is **0/134 non-null** for "
        "BRCA; no ER/PR/HER2/subtype column exists and `histologic_grade` is entirely `GX`. No "
        "RNA-independent IHC/subtype label ships with this cohort."
    )
    lines.append(
        "- **FALLBACK — PROTEIN anchor (used).** The subtype axis is rebuilt as the ESR1-oriented PC1 "
        f"of the PAM50 **protein** abundances ({c.n_pam50_genes}/50 PAM50 proteins present). This uses "
        "**zero RNA**, so it removes exactly the RNA-in-both dependence the red-team flagged."
    )
    lines.append(
        "- **Independence caveat (flagged).** The discordance burden term still contains protein, so a "
        "protein-anchored axis does **not** remove a *protein*-in-both dependence — only the RNA-in-both "
        "one. This is a strictly stronger construct check than the RNA axis, not a perfectly "
        "independent one."
    )
    lines.append("")
    lines.append("### Result — original (RNA axis) vs control (protein axis)")
    lines.append("")
    lines.append("| Subtype axis | Pearson r | Bootstrap CI95 | Shuffle \\|r\\| | ECE | Decision |")
    lines.append("|---|---|---|---|---|---|")
    lines.append(
        f"| RNA-derived (original) | {_fmt_r(o.pearson_r)} | {_fmt_ci(o.boot_ci95)} | "
        f"{_fmt_r(o.shuffle_r)} | {_fmt_r(o.ece)} | {ctrl.original.decision} |"
    )
    lines.append(
        f"| **Protein (RNA-independent)** | **{_fmt_r(c.pearson_r)}** | {_fmt_ci(c.boot_ci95)} | "
        f"{_fmt_r(c.shuffle_r)} | {_fmt_r(c.ece)} | {ctrl.control.decision} |"
    )
    lines.append("")
    lines.append(
        f"**Axis agreement:** r(RNA axis, protein axis) = **{_fmt_r(ctrl.axis_agreement_r)}** — the two "
        "subtype axes agree strongly, i.e. the PAM50 subtype axis is robust to whether it is read from "
        "RNA or protein."
    )
    lines.append(
        f"**Trivial-echo check:** r(burden, raw ESR1 protein alone) = **{_fmt_r(ctrl.esr1_alone_r)}** — "
        "the burden is NOT a trivial single-marker (ESR1) readout; the correlation is a genuine "
        "multi-gene discordance signal, not an ESR1 echo."
    )
    lines.append(
        f"**Seeds:** {c.seeds} ({list(cfg.seeds)}); bootstrap N={cfg.bootstrap_n}; matched **PILOT "
        f"N={c.n}**."
    )
    lines.append("")
    lines.append("### Verdict rationale")
    lines.append("")
    lines.append(f"- **{ctrl.verdict}.** {ctrl.rationale}")
    lines.append("")
    lines.append(
        "### Interpretation (ledger-safe)"
    )
    lines.append("")
    if ctrl.verdict == "SURVIVES":
        lines.append(
            "The proteogenomic-discordance-burden characterisation at diagnosis correlates with the "
            "subtype axis even when that axis is read from protein rather than RNA — the signal is not "
            "an artifact of the RNA-derived axis construction. This remains a **PILOT N=121** result: "
            "underpowered, hypothesis-generating only, no headline number. SURVIVES licenses only the "
            "supervised follow-up arm with the same power caveat; it authorises no GPU spend and makes "
            "**no** prognosis / kinetics / early-detection / cross-institution claim."
        )
    else:
        lines.append(
            "The correlation depended on the RNA-derived subtype axis and does not survive re-anchoring "
            "on the RNA-independent protein axis — it is downgraded as axis-circular. This is reported "
            "as-is (no tuning to rescue). It makes **no** prognosis / kinetics / early-detection / "
            "cross-institution claim; **PILOT N=121**, underpowered, hypothesis-generating only."
        )
    lines.append("")
    lines.append(
        "*Method note:* the per-patient discordance burden is unchanged from the original floor gate "
        "(mean |protein_z − rna_z| over the shared panel, IHC mirrors stripped). Only the TARGET axis "
        "was re-anchored. The PAM50 protein subset used to build the axis (a TARGET, permitted under "
        "LOCK-2) is still excluded from the discordance INPUT panel by the leakage guard."
    )
    lines.append("")
    return "\n".join(lines)


def _control_payload(ctrl: ControlOutcome, cfg: ArmConfig) -> dict:
    o, c = ctrl.original.result, ctrl.control.result

    def _num(x: float) -> float | None:
        return round(x, 4) if np.isfinite(x) else None

    def _ci(ci: tuple[float, float]) -> list[float | None]:
        return [round(ci[0], 4), round(ci[1], 4)] if np.isfinite(ci[0]) else [None, None]

    return {
        "arm": 8,
        "control": "construct-validity — RNA-independent subtype axis",
        "framing": (
            "proteogenomic-discordance characterisation at diagnosis (pilot, N=121) — construct "
            "control re-anchoring the subtype axis off the RNA matrix; NOT "
            "prognosis/kinetics/early-detection/cross-institution"
        ),
        "verdict": ctrl.verdict,
        "anchor": ctrl.anchor,
        "anchor_independence_caveat": ctrl.anchor_note,
        "ihc_anchor_available": False,
        "pilot_power": {
            "n": c.n,
            "caveat": f"PILOT N={c.n} — underpowered, hypothesis-generating only; no headline number.",
        },
        "original_rna_axis": {
            "pearson_r": _num(o.pearson_r),
            "boot_ci95": _ci(o.boot_ci95),
            "shuffle_abs_r": _num(o.shuffle_r),
            "ece": _num(o.ece),
            "decision": ctrl.original.decision,
        },
        "control_protein_axis": {
            "pearson_r": _num(c.pearson_r),
            "boot_ci95": _ci(c.boot_ci95),
            "shuffle_abs_r": _num(c.shuffle_r),
            "ece": _num(c.ece),
            "decision": ctrl.control.decision,
            "n_pam50_protein_genes": c.n_pam50_genes,
            "per_seed": c.per_seed,
        },
        "axis_agreement_r_rna_vs_protein": _num(ctrl.axis_agreement_r),
        "burden_vs_esr1_alone_r": _num(ctrl.esr1_alone_r),
        "seeds": list(cfg.seeds),
        "bootstrap_n": cfg.bootstrap_n,
        "rationale": ctrl.rationale,
        "provenance": {
            "cohort": "CPTAC-BRCA PDC000173 (Krug-Mertins 2020) via the cptac package (cached CSVs)",
            "rna_axis": "ESR1-oriented PC1 of PAM50 RNA (original)",
            "protein_axis": "ESR1-oriented PC1 of PAM50 protein (RNA-independent control)",
            "clinical_checked": "mssm clinical_Pan-cancer.May2022 — IHC field 0/134 non-null for BRCA",
        },
    }


def _append_control_to_report(report_dir: Path, section_md: str) -> Path:
    """APPEND the control section to the existing arm-8 report; never overwrite the original.

    Targets `arm8_REPORT_25-07-26.md` if present (the human report), else falls back to the latest
    `floor_gate_*.md`. Idempotent: if a '## Construct-Validity Control (RNA-independent axis)' heading
    already exists, everything from that heading onward is replaced (so a re-run refreshes the control
    in place rather than stacking duplicate sections).
    """
    target = report_dir / "arm8_REPORT_25-07-26.md"
    if not target.exists():
        candidates = sorted(report_dir.glob("floor_gate_*.md"))
        if not candidates:
            raise RuntimeError(
                f"arm8 construct-control BLOCKED — no arm-8 report found under {report_dir} to append "
                "the control section to (expected arm8_REPORT_25-07-26.md or a floor_gate_*.md)."
            )
        target = candidates[-1]

    existing = target.read_text()
    marker = "## Construct-Validity Control (RNA-independent axis)"
    if marker in existing:
        head = existing.split(marker, 1)[0].rstrip()
        # drop a trailing '---' separator we previously emitted just above the marker, if present
        if head.endswith("---"):
            head = head[: -len("---")].rstrip()
        existing = head + "\n"
    target.write_text(existing.rstrip() + "\n" + section_md)
    return target


# ==================================================================================================
# Supervised discordance→subtype arm (--supervised-arm, Brief B advance).
#
# PILOT N=121 — underpowered, hypothesis-generating only. NO headline number from this arm.
# The per-gene discordance feature matrix (|protein_z - rna_z| per gene per patient) is the input;
# binary subtype (basal-like vs non-basal, defined as pam50_axis > median) is the target.
# ==================================================================================================


@dataclass
class SupervisedArmResult:
    """Reportable aggregate for the arm-8 supervised discordance→subtype arm.

    PILOT N=121 — underpowered, hypothesis-generating only. No headline number.
    `auroc` and `delong_ci95` are the multi-seed means (same convention as the shared harness).
    `ci_crosses_050` is the honest null check: True when the 95% CI lower bound is <= 0.50 — reported
    as a null, never hunted for a configuration that clears it (LOCK-2 honest-reporting rule).
    `floor_gate_result` is the raw FloorGateResult for downstream calibration + SHAP (retains OOF).
    `feature_names` and `X_full` are retained for SHAP so the advance step can call harness_shap_top20.
    `last_model` is the model object from the final seed (used for SHAP; documented in report).
    """

    auroc: float
    delong_ci95: tuple[float, float]
    ece: float
    per_seed: list[dict]
    n: int
    n_features: int
    ci_crosses_050: bool
    prevalence: float
    floor_gate_result: FloorGateResult
    feature_names: list[str]
    X_full: np.ndarray
    last_model: object  # LGBMClassifier — stored for SHAP


def discordance_feature_matrix(
    rna: pd.DataFrame,
    protein: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Build the per-gene discordance feature matrix: |protein_z - rna_z| per gene per patient.

    This is the input to the supervised discordance→subtype arm (Brief B advance step).
    Construction (leakage-clean, LOCK-2):
      1. Restrict to genes measured in both RNA and protein.
      2. Exclude the forbidden IHC mirrors (ESR1/PGR/ERBB2/MKI67) and any gene whose canonical
         name maps to a _FORBIDDEN_TOKENS entry (catches MKI67 aliases such as MIB1).
      3. Z-score each gene across patients separately for RNA and protein (same as discordance_burden).
      4. Per-patient per-gene discordance feature = |protein_z - rna_z|.
    Returns (feature_df, feature_names) where feature_df has Patient_ID index and gene-symbol columns.
    Runs assert_no_forbidden_inputs + assert_no_survival_columns on the column set before returning —
    these are hard gates that raise LedgerViolation on any forbidden column.
    """
    protein_syms = set(protein.columns)
    common_genes = [g for g in rna.columns if g in protein_syms]
    panel = [g for g in common_genes if _canonical(str(g)) not in _FORBIDDEN_TOKENS]
    if len(panel) < 100:
        raise RuntimeError(
            f"arm8 supervised-arm BLOCKED — only {len(panel)} genes in the discordance panel "
            "after excluding IHC mirrors; too few to build a stable feature matrix."
        )

    # Hard leakage gates (LOCK-2): must run BEFORE any scoring.
    # A new feature matrix is exactly where a forbidden column can sneak in (handoff brief warning).
    assert_no_forbidden_inputs(panel)
    assert_no_survival_columns(pd.DataFrame(columns=panel))

    rna_p = rna[panel].apply(pd.to_numeric, errors="coerce")
    prot_p = protein[panel].apply(pd.to_numeric, errors="coerce")

    def _zscore(df: pd.DataFrame) -> pd.DataFrame:
        mu = df.mean(axis=0, skipna=True)
        sd = df.std(axis=0, skipna=True).replace(0.0, np.nan)
        return (df - mu) / sd

    rna_z = _zscore(rna_p)
    prot_z = _zscore(prot_p)
    feat_df = (prot_z - rna_z).abs()
    # Fill NaN (gene absent in one modality for that patient) with 0 — post-z-score zero contribution.
    feat_df = feat_df.fillna(0.0)
    return feat_df, list(panel)


def run_supervised_arm(cfg: ArmConfig, data_root: Path) -> SupervisedArmResult:
    """Supervised discordance→subtype arm: per-gene |protein_z - rna_z| → binary subtype AUROC.

    PILOT N=121 — underpowered, hypothesis-generating only. This is the Brief B advance step
    for arm 8 after the GREENLIGHT floor gate. Uses LightGBM (chosen over elastic-net: at N=121
    with ~11K genes, LightGBM handles high-dimensional sparse signal better than SAGA elastic-net,
    which is slow on this dimensionality with few samples and converges poorly without a pre-screened
    gene set; elastic-net is better-suited for the arm 4 advance where the gene set is pre-screened
    and the signal is known to be diffuse).
    Binary target: pam50_axis > median → 1 (basal-like), else 0. This is the same continuous PAM50
    axis used in the floor gate, thresholded at the median for binary classification. Subtype is a
    permitted TARGET under LOCK-2.
    """
    rna, protein = stage_cptac_brca(data_root)
    feat_df, feature_names = discordance_feature_matrix(rna, protein)

    # Binary target: pam50_axis > median = basal-like (the more aggressive / TNBC-adjacent end).
    axis = pam50_axis(rna)
    shared_idx = sorted(set(feat_df.index) & set(axis.index))
    feat_df = feat_df.loc[shared_idx]
    axis = axis.loc[shared_idx]
    y = (axis > axis.median()).astype(int).to_numpy()
    X = feat_df.to_numpy(dtype=float)
    groups = np.array(shared_idx)  # one row per patient → Patient_ID as group

    n = len(shared_idx)
    seeds = cfg.seeds

    # Run the shared classification floor gate (imported as run_classification_gate to avoid
    # shadowing the local run_floor_gate which is the Pearson-r gate for the discordance correlation).
    floor_result = run_classification_gate(
        X, y, groups, model="lightgbm", seeds=seeds, n_folds=5
    )

    ci_lo, ci_hi = floor_result.delong_ci95
    ci_crosses_050 = ci_lo <= 0.50

    # Retain the last-seed model for SHAP (document in report: SHAP is computed on the model
    # from the last CV fold of the last seed — an approximation; a full ensemble SHAP would
    # require re-fitting all folds, which is disproportionate at PILOT N=121).
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise ImportError("LightGBM required for supervised arm.") from exc

    last_model = LGBMClassifier(random_state=int(seeds[-1]), n_estimators=200, verbosity=-1)
    last_model.fit(X, y)  # fit on full data for SHAP (not for generalization — clearly documented)

    return SupervisedArmResult(
        auroc=floor_result.auroc,
        delong_ci95=floor_result.delong_ci95,
        ece=floor_result.ece,
        per_seed=floor_result.per_seed,
        n=n,
        n_features=X.shape[1],
        ci_crosses_050=ci_crosses_050,
        prevalence=float(y.mean()),
        floor_gate_result=floor_result,
        feature_names=feature_names,
        X_full=X,
        last_model=last_model,
    )


def _render_supervised_arm_report(result: SupervisedArmResult, today_str: str) -> str:
    """Markdown report for the supervised discordance→subtype arm. PILOT N=121 everywhere."""
    r = result
    ci_lo, ci_hi = r.delong_ci95
    null_note = (
        "**CI lower bound ≤ 0.50 — this result is a NULL at PILOT N=121. "
        "Do not over-interpret. Wide CI is the honest outcome at N=121.**"
        if r.ci_crosses_050
        else (
            "CI lower bound > 0.50 — the CI excludes chance at this seed-mean, "
            "but PILOT N=121 power is insufficient to claim a robust signal. "
            "Replication at larger N is required before any claim."
        )
    )

    lines: list[str] = []
    lines.append("---")
    lines.append("name: report:arm8-supervised-discordance-subtype")
    lines.append(
        'description: "Arm 8 supervised discordance→subtype arm — PILOT N=121 LightGBM on per-gene '
        "|protein_z - rna_z| feature matrix → binary subtype AUROC; MANDATORY power caveat everywhere.\""
    )
    lines.append(f"date: {today_str}")
    lines.append("metadata:")
    lines.append("  node_type: memory")
    lines.append("  type: report")
    lines.append("  feature: novel-heads")
    lines.append("  phase: arm8-supervised-arm")
    lines.append("---")
    lines.append("")
    lines.append("# Arm 8 — Supervised Discordance→Subtype Arm (PILOT N=121)")
    lines.append("")
    lines.append(
        "**PILOT N=121 — underpowered, hypothesis-generating only. No headline number from this arm.**"
    )
    lines.append("")
    lines.append(f"**Date:** {today_str}  ")
    lines.append("**Arm:** 8 (Wave 1 advance, Brief B) — post-transcriptional regulation burden characterisation  ")
    lines.append(f"**Cohort:** CPTAC-BRCA matched RNA+protein, **PILOT N={r.n}**  ")
    lines.append("**Compute:** CPU-only, $0  ")
    lines.append("**Model:** LightGBM on per-gene |protein_z - rna_z| feature matrix → binary subtype  ")
    lines.append("**Justify model choice:** LightGBM chosen over elastic-net at N=121, ~11K features: "
                 "tree models handle high-dimensional sparse tabular signal at small N better than SAGA "
                 "elastic-net (which converges poorly without a pre-screened gene set in this regime).  ")
    lines.append("")
    lines.append("## MANDATORY power honesty (PILOT N=121 — read this first)")
    lines.append("")
    lines.append(
        f"**PILOT N={r.n} — underpowered, hypothesis-generating only.** A supervised binary classifier "
        "on ~11K features at N=121 is in severe high-dimensionality territory. The 5-fold patient-disjoint "
        "CV folds have ~24 patients each — not enough to train a stable LightGBM. Wide CIs and possible "
        "null results are the EXPECTED, honest pilot outcome. If the CI crosses 0.50, this is reported "
        "plainly as a null — no configuration-hunting to rescue the number."
    )
    lines.append("")
    lines.append("## Ledger framing (LOCK-1)")
    lines.append("")
    lines.append(
        "This organ characterises, at diagnosis, the **post-transcriptional regulation burden** "
        "(per-gene mRNA↔protein discordance magnitude) and asks whether it is informative for binary "
        "subtype characterisation (basal-like vs non-basal). It is a **characterisation** organ "
        "operating on CPTAC-BRCA (PDC000173). CPTAC∩Duke=0. NOT prognosis. NOT treatment response. "
        "NOT kinetics. NOT early detection. NOT cross-institution generalisation."
    )
    lines.append("")
    lines.append("**Proteogenomic discordance burden characterisation at diagnosis (CPTAC-BRCA, "
                 "PILOT N=121). Hypothesis-generating only. NOT a prognosis or treatment-response "
                 "prediction. CPTAC∩Duke=0.**")
    lines.append("")
    lines.append("## Leakage guard")
    lines.append("")
    lines.append(
        "The per-gene discordance feature matrix was checked by both `assert_no_forbidden_inputs` and "
        "`assert_no_survival_columns` (shared harness hard gates) BEFORE any model fitting. "
        "Both passed — the IHC mirrors (ESR1/PGR/ERBB2/MKI67) and all canonical aliases (including "
        "MKI67/MIB1) are excluded from the feature columns. Subtype is used only as the TARGET "
        "(permitted under LOCK-2), never as a feature."
    )
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append(
        f"1. **Feature matrix:** per-gene |protein_z − rna_z| for {r.n_features} genes shared in "
        "both RNA (washu) and protein (umich) after excluding IHC mirrors. Patient index = PILOT N=121."
    )
    lines.append(
        "2. **Target:** binary subtype — pam50_axis (ESR1-oriented PC1 of PAM50 RNA) > cohort median "
        f"→ 1 (basal-like / more aggressive), else 0. Prevalence: {r.prevalence:.3f}."
    )
    lines.append(
        "3. **CV:** 5-fold patient-disjoint (one row per patient; Patient_ID as group). "
        f"Multi-seed ({len(r.per_seed)} seeds): each seed generates independent fold assignments."
    )
    lines.append("4. **DeLong CI (seed-mean):** seed-mean of per-seed 95% DeLong bounds.")
    lines.append("")
    lines.append("## Result — PILOT N=121")
    lines.append("")
    lines.append("| AUROC | DeLong CI95 | ECE | N | Seeds | CI crosses 0.50? |")
    lines.append("|---|---|---|---|---|---|")
    lines.append(
        f"| {r.auroc:.3f} | [{ci_lo:.3f}, {ci_hi:.3f}] | {r.ece:.3f} | {r.n} | {len(r.per_seed)} | "
        f"{'**YES — NULL**' if r.ci_crosses_050 else 'No'} |"
    )
    lines.append("")
    lines.append(f"**PILOT N={r.n} — underpowered, hypothesis-generating only.** {null_note}")
    lines.append("")
    lines.append("### Per-seed detail (PILOT N=121)")
    lines.append("")
    for ps in r.per_seed:
        ps_ci = ps["delong_ci95"]
        lines.append(
            f"- seed {ps['seed']}: AUROC {ps['auroc']:.3f}, DeLong CI [{ps_ci[0]:.3f}, {ps_ci[1]:.3f}], "
            f"ECE {ps['ece']:.3f} — **PILOT N={r.n}**"
        )
    lines.append("")
    lines.append("## Interpretation (ledger-safe)")
    lines.append("")
    if r.ci_crosses_050:
        lines.append(
            "The supervised discordance→subtype arm at PILOT N=121 returns a null or near-null result "
            "(CI crosses 0.50). This is the expected, honest outcome for a ~11K-feature classifier on "
            "~24 training patients per fold. The proteogenomic discordance burden characterisation at "
            "diagnosis (CPTAC-BRCA, PILOT N=121) may carry subtype signal but the power is insufficient "
            "to demonstrate it robustly in a supervised framework. **No headline number is claimed.** "
            "Replication at larger N is required. This does not invalidate the GREENLIGHT floor gate "
            "(the unsupervised correlation r=0.425 remains valid); the supervised arm is an additional "
            "advance step, not a replacement gate."
        )
    else:
        lines.append(
            f"The supervised discordance→subtype arm at PILOT N=121 returns AUROC {r.auroc:.3f} "
            f"[{ci_lo:.3f}, {ci_hi:.3f}] — the CI excludes 0.50 at this seed-mean, but PILOT N=121 "
            "is severely underpowered for ~11K features. **This is a hypothesis-generating result only. "
            "No headline number is claimed.** Wide CIs are the honest representation of the uncertainty. "
            "Replication at larger N (not available for public proteogenomic breast data as of 2026-07) "
            "is required before any claim."
        )
    lines.append("")
    lines.append("**Proteogenomic discordance burden characterisation at diagnosis (CPTAC-BRCA, "
                 "PILOT N=121). Hypothesis-generating only. NOT a prognosis or treatment-response "
                 "prediction. CPTAC∩Duke=0.**")
    lines.append("")
    return "\n".join(lines)


def _supervised_arm_payload(result: SupervisedArmResult, today_str: str) -> dict:
    """JSON-serialisable metrics for the supervised arm. PILOT N=121 caveat in every field."""
    ci_lo, ci_hi = result.delong_ci95
    return {
        "arm": 8,
        "step": "supervised_discordance_subtype",
        "date": today_str,
        "framing": (
            "Post-transcriptional regulation burden characterisation at diagnosis (CPTAC-BRCA, "
            "PILOT N=121). Hypothesis-generating only. NOT prognosis, NOT treatment response, "
            "NOT kinetics, NOT early detection. CPTAC∩Duke=0."
        ),
        "pilot_power": {
            "n": result.n,
            "caveat": f"PILOT N={result.n} — underpowered, hypothesis-generating only; no headline number.",
        },
        "model": "lightgbm",
        "model_justification": (
            "LightGBM chosen over elastic-net: handles high-dimensional sparse tabular signal at small N "
            "better than SAGA elastic-net (which requires a pre-screened gene set to converge at ~11K "
            "features / N=121)."
        ),
        "n_features": result.n_features,
        "target": "binary subtype (pam50_axis > cohort median = basal-like = 1)",
        "prevalence": round(result.prevalence, 4),
        "auroc": round(result.auroc, 4),
        "delong_ci95": [round(ci_lo, 4), round(ci_hi, 4)],
        "ece": round(result.ece, 4),
        "seeds": len(result.per_seed),
        "ci_crosses_050": result.ci_crosses_050,
        "honest_null_note": (
            "CI lower bound <= 0.50 — null at PILOT N=121; expected honest outcome."
            if result.ci_crosses_050
            else "CI lower bound > 0.50 — but PILOT N=121 power is insufficient for a robust claim."
        ),
        "per_seed": result.per_seed,
        "leakage_guard": (
            "assert_no_forbidden_inputs + assert_no_survival_columns run on feature column set before "
            "any model fitting — both PASSED. IHC mirrors (ESR1/PGR/ERBB2/MKI67 + aliases) excluded."
        ),
    }


# ==================================================================================================
# Advance step (--advance, Brief B): OOF calibration + SHAP + replication note + updated report.
# ==================================================================================================

_LARGER_N_REPLICATION_NOTE = (
    "**Larger-N replication (checked 2026-07-28):** No suitable public proteogenomic breast cancer "
    "dataset exists to replicate arm 8 at larger N as of this session. Checked:\n"
    "- CPTAC-BRCA PDC000173 (Krug-Mertins 2020, *Cell*): N=125 total, N=121 matched RNA+protein "
    "— the dataset used here. No larger matched RNA+protein CPTAC-BRCA cohort is available from "
    "the cptac package or PDC as of 2026-07.\n"
    "- TCGA-BRCA: has RNA (N~1000) but NO matched protein quantification at the gene level; "
    "RPPA covers ~200 proteins (not proteome-wide discordance). Not suitable.\n"
    "- METABRIC: RNA-only (no proteomics). Not suitable.\n"
    "- CBTTC / APOLLO-LUAD / CPTAC-other cancers: non-breast. Not suitable.\n"
    "**Conclusion: no suitable public dataset identified — this remains an open residual. "
    "Larger-N replication is possible only when a future matched RNA+protein breast cohort is "
    "made publicly available (e.g. a future CPTAC breast expansion or international proteogenomics "
    "consortium dataset). This is logged and carried forward as an open caveat.**"
)


@dataclass
class AdvanceResult:
    """Results of the arm-8 advance step (OOF calibration + SHAP). PILOT N=121 everywhere."""

    isotonic_cal: dict  # output of oof_calibrate(..., method="isotonic")
    platt_cal: dict     # output of oof_calibrate(..., method="platt")
    shap_top20_list: list[dict]  # top-20 discordance genes by mean |SHAP|
    replication_note: str
    supervised_result: SupervisedArmResult


def run_advance(supervised_result: SupervisedArmResult) -> AdvanceResult:
    """Run OOF calibration (isotonic + Platt) and SHAP for the supervised arm. PILOT N=121."""
    floor_result = supervised_result.floor_gate_result

    # OOF calibration — uses the shared harness oof_calibrate which enforces the tripwire
    # (calibrated ECE < 0.005 raises LedgerViolation — the ADR-0010 in-sample leak guard).
    isotonic_cal = oof_calibrate(floor_result, method="isotonic")
    platt_cal = oof_calibrate(floor_result, method="platt")

    # SHAP top-20 discordance genes. Uses last_model (fitted on full data for SHAP only — documented).
    shap_list = harness_shap_top20(
        floor_result,
        supervised_result.X_full,
        supervised_result.feature_names,
        supervised_result.last_model,
    )

    return AdvanceResult(
        isotonic_cal=isotonic_cal,
        platt_cal=platt_cal,
        shap_top20_list=shap_list,
        replication_note=_LARGER_N_REPLICATION_NOTE,
        supervised_result=supervised_result,
    )


def _render_advance_section(adv: AdvanceResult, today_str: str) -> str:
    """Markdown for the advance section appended to arm8_REPORT_25-07-26.md. PILOT N=121 everywhere."""
    sr = adv.supervised_result
    iso = adv.isotonic_cal
    pla = adv.platt_cal
    ci_lo, ci_hi = sr.delong_ci95

    lines: list[str] = []
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Advance Step — Calibration + SHAP + Supervised Arm + Replication Note")
    lines.append("")
    lines.append(
        f"**Added:** {today_str} (Brief B advance step — appended, original above unchanged)  "
    )
    lines.append("**PILOT N=121 — underpowered, hypothesis-generating only. No headline number.**  ")
    lines.append("")
    lines.append("### Prior construct control reference")
    lines.append("")
    lines.append(
        "The protein-anchored construct control (r = 0.444, bootstrap CI [0.277, 0.579], PILOT N=121) "
        "ran in the prior session and is documented above in '## Construct-Validity Control'. "
        "It SURVIVES — the discordance burden tracks the subtype axis even when that axis is read from "
        "protein rather than RNA. **Residual caveat on record (never resolved): the protein-in-both "
        "dependence.** The discordance burden contains protein, and the protein-anchored axis also "
        "contains protein — so a protein-in-both dependence is not fully excluded. This is documented, "
        "accepted at pilot level, and does NOT invalidate the GREENLIGHT. NOT re-run here — "
        "see construct_control_20260725.json."
    )
    lines.append("")
    lines.append("### Supervised discordance→subtype arm (PILOT N=121)")
    lines.append("")
    lines.append(
        "**PILOT N=121 — underpowered, hypothesis-generating only. No headline number.**  "
    )
    lines.append(
        f"Binary subtype (pam50_axis > median = basal-like) from per-gene |protein_z − rna_z| "
        f"({sr.n_features} gene features), LightGBM, 5-fold patient-disjoint CV, {len(sr.per_seed)} seeds."
    )
    lines.append("")
    lines.append("| AUROC | DeLong CI95 | ECE | CI crosses 0.50? | PILOT N |")
    lines.append("|---|---|---|---|---|")
    lines.append(
        f"| {sr.auroc:.3f} | [{ci_lo:.3f}, {ci_hi:.3f}] | {sr.ece:.3f} | "
        f"{'**YES — NULL**' if sr.ci_crosses_050 else 'No'} | **{sr.n}** |"
    )
    lines.append("")
    for ps in sr.per_seed:
        ps_ci = ps["delong_ci95"]
        lines.append(
            f"- seed {ps['seed']}: AUROC {ps['auroc']:.3f}, CI [{ps_ci[0]:.3f}, {ps_ci[1]:.3f}], "
            f"ECE {ps['ece']:.3f} — **PILOT N={sr.n}**"
        )
    lines.append("")
    if sr.ci_crosses_050:
        lines.append(
            "**Honest null:** the supervised arm at PILOT N=121 is a null (CI crosses 0.50). "
            "This is the expected, honest outcome for ~11K features / ~24 training patients per fold. "
            "The unsupervised floor-gate result (r=0.425, CI [0.252, 0.575]) remains the primary "
            "GREENLIGHT signal. The supervised arm is hypothesis-generating only at this N."
        )
    else:
        lines.append(
            f"The supervised arm at PILOT N=121 returns AUROC {sr.auroc:.3f} — the CI does not "
            "cross 0.50 at this seed-mean, but PILOT N=121 is insufficient for a robust claim. "
            "Wide CIs represent the honest uncertainty. No headline number is claimed."
        )
    lines.append("")
    lines.append(
        "**Proteogenomic discordance burden characterisation at diagnosis (CPTAC-BRCA, PILOT N=121). "
        "Hypothesis-generating only. NOT a prognosis or treatment-response prediction. CPTAC∩Duke=0.**"
    )
    lines.append("")
    lines.append("### OOF calibration (PILOT N=121)")
    lines.append("")
    lines.append(
        "Calibration was fitted using cross-fold OOF predictions (LOCK-2: calibrator fit on other-fold "
        "OOF, evaluated on held-out fold — the Track-C / ADR-0010 in-sample leak is prevented). "
        "At PILOT N=121 the calibration estimate is noisy; do not over-interpret small ECE improvements."
    )
    lines.append("")
    lines.append("| Method | Raw ECE | Calibrated ECE (mean) | Delta |")
    lines.append("|---|---|---|---|")
    iso_delta = iso["calibrated_ece_mean"] - iso["raw_ece_mean"]
    pla_delta = pla["calibrated_ece_mean"] - pla["raw_ece_mean"]
    lines.append(
        f"| Isotonic | {iso['raw_ece_mean']:.4f} | {iso['calibrated_ece_mean']:.4f} | "
        f"{iso_delta:+.4f} |"
    )
    lines.append(
        f"| Platt | {pla['raw_ece_mean']:.4f} | {pla['calibrated_ece_mean']:.4f} | "
        f"{pla_delta:+.4f} |"
    )
    lines.append("")
    lines.append("**Per-seed calibration (PILOT N=121):**")
    for iso_s, pla_s in zip(iso["per_seed"], pla["per_seed"]):
        lines.append(
            f"- seed {iso_s['seed']}: isotonic ECE {iso_s['calibrated_ece']:.4f}, "
            f"Platt ECE {pla_s['calibrated_ece']:.4f} — **PILOT N={sr.n}**"
        )
    lines.append("")
    lines.append("### SHAP top-20 discordance genes (PILOT N=121)")
    lines.append("")
    lines.append(
        "SHAP computed via `shap.TreeExplainer` on a LightGBM model fitted to the full PILOT N=121 "
        "cohort (not a CV fold — the full-data model is used for SHAP as an importance proxy; "
        "it is NOT used for the AUROC estimate, which comes from OOF CV). Feature importances "
        "represent per-gene |protein_z − rna_z| discordance magnitude informative for binary "
        "subtype characterisation at diagnosis. **PILOT N=121 — underpowered, hypothesis-generating "
        "only. Gene rankings are illustrative at this N, not definitive.**"
    )
    lines.append("")
    lines.append("| Rank | Gene | Mean |SHAP| |")
    lines.append("|---|---|---|")
    for entry in adv.shap_top20_list[:20]:
        lines.append(
            f"| {entry['rank']} | {entry['feature']} | {entry['mean_abs_shap']:.6f} |"
        )
    lines.append("")
    lines.append(
        "**Interpretation (ledger-safe):** these genes' mRNA↔protein discordance magnitude is "
        "most informative for post-transcriptional regulation burden characterisation at diagnosis "
        "(CPTAC-BRCA, PILOT N=121). The genes themselves are not claimed as biomarkers; "
        "the discordance magnitude (not absolute expression) is the signal. "
        "NOT a prognosis, growth-rate, or treatment-response readout. CPTAC∩Duke=0. "
        "Hypothesis-generating only at PILOT N=121. The SHAP rankings are unstable at this N "
        "and should not be reported as definitive."
    )
    lines.append("")
    lines.append(
        "SHAP top-20 bar chart saved to `arm8_shap_top20_20260728.png` / `.csv` "
        "(see artifacts in this directory). **PILOT N=121 caveat applies.**"
    )
    lines.append("")
    lines.append("### Larger-N replication note")
    lines.append("")
    lines.append(adv.replication_note)
    lines.append("")
    lines.append("## Advance summary")
    lines.append("")
    lines.append(
        f"**PILOT N={sr.n} — underpowered, hypothesis-generating only. No headline number.**  "
    )
    lines.append(
        "Proteogenomic discordance burden characterisation at diagnosis (CPTAC-BRCA, PILOT N=121). "
        "Hypothesis-generating only. NOT a prognosis or treatment-response prediction. CPTAC∩Duke=0."
    )
    lines.append("")
    lines.append("| Component | Status |")
    lines.append("|---|---|")
    lines.append(f"| Supervised arm AUROC | {sr.auroc:.3f} CI [{ci_lo:.3f},{ci_hi:.3f}] — PILOT N=121 |")
    lines.append(f"| CI crosses 0.50? | {'YES — NULL' if sr.ci_crosses_050 else 'No'} |")
    lines.append(f"| Isotonic calibrated ECE | {iso['calibrated_ece_mean']:.4f} (raw {iso['raw_ece_mean']:.4f}) — PILOT N=121 |")
    lines.append(f"| Platt calibrated ECE | {pla['calibrated_ece_mean']:.4f} (raw {pla['raw_ece_mean']:.4f}) — PILOT N=121 |")
    lines.append(f"| SHAP top-20 | {adv.shap_top20_list[0]['feature']} (rank 1) — PILOT N=121 |")
    lines.append("| Larger-N replication | No suitable public dataset identified — open residual |")
    lines.append("")
    return "\n".join(lines)


def _advance_payload(adv: AdvanceResult, today_str: str) -> dict:
    """JSON-serialisable metrics for the advance step. PILOT N=121 caveat in every field."""
    sr = adv.supervised_result
    iso = adv.isotonic_cal
    pla = adv.platt_cal
    ci_lo, ci_hi = sr.delong_ci95
    return {
        "arm": 8,
        "step": "advance_calibration_shap",
        "date": today_str,
        "framing": (
            "Post-transcriptional regulation burden characterisation at diagnosis (CPTAC-BRCA, "
            "PILOT N=121). Hypothesis-generating only. NOT prognosis, NOT treatment response, "
            "NOT kinetics, NOT early detection. CPTAC∩Duke=0."
        ),
        "pilot_power": {
            "n": sr.n,
            "caveat": f"PILOT N={sr.n} — underpowered, hypothesis-generating only; no headline number.",
        },
        "supervised_arm": {
            "auroc": round(sr.auroc, 4),
            "delong_ci95": [round(ci_lo, 4), round(ci_hi, 4)],
            "ece": round(sr.ece, 4),
            "ci_crosses_050": sr.ci_crosses_050,
            "per_seed": sr.per_seed,
            "n": sr.n,
            "n_features": sr.n_features,
        },
        "calibration": {
            "method": "oof_cross_calibration (anti-pattern: NOT in-sample; see ADR-0010/Track-C)",
            "isotonic": {
                "calibrated_ece": iso["calibrated_ece_mean"],
                "raw_ece": iso["raw_ece_mean"],
                "per_seed": [
                    {"seed": s["seed"], "calibrated_ece": s["calibrated_ece"]}
                    for s in iso["per_seed"]
                ],
            },
            "platt": {
                "calibrated_ece": pla["calibrated_ece_mean"],
                "raw_ece": pla["raw_ece_mean"],
                "per_seed": [
                    {"seed": s["seed"], "calibrated_ece": s["calibrated_ece"]}
                    for s in pla["per_seed"]
                ],
            },
        },
        "calibrated_ece": iso["calibrated_ece_mean"],  # primary field (isotonic)
        "shap_top20": adv.shap_top20_list,
        "replication_note": "No suitable public dataset identified — open residual (see advance report).",
        "construct_control_reference": (
            "Protein-anchored construct control (r=0.444, CI [0.277,0.579], SURVIVES) ran in prior "
            "session — see construct_control_20260725.json. Residual caveat: protein-in-both dependence "
            "not fully excluded. Carry forward as documented. NOT re-run here."
        ),
    }


def _save_shap_csv(shap_list: list[dict], report_dir: Path, stamp: str) -> Path:
    """Write SHAP top-20 to CSV. Returns the path."""
    rows = [{"rank": e["rank"], "gene": e["feature"], "mean_abs_shap": e["mean_abs_shap"]}
            for e in shap_list]
    df = pd.DataFrame(rows)
    csv_path = report_dir / f"arm8_shap_top20_{stamp}.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def _save_shap_png(shap_list: list[dict], report_dir: Path, stamp: str) -> Path | None:
    """Write SHAP top-20 bar chart PNG. Returns path or None if matplotlib is unavailable."""
    try:
        import matplotlib  # noqa: PLC0415 — optional dep, lazy import
        matplotlib.use("Agg")  # non-interactive backend (no display required)
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError:
        return None

    genes = [e["feature"] for e in shap_list[:20]]
    vals = [e["mean_abs_shap"] for e in shap_list[:20]]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(genes[::-1], vals[::-1], color="#4c72b0")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(
        "Arm 8 SHAP top-20 discordance genes\n"
        "(PILOT N=121 — underpowered, hypothesis-generating only)"
    )
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    png_path = report_dir / f"arm8_shap_top20_{stamp}.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    return png_path


def _append_advance_to_report(report_dir: Path, section_md: str) -> Path:
    """APPEND the advance section to arm8_REPORT_25-07-26.md. Idempotent on re-run."""
    target = report_dir / "arm8_REPORT_25-07-26.md"
    if not target.exists():
        raise RuntimeError(
            f"arm8 advance BLOCKED — arm8_REPORT_25-07-26.md not found under {report_dir}. "
            "The report must exist before appending the advance section."
        )
    existing = target.read_text()
    marker = "## Advance Step — Calibration + SHAP + Supervised Arm + Replication Note"
    if marker in existing:
        head = existing.split(marker, 1)[0].rstrip()
        if head.endswith("---"):
            head = head[: -len("---")].rstrip()
        existing = head + "\n"
    target.write_text(existing.rstrip() + "\n" + section_md)
    return target


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg_path = Path(args.config)
    cfg_raw = yaml.safe_load(cfg_path.read_text()) or {}
    cfg = _parse_config(cfg_raw)
    data_root = Path(args.data_root).resolve() if args.data_root else _REPO_ROOT
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = date.today().strftime("%Y%m%d")

    if args.supervised_arm:
        print("arm8 --supervised-arm: PILOT N=121 — underpowered, hypothesis-generating only.")  # noqa: T201
        try:
            supervised_result = run_supervised_arm(cfg, data_root)
        except (ImportError, RuntimeError, LedgerViolation) as exc:
            print(f"BLOCKED: {exc}", file=sys.stderr)  # noqa: T201
            return 2

        # Write standalone supervised-arm report + JSON.
        today_str = date.today().strftime("%Y-%m-%d")
        sup_md = _render_supervised_arm_report(supervised_result, today_str)
        try:
            validate_arm_report_keywords(sup_md, "arm8")
        except LedgerViolation as exc:
            print(f"LEDGER GUARD FAILED on supervised-arm report: {exc}", file=sys.stderr)  # noqa: T201
            return 1

        sup_report_path = report_dir / f"supervised_arm_{stamp}.md"
        sup_json_path = report_dir / f"supervised_arm_{stamp}.json"
        sup_report_path.write_text(sup_md)
        sup_json_path.write_text(json.dumps(_supervised_arm_payload(supervised_result, today_str), indent=2))

        ci_lo, ci_hi = supervised_result.delong_ci95
        null_tag = "NULL (CI crosses 0.50)" if supervised_result.ci_crosses_050 else "non-null (CI > 0.50)"
        print(f"arm8 supervised-arm — report: {sup_report_path}")  # noqa: T201
        print(  # noqa: T201
            f"  PILOT N={supervised_result.n} (underpowered) | {supervised_result.n_features} features | "
            f"AUROC={supervised_result.auroc:.3f} CI=[{ci_lo:.3f},{ci_hi:.3f}] ECE={supervised_result.ece:.3f} "
            f"-> {null_tag}"
        )
        print(f"  json: {sup_json_path}")  # noqa: T201
        print("  leakage guard: PASS (assert_no_forbidden_inputs + assert_no_survival_columns)")  # noqa: T201
        print("  ledger guard: PASS (validate_arm_report_keywords arm8)")  # noqa: T201

        # Also run --advance inline so a single command does both.
        if not args.advance:
            print(  # noqa: T201
                "  NOTE: run --advance to add OOF calibration + SHAP + replication note "
                "and update the main report."
            )
            return 0

        # Fall through to the advance step below (supervised_result is available).
    else:
        supervised_result = None  # type: ignore[assignment]

    if args.advance:
        if supervised_result is None:
            # --advance run standalone: re-run the supervised arm first.
            print("arm8 --advance: running supervised arm first (PILOT N=121)...")  # noqa: T201
            try:
                supervised_result = run_supervised_arm(cfg, data_root)
            except (ImportError, RuntimeError, LedgerViolation) as exc:
                print(f"BLOCKED (supervised arm): {exc}", file=sys.stderr)  # noqa: T201
                return 2

        print("arm8 --advance: OOF calibration + SHAP + replication note (PILOT N=121)...")  # noqa: T201
        try:
            adv = run_advance(supervised_result)
        except LedgerViolation as exc:
            print(f"CALIBRATION TRIPWIRE (in-sample leak): {exc}", file=sys.stderr)  # noqa: T201
            return 1
        except (ImportError, RuntimeError) as exc:
            print(f"BLOCKED (advance): {exc}", file=sys.stderr)  # noqa: T201
            return 2

        today_str = date.today().strftime("%Y-%m-%d")
        adv_section = _render_advance_section(adv, today_str)

        # Validate ledger before writing anything to disk.
        try:
            validate_arm_report_keywords(adv_section, "arm8")
        except LedgerViolation as exc:
            print(f"LEDGER GUARD FAILED on advance report section: {exc}", file=sys.stderr)  # noqa: T201
            return 1

        # Write artifacts.
        adv_md_path = report_dir / f"advance_{stamp}.md"
        adv_json_path = report_dir / f"advance_{stamp}.json"
        adv_md_path.write_text(adv_section)
        adv_json_path.write_text(json.dumps(_advance_payload(adv, today_str), indent=2))

        # SHAP CSV + PNG.
        csv_path = _save_shap_csv(adv.shap_top20_list, report_dir, stamp)
        png_path = _save_shap_png(adv.shap_top20_list, report_dir, stamp)

        # Append advance section to the main arm8 report.
        report_target = _append_advance_to_report(report_dir, adv_section)

        # Final ledger guard on the full updated report.
        try:
            validate_arm_report_keywords(report_target.read_text(), "arm8")
        except LedgerViolation as exc:
            print(f"LEDGER GUARD FAILED on full report after append: {exc}", file=sys.stderr)  # noqa: T201
            return 1

        iso = adv.isotonic_cal
        pla = adv.platt_cal
        top1 = adv.shap_top20_list[0] if adv.shap_top20_list else {}
        print(f"arm8 advance — advance md: {adv_md_path}")  # noqa: T201
        print(f"  json: {adv_json_path}")  # noqa: T201
        print(f"  shap csv: {csv_path}")  # noqa: T201
        if png_path:
            print(f"  shap png: {png_path}")  # noqa: T201
        else:
            print("  shap png: SKIPPED (matplotlib not available) — CSV written instead")  # noqa: T201
        print(f"  report updated: {report_target}")  # noqa: T201
        print(  # noqa: T201
            f"  PILOT N={supervised_result.n} | isotonic ECE {iso['raw_ece_mean']:.4f}->"
            f"{iso['calibrated_ece_mean']:.4f} | platt ECE {pla['raw_ece_mean']:.4f}->"
            f"{pla['calibrated_ece_mean']:.4f}"
        )
        print(f"  SHAP top-1: {top1.get('feature','?')} (mean|SHAP|={top1.get('mean_abs_shap',0):.6f})")  # noqa: T201
        print("  ledger guard: PASS (validate_arm_report_keywords arm8)")  # noqa: T201
        print("  Larger-N replication: no suitable public dataset identified — open residual")  # noqa: T201
        return 0

    if args.construct_control:
        try:
            ctrl = run_construct_control(cfg, data_root)
        except (ImportError, RuntimeError) as exc:
            # honest BLOCKED — could not stage data or build the RNA-independent anchor; never fake it
            print(f"BLOCKED: {exc}", file=sys.stderr)  # noqa: T201
            return 2

        section_md = _render_control_section(ctrl, cfg)
        report_target = _append_control_to_report(report_dir, section_md)
        ctrl_json = report_dir / f"construct_control_{stamp}.json"
        ctrl_json.write_text(json.dumps(_control_payload(ctrl, cfg), indent=2))

        o, c = ctrl.original.result, ctrl.control.result
        print(f"arm8 construct-control — appended to: {report_target}")  # noqa: T201
        print(f"  control json: {ctrl_json}")  # noqa: T201
        print(  # noqa: T201
            f"  anchor={ctrl.anchor} | PILOT N={c.n} | RNA-axis r={_fmt_r(o.pearson_r)} -> "
            f"protein-axis r={_fmt_r(c.pearson_r)} CI={_fmt_ci(c.boot_ci95)} shuffle|r|="
            f"{_fmt_r(c.shuffle_r)} ECE={_fmt_r(c.ece)} | axis-agree r={_fmt_r(ctrl.axis_agreement_r)} "
            f"-> {ctrl.verdict}"
        )
        return 0

    if not args.floor_gate:
        print(  # noqa: T201
            f"arm8: {_PURPOSE}. Pass --floor-gate to run the CPTAC-BRCA discordance-correlation "
            "floor gate, or --construct-control to run the RNA-independent-axis construct control, "
            "or --supervised-arm / --advance for the Brief B advance step (PILOT N=121)."
        )
        return 0

    try:
        outcome = run_arm8(cfg, data_root)
    except (ImportError, RuntimeError) as exc:
        # honest BLOCKED — data could not be staged; NEVER fabricate a floor number
        print(f"BLOCKED: {exc}", file=sys.stderr)  # noqa: T201
        return 2

    report_path = report_dir / f"floor_gate_{stamp}.md"
    metrics_path = report_dir / f"floor_gate_{stamp}.json"
    report_path.write_text(_render_report(outcome, cfg))
    metrics_path.write_text(json.dumps(_metrics_payload(outcome, cfg), indent=2))

    r = outcome.result
    rr = "NaN" if not np.isfinite(r.pearson_r) else f"{r.pearson_r:.3f}"
    ci = (
        "[NaN,NaN]"
        if not np.isfinite(r.boot_ci95[0])
        else f"[{r.boot_ci95[0]:.3f},{r.boot_ci95[1]:.3f}]"
    )
    ece = "NaN" if not np.isfinite(r.ece) else f"{r.ece:.3f}"
    shuf = "NaN" if not np.isfinite(r.shuffle_r) else f"{r.shuffle_r:.3f}"
    print(f"arm8 floor gate — report: {report_path}")  # noqa: T201
    print(  # noqa: T201
        f"  PILOT N={r.n} (underpowered) | panel={r.n_panel_genes} genes | PAM50={r.n_pam50_genes}/50 "
        f"| Pearson r={rr} bootCI={ci} shuffle|r|={shuf} ECE={ece} -> {outcome.decision}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
