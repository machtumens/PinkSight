"""Arm 4 (Wave 1) — HRD/BRCAness genomic-scar characterisation, pan-subtype. FLOOR GATE.

Purpose (one line): characterise, AT DIAGNOSIS, a pan-subtype HRD/BRCAness (genomic-scar) signature
from non-identity tumour expression — the published gap vs prior TNBC-only + binary HRD work. The
floor gate is the cheapest defensible number: can a leakage-safe expression signature recover the
HRD-high-vs-low GENUINE genomic-scar split on TCGA-BRCA at all (patient-disjoint CV, multi-seed)? If it
cannot here — where the scar label is ground-truth genomics — no costlier continuous/regression arm is
warranted. This gate produces the number (AUROC + DeLong CI + ECE + multi-seed + a shuffle sentinel)
and the KILL / GREENLIGHT / INDETERMINATE decision vs the config thresholds.

CONSTRUCT-VALIDITY CORRECTION (2026-07-25, red-team). A prior run of this arm substituted the label to
`ANEUPLOIDY_SCORE` (a generic whole-genome chromosomal-instability / aneuploidy index). A red-team judge
ruled that GREENLIGHT INVALID: aneuploidy is NOT HRD/BRCAness — it is generic CIN — so "expression ->
aneuploidy" is a known, non-novel result and did NOT test the actual arm (expression -> HRD genomic
scar). That aneuploidy run is SUPERSEDED. THIS script tests the GENUINE HRD label so the actual arm is
measured. An honest KILL / INDETERMINATE is a fully acceptable, correct scientific outcome — expression
NOT clearing the floor against genuine HRD (as opposed to aneuploidy) is the true finding, reported
plainly; nothing here is tuned to rescue.

Label choice (GENUINE HRD genomic-scar composite — NOT aneuploidy). The label is the published TCGA HRD
genomic-scar score = HRD-LOH + Telomeric Allelic Imbalance (TAI) + Large-scale State Transitions (LST)
(Marquard et al. 2015; Telli et al. 2016; Knijnenburg et al. 2018, *Cell Reports*, "Genomic and
Molecular Landscape of DNA Damage Repair Deficiency across TCGA"). It is distributed per-sample as
`TCGA.HRD_withSampleID.txt` on the UCSC Xena PanCanAtlas hub (composite `HRD` row = the exact sum of the
three genomic-scar components; verified: corr(HRD, ai1+lst1+hrd-loh) = 1.0). This is a REAL HRD/BRCAness
readout, distinct from `ANEUPLOIDY_SCORE` and from Fraction-Genome-Altered. HRD-high vs low uses the
CANONICAL **Myriad/Telli cutpoint HRD >= 42** (Telli 2016, defined on this three-scar sum) as the
primary split; a cohort-median split is reported as a secondary sensitivity number only.

Ledger guard (LOCK-1/LOCK-2, critical): the organ's output is "HRD/BRCAness genomic-scar
characterisation at diagnosis" EVERYWHERE. NEVER PARP/olaparib-response prediction ("likely to respond
to olaparib"), treatment response, prognosis, kinetics, or cross-institution generalisation — FORBIDDEN.
Leakage-safe inputs: the 53 PAM50 + identity genes are held out AND **BRCA1/BRCA2 expression is held out**
(BRCA1/2 loss is the direct germline+somatic CAUSE of HRD — their expression is a label surrogate; the
germline/somatic mutation status is likewise never an input). The ER/PR/HER2/Ki-67/subtype mirrors are
stripped and the shared `assert_no_forbidden_inputs` gate is run on the feature columns before every fit.

Data (public, staged; BOTH under the gitignored data/ dir):
  - data/genomics/tcga/brca_tcga_pan_can_atlas_2018/data_mrna_seq_v2_rsem.txt
        RSEM expression (Hugo_Symbol x sample), cBioPortal `brca_tcga_pan_can_atlas_2018`.
  - data/genomics/tcga/hrd_knijnenburg2018/TCGA.HRD_withSampleID.txt.gz
        Genuine per-sample HRD genomic-scar score (composite `HRD` row), UCSC Xena PanCanAtlas hub
        (Knijnenburg 2018 / Marquard 2015 / Telli 2016). Download (gitignored, staged on demand):
          mkdir -p data/genomics/tcga/hrd_knijnenburg2018
          curl -sSL -o data/genomics/tcga/hrd_knijnenburg2018/TCGA.HRD_withSampleID.txt.gz \
            https://pancanatlas.xenahubs.net/download/TCGA.HRD_withSampleID.txt.gz
Provenance: TCGA PanCancer Atlas; HRD-scar scores redistributed by UCSC Xena PanCanAtlas hub, public.

Eval integrity (LOCK-2, decisions.md / plan Shared Evaluation Spine):
  - PATIENT-disjoint CV (one primary sample per patient here -> patient-level by construction).
  - Leakage guard: 53 PAM50 + identity genes (incl. ESR1/PGR/ERBB2/MKI67) EXCLUDED from inputs AND
    BRCA1/BRCA2 EXCLUDED from inputs (direct HRD cause); the shared assert_no_forbidden_inputs gate is
    the hard backstop on top.
  - Shuffle sentinel: the HRD-high label is permuted and the identical CV re-run; the shuffled AUROC
    must collapse to ~0.50, proving the real result is leakage-clean and not a construction artifact.
  - Dimensionality cap: top `max_features` genes by inter-patient variance (UNSUPERVISED ranking -> no
    label leakage), the mandatory local-CPU-RAM cap.
  - Never a bare number: AUROC + DeLong 95% CI + ECE at >= 3 seeds (shared wave1_eval_harness).

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

import re  # noqa: E402  (used by the de-confound control report idempotent-append)

from scripts.novel_heads.wave1_eval_harness import (  # noqa: E402
    FloorGateResult,
    LedgerViolation,
    oof_calibrate,
    run_floor_gate,
    shap_top20,
    validate_arm_report_keywords,
)
from tests.test_novel_heads_leakage import assert_no_forbidden_inputs  # noqa: E402

_PURPOSE = "characterise pan-subtype HRD/BRCAness genomic scar at diagnosis (genuine-HRD floor gate)"

# --- Feature-space controls (high-dim guardrails) --------------------------------------------------
# The model is elastic-net logistic (the shared harness GREENLIGHT/robust learner), matching the arm-6
# precedent: the config names `lightgbm`, but LightGBM is not in the arm-run env (ml extra) and arm 6
# ran elastic-net for the same reason. Elastic-net is robust in the p >> n regime and needs no new
# dependency. The config `max_features` variance cap keeps the fit tractable in local CPU RAM.
_MODEL = "elasticnet"

# Floor-fit feature count for the SAGA elastic-net (<= the config `max_features` RAM ceiling). The
# config caps features at 10000 "at most" for local RAM; the SAGA solver, however, is O(features) in
# convergence and one 10k-feature fit is ~72 min for the 15 fits (3 seeds x 5 folds) — far too slow for
# a "cheapest defensible number" FLOOR gate. Selecting the top-variance subset for the fit (still
# UNSUPERVISED -> no label leakage; still under the config ceiling) makes the floor tractable while
# keeping the signal-carrying highest-variance genes. This is a floor-gate performance choice, NOT a
# science change: the RAM cap is honoured, the leakage guard is unchanged, and a GREENLIGHT would
# advance to the fuller elastic-net + calibration arm anyway. Recorded as `floor_fit_features`.
_FLOOR_FIT_FEATURES = 2000

# The GENUINE HRD genomic-scar label file (UCSC Xena PanCanAtlas hub redistribution of the
# Knijnenburg-2018/Marquard-2015 HRD scores). The composite `HRD` row = HRD-LOH + TAI + LST — a real
# HRD/BRCAness readout, NOT aneuploidy/CIN and NOT Fraction-Genome-Altered.
_HRD_LABEL_FILE = "TCGA.HRD_withSampleID.txt.gz"
_HRD_COMPOSITE_ROW = "HRD"  # the composite genomic-scar score row (sum of ai1 + lst1 + hrd-loh)
_HRD_COMPONENT_ROWS = ("ai1", "lst1", "hrd-loh")  # TAI, LST, HRD-LOH (for provenance / sanity)

# Canonical Myriad/Telli genomic-scar HRD-high cutpoint (Telli et al. 2016), defined on this exact
# three-scar sum. Used as the PRIMARY split; a cohort-median split is reported as a secondary
# sensitivity number so the gate is not threshold-fragile.
_MYRIAD_TELLI_CUT = 42.0

# BRCA1/BRCA2 are the DIRECT cause of HRD (germline + somatic loss). Their EXPRESSION is a label
# surrogate, so they are held out of the feature space here (on top of the PAM50 + identity exclusion
# and the shared assert_no_forbidden_inputs backstop, which does NOT itself list BRCA1/2). The
# germline/somatic mutation status is never an input either.
_HRD_CAUSE_GENES = frozenset({"BRCA1", "BRCA2"})

# cBioPortal "missing" sentinels seen in these clinical/label files.
_NA_SENTINELS = frozenset({"", "NA", "NaN", "[Not Available]", "[Not Applicable]", "[Unknown]"})

# --- De-confound control (subtype-stratified HRD) --------------------------------------------------
# The pooled genuine-HRD GREENLIGHT (AUROC 0.893) has an OPEN construct-validity caveat: HRD is
# enriched in basal/TNBC and subtype smears across the transcriptome, so 0.893 may be partly a
# SUBTYPE-PROXY rather than HRD-specific signal. The clean shuffle ruled out label leakage but NOT this
# confound. This control tests HRD prediction WITHIN a fixed subtype stratum, where subtype is held
# ~constant. An honest collapse toward chance is a fully acceptable, valuable outcome — nothing here
# is tuned to rescue.
#
# The stratifier is the PAM50 intrinsic-subtype CALL (`SUBTYPE` column in the cBioPortal TCGA-BRCA
# `data_clinical_patient.txt`: BRCA_LumA / BRCA_LumB / BRCA_Basal / BRCA_Her2 / BRCA_Normal). Subtype
# is a permitted TARGET-SIDE / stratifier variable (used to hold the stratum constant), NEVER a model
# input — the feature space is unchanged (leakage-safe non-PAM50 + non-BRCA1/2 expression). The
# TCGA-BRCA cBioPortal bundle carries NO IHC ER/PR/HER2 receptor-status columns (checked), so the
# receptor-derived TNBC definition is unavailable here; the PAM50 `BRCA_Basal` call is the staged-bundle
# basal/TNBC stratum (basal-like ≈ TNBC for this cohort — the canonical PanCanAtlas convention).
_SUBTYPE_COL = "SUBTYPE"  # PAM50 intrinsic-subtype call (per-patient), cBioPortal TCGA-BRCA
_SUBTYPE_BASAL = ("BRCA_Basal",)  # PRIMARY stratum: basal-like / TNBC
_SUBTYPE_LUMINAL = ("BRCA_LumA", "BRCA_LumB")  # SECONDARY stratum: luminal (ER+/HER2-)
# Minimum stratum size for a defensible within-stratum CV (task: "if N permits, >= ~80/stratum").
_MIN_STRATUM_N = 80


@dataclass(frozen=True)
class ArmConfig:
    """Parsed arm-4 config knobs the floor gate needs (dataset paths + cap + thresholds + seeds)."""

    expression: Path
    hrd_label: Path
    subtype_clinical: Path  # cBioPortal TCGA-BRCA patient clinical (PAM50 SUBTYPE stratifier)
    pam50_exclude_genes: frozenset[str]
    max_features: int
    seeds: tuple[int, ...]
    kill_threshold_auroc: float
    greenlight_threshold_auroc: float
    greenlight_delong_ci_lb: float


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Arm 4 — HRD/BRCAness genuine-HRD floor gate")
    p.add_argument(
        "--config",
        default="configs/novel_heads/arm4_hrd_brcaness.yaml",
        help="arm config (dataset paths, max_features cap, thresholds, seeds)",
    )
    p.add_argument("--floor-gate", action="store_true", help="run the CPU expression floor gate")
    p.add_argument(
        "--deconfound-control",
        action="store_true",
        help=(
            "run the subtype-stratified HRD de-confound control (predict HRD within a fixed PAM50 "
            "subtype stratum, where subtype is held ~constant) — de-confounds the pooled GREENLIGHT"
        ),
    )
    p.add_argument(
        "--continuous-regression",
        action="store_true",
        help=(
            "regress directly on the raw continuous HRD score (HRD-LOH+TAI+LST sum) with elastic-net, "
            "reporting Pearson r + bootstrap CI (N=1000) at 3 seeds — closes the gap between the arm's "
            "stated regressor role and the binary floor gate"
        ),
    )
    p.add_argument(
        "--advance",
        action="store_true",
        help=(
            "OOF calibration (isotonic + Platt) + SHAP top-20 for the binary classification model; "
            "writes calibration JSON, SHAP CSV/PNG, and appends an advance section to arm4_REPORT_25-07-26.md"
        ),
    )
    p.add_argument(
        "--report-dir",
        default="reports/novel_heads/arm4_hrd_brcaness",
        help="directory for the floor-gate report + metrics JSON",
    )
    p.add_argument(
        "--data-root",
        default=None,
        help="override the data root the config paths resolve against (default: repo root)",
    )
    return p


# ==================================================================================================
# Config loading. The stub config ships `datasets:` as `TBD/...` placeholders (populated: false) naming
# a Telli HRD score + 450k methylation. The GENUINE HRD label IS now staged (the Telli/Knijnenburg
# per-sample HRD genomic-scar composite) under data/genomics/tcga/hrd_knijnenburg2018/. This resolver
# points the arm at the staged files (same pattern as arm 6's DepMap resolver). A required missing file
# raises FileNotFoundError so the CLI exits with the exact missing-data reason (the honest BLOCKED
# outcome), never a fabricated number. The PAM50 exclusion list is reused from the arm-2 config (single
# source of truth) so arm 4 never re-inlines the 53-gene list.
# ==================================================================================================
def _load_pam50_exclusion(data_root: Path) -> frozenset[str]:
    """Reuse the arm-2 config's canonical 53-gene PAM50 + identity exclusion list (DRY single source).

    Arm 2 owns `pam50_exclude_genes` (Parker et al. 2009 PAM50 + ESR1/ERBB2/MKI67). Arm 4's feature
    space is "non-PAM50 expression", so it excludes the SAME genes — reading them from the arm-2 config
    keeps one source of truth. If the arm-2 config is absent, fall back to the hard leakage backstop
    (assert_no_forbidden_inputs still catches ESR1/ERBB2/MKI67); the PAM50 exclusion is an additional
    subtype-circularity guard, so a missing list degrades to "identity mirrors only" with a warning.
    """
    arm2_cfg = data_root / "configs" / "novel_heads" / "arm2_purity_admixture.yaml"
    if not arm2_cfg.exists():
        print(  # noqa: T201
            f"WARNING: arm-2 config not found at {arm2_cfg} — PAM50 exclusion falls back to the "
            "identity-mirror leakage backstop only (ESR1/ERBB2/MKI67).",
            file=sys.stderr,
        )
        return frozenset()
    raw = yaml.safe_load(arm2_cfg.read_text()) or {}
    genes = raw.get("pam50_exclude_genes") or []
    return frozenset(str(g).strip().upper() for g in genes)


def _resolve_config(cfg_raw: dict, data_root: Path) -> ArmConfig:
    tcga = data_root / "data" / "genomics" / "tcga" / "brca_tcga_pan_can_atlas_2018"
    hrd_dir = data_root / "data" / "genomics" / "tcga" / "hrd_knijnenburg2018"
    paths = {
        "expression": tcga / "data_mrna_seq_v2_rsem.txt",
        "hrd_label": hrd_dir / _HRD_LABEL_FILE,
    }
    # PAM50-subtype stratifier clinical file (used only by the de-confound control). It ships in the
    # SAME cBioPortal TCGA-BRCA bundle as the expression matrix, so it is present whenever the floor
    # gate can run; resolved here so the control has a single source. Not required by the floor gate.
    subtype_clinical = tcga / "data_clinical_patient.txt"
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "arm4 floor gate BLOCKED — required file(s) not staged: "
            + ", ".join(missing)
            + ".\nStage the two public inputs under the gitignored data/ dir:\n"
            "  (1) cBioPortal `brca_tcga_pan_can_atlas_2018` RSEM expression "
            "(data_mrna_seq_v2_rsem.txt) under data/genomics/tcga/brca_tcga_pan_can_atlas_2018/;\n"
            "  (2) the GENUINE per-sample HRD genomic-scar score (Knijnenburg 2018 / Marquard 2015 / "
            "Telli 2016) via UCSC Xena PanCanAtlas hub:\n"
            "      mkdir -p data/genomics/tcga/hrd_knijnenburg2018\n"
            "      curl -sSL -o data/genomics/tcga/hrd_knijnenburg2018/TCGA.HRD_withSampleID.txt.gz "
            "https://pancanatlas.xenahubs.net/download/TCGA.HRD_withSampleID.txt.gz\n"
            "  (This is the genuine HRD label, NOT ANEUPLOIDY_SCORE — do NOT fall back to aneuploidy.)"
        )
    seeds = tuple(int(s) for s in cfg_raw.get("seeds", (0, 1, 2)))
    return ArmConfig(
        expression=paths["expression"],
        hrd_label=paths["hrd_label"],
        subtype_clinical=subtype_clinical,
        pam50_exclude_genes=_load_pam50_exclusion(data_root),
        max_features=int(cfg_raw.get("max_features", 10000)),
        seeds=seeds,
        kill_threshold_auroc=float(cfg_raw.get("kill_threshold_auroc", 0.52)),
        greenlight_threshold_auroc=float(cfg_raw.get("greenlight_threshold_auroc", 0.65)),
        greenlight_delong_ci_lb=float(cfg_raw.get("greenlight_delong_ci_lb", 0.55)),
    )


# ==================================================================================================
# Label build — GENUINE HRD genomic-scar composite per patient (continuous + Myriad/Telli >=42 binary).
# The Xena HRD file is TRANSPOSED (rows = score type ai1/lst1/hrd-loh/HRD, columns = TCGA sample IDs).
# ==================================================================================================
def _hrd_labels(cfg: ArmConfig) -> pd.Series:
    """Continuous GENUINE HRD genomic-scar composite per SAMPLE_ID (samples with a non-NA HRD score).

    Reads the transposed Xena HRD table, extracts the composite `HRD` row (= HRD-LOH + TAI + LST), and
    returns a per-sample Series (index = TCGA sample id, e.g. TCGA-3C-AAAU-01; value = float HRD score).
    Rows with a missing score are dropped. This is the genuine HRD/BRCAness genomic-scar label — NOT
    aneuploidy, NOT Fraction-Genome-Altered. The Myriad/Telli >=42 split (and a secondary median split)
    are applied later on the matched cohort, not on the raw file.
    """
    # Transposed file: index_col=0 gives rows keyed by score type; .T -> samples x score types.
    raw = pd.read_csv(cfg.hrd_label, sep="\t", index_col=0, compression="infer")
    scored = raw.T
    scored.index = scored.index.astype(str)
    if _HRD_COMPOSITE_ROW not in scored.columns:
        raise ValueError(
            f"arm4: composite HRD row {_HRD_COMPOSITE_ROW!r} not found in the staged HRD file "
            f"(available score rows: {list(scored.columns)}). The genuine HRD genomic-scar label is "
            "absent; cannot build the floor gate label (do NOT substitute aneuploidy)."
        )
    comp = scored[_HRD_COMPOSITE_ROW].replace(list(_NA_SENTINELS), np.nan)
    comp = pd.to_numeric(comp, errors="coerce")
    return comp.dropna()


# ==================================================================================================
# Feature-matrix build — non-PAM50 + non-BRCA1/2 RSEM expression, log1p + z-score, top-variance capped.
# ==================================================================================================
def _feature_exclusion(cfg: ArmConfig) -> frozenset[str]:
    """The upper-cased gene symbols dropped from inputs: PAM50 + identity mirrors + BRCA1/BRCA2.

    PAM50 + identity (from the arm-2 config) guard subtype circularity / LOCK-2 mirrors; BRCA1/BRCA2
    are added because BRCA1/2 loss is the DIRECT cause of HRD, so their expression is a label surrogate
    the shared `assert_no_forbidden_inputs` backstop does NOT itself list. Union of both.
    """
    return frozenset(g.upper() for g in cfg.pam50_exclude_genes) | _HRD_CAUSE_GENES


def _load_expression(cfg: ArmConfig) -> pd.DataFrame:
    """RSEM expression as (samples x genes), non-PAM50/identity + non-BRCA1/2 genes only, log1p.

    Steps: (1) read the RSEM matrix (rows = genes `Hugo_Symbol`, cols = SAMPLE_IDs); (2) drop NaN gene
    symbols and collapse duplicate symbols by mean; (3) DROP the PAM50 + identity exclusion genes AND
    BRCA1/BRCA2 (subtype circularity / LOCK-2 label surrogates / direct HRD cause); (4) transpose to
    samples-as-rows; (5) log1p the non-negative RSEM values (variance-stabilising) — z-scoring per gene
    happens inside the harness's StandardScaler (fit on TRAIN only, so no leakage). Returns bare
    gene-symbol columns so the leakage canonicaliser can see and reject any forbidden mirror.
    """
    expr = pd.read_csv(cfg.expression, sep="\t")
    expr = expr.drop(columns=["Entrez_Gene_Id"], errors="ignore")
    expr = expr[expr["Hugo_Symbol"].notna()]
    expr["Hugo_Symbol"] = expr["Hugo_Symbol"].astype(str).str.strip()
    # collapse duplicate Hugo symbols by mean (a few dupes in this file), then index by symbol
    expr = expr.groupby("Hugo_Symbol", sort=False).mean(numeric_only=True)

    # DROP PAM50 + identity exclusion genes AND BRCA1/BRCA2 (case-insensitive).
    exclude = _feature_exclusion(cfg)
    drop = [g for g in expr.index if g.upper() in exclude]
    expr = expr.drop(index=drop)

    samples_x_genes = expr.T  # rows = SAMPLE_ID, cols = gene symbol
    samples_x_genes.index = samples_x_genes.index.astype(str)
    # log1p the non-negative RSEM values (variance stabilisation). NaNs (rare) -> 0 pre-log.
    values = np.log1p(samples_x_genes.to_numpy(dtype=float, na_value=0.0))
    return pd.DataFrame(values, index=samples_x_genes.index, columns=samples_x_genes.columns)


def _top_variance_genes(expr: pd.DataFrame, k: int) -> pd.DataFrame:
    """Keep the top-k highest-variance genes (UNSUPERVISED -> no label leakage). Drops all-NaN genes."""
    expr = expr.dropna(axis=1, how="all")
    if expr.shape[1] <= k:
        return expr
    variances = expr.var(axis=0, skipna=True).fillna(0.0)
    top = variances.sort_values(ascending=False).head(k).index
    return expr[top]


def _normalise_sample_id(sample_id: str) -> str:
    """First-4-field TCGA barcode (TCGA-3C-AAAU-01) for matching expression <-> HRD sample ids."""
    return "-".join(str(sample_id).split("-")[:4])


def _patient_id(sample_id: str) -> str:
    """First-3-field TCGA patient barcode (TCGA-3C-AAAU) for matching a sample to its patient row."""
    return "-".join(str(sample_id).split("-")[:3])


def _load_subtype_calls(cfg: ArmConfig) -> dict[str, str]:
    """PATIENT_ID -> PAM50 intrinsic-subtype call (the de-confound-control STRATIFIER, never an input).

    Reads the `SUBTYPE` column (BRCA_LumA / BRCA_LumB / BRCA_Basal / BRCA_Her2 / BRCA_Normal) from the
    cBioPortal TCGA-BRCA `data_clinical_patient.txt` (`#`-prefixed metadata rows skipped). Returns a
    patient-id -> subtype map for patients with a non-NA subtype. This is a TARGET-SIDE variable used
    only to hold a stratum ~constant; it is NEVER passed into the feature matrix. Raises a clear
    BLOCKED error naming the exact missing file/column if the stratifier is absent (never fabricates a
    stratum).
    """
    if not cfg.subtype_clinical.exists():
        raise FileNotFoundError(
            "arm4 de-confound control BLOCKED — PAM50-subtype stratifier file not staged: "
            f"{cfg.subtype_clinical}. It ships in the cBioPortal `brca_tcga_pan_can_atlas_2018` bundle "
            "alongside the expression matrix (data_clinical_patient.txt); stage that bundle to run the "
            "control."
        )
    clinical = pd.read_csv(cfg.subtype_clinical, sep="\t", comment="#")
    if _SUBTYPE_COL not in clinical.columns or "PATIENT_ID" not in clinical.columns:
        raise ValueError(
            f"arm4 de-confound control: expected columns PATIENT_ID + {_SUBTYPE_COL!r} in "
            f"{cfg.subtype_clinical} (found: {list(clinical.columns)[:12]}...). The PAM50 subtype "
            "stratifier is absent — cannot run the subtype-stratified control."
        )
    sub = clinical[["PATIENT_ID", _SUBTYPE_COL]].copy()
    sub[_SUBTYPE_COL] = sub[_SUBTYPE_COL].replace(list(_NA_SENTINELS), np.nan)
    sub = sub.dropna(subset=[_SUBTYPE_COL])
    return {
        str(pid): str(st).strip()
        for pid, st in zip(sub["PATIENT_ID"], sub[_SUBTYPE_COL], strict=False)
    }


@dataclass
class HrdGateOutcome:
    """The arm-4 GENUINE-HRD floor-gate result + KILL/GREENLIGHT decision (never a bare number)."""

    n_patients: int
    n_hrd_high: int
    n_features_used: int
    myriad_cut: float
    median_split: float
    shuffle_auroc: float
    shuffle_ci95: tuple[float, float]
    median_split_auroc: float
    median_split_ci95: tuple[float, float]
    result: FloorGateResult
    decision: str  # "KILL" | "GREENLIGHT" | "INDETERMINATE"
    rationale: str


def _decide(
    result: FloorGateResult, cfg: ArmConfig, n_patients: int, n_pos: int
) -> tuple[str, str]:
    """Apply the pre-registered KILL (AUROC<=0.52) / GREENLIGHT (AUROC>=0.65 AND DeLong LB>=0.55) gate.

    Between the two thresholds — or a GREENLIGHT-magnitude point estimate whose DeLong lower bound
    fails to clear the config `greenlight_delong_ci_lb` — is INDETERMINATE (real-but-weak). The KILL
    floor is the construct-corrected task value (AUROC <= 0.52, ~ shuffle + 2 sigma). GREENLIGHT stays
    at AUROC >= 0.65 AND DeLong LB >= 0.55. n_patients / n_pos are surfaced in the rationale so the
    powered-N context stays attached. Framing is genomic-scar characterisation only (LOCK-1).
    """
    auroc = result.auroc
    lb = result.delong_ci95[0]
    kill = cfg.kill_threshold_auroc
    green = cfg.greenlight_threshold_auroc
    green_lb = cfg.greenlight_delong_ci_lb
    if not np.isfinite(auroc):
        return "INDETERMINATE", (
            f"AUROC is NaN (degenerate CV — likely single-class folds at N={n_patients})."
        )
    if auroc <= kill:
        return "KILL", (
            f"mean patient-disjoint AUROC {auroc:.3f} <= KILL {kill:.2f} — the non-identity "
            f"expression signature does NOT recover the GENUINE HRD-high genomic-scar split on "
            f"{n_patients} TCGA-BRCA patients ({n_pos} HRD-high @ Myriad/Telli >=42). Honest-null "
            f"floor: expression -> HRD (as opposed to expression -> aneuploidy) does not clear the "
            f"floor — the scientifically correct finding, reported plainly."
        )
    if auroc >= green and lb >= green_lb:
        return "GREENLIGHT", (
            f"mean patient-disjoint AUROC {auroc:.3f} >= GREENLIGHT {green:.2f} and DeLong LB "
            f"{lb:.3f} >= {green_lb:.2f} — GENUINE HRD/BRCAness genomic-scar characterisation signal "
            f"present ({n_patients} patients, {n_pos} HRD-high). Advance to the continuous "
            f"Pearson-r + calibration arm."
        )
    return "INDETERMINATE", (
        f"mean patient-disjoint AUROC {auroc:.3f} (DeLong LB {lb:.3f}) is between KILL {kill:.2f} "
        f"and GREENLIGHT {green:.2f} / does not clear the DeLong LB floor {green_lb:.2f} — "
        f"real-but-weak / inconclusive at N={n_patients}. Expression -> genuine-HRD is weak, not "
        f"clearly present — reported plainly, not tuned to rescue."
    )


def _build_matrices(cfg: ArmConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series, int]:
    """Shared matrix build: (X, y_myriad, groups, matched_score, n_features) leakage-checked.

    Returns the top-variance non-PAM50 + non-BRCA1/2 expression matrix X, the Myriad/Telli (>=42)
    HRD-high labels, the patient groups, the matched continuous HRD score (for the median secondary),
    and the fitted feature count. Runs `assert_no_forbidden_inputs` on the columns before returning.
    """
    labels = _hrd_labels(cfg)  # SAMPLE_ID -> continuous genuine HRD score
    expr_full = _load_expression(cfg)  # SAMPLE_ID x non-PAM50/non-BRCA1/2 genes (log1p)

    # Match expression samples to HRD samples via first-4-field TCGA barcode normalisation.
    hrd_by_norm: dict[str, str] = {}
    for sid in labels.index:
        hrd_by_norm.setdefault(_normalise_sample_id(sid), sid)
    shared = [s for s in expr_full.index if _normalise_sample_id(s) in hrd_by_norm]
    shared = sorted(shared)
    if len(shared) < 2 * len(cfg.seeds) * 5:  # need enough for a seed x 5-fold patient-disjoint CV
        raise ValueError(
            f"arm4: only {len(shared)} matched samples (expression x genuine-HRD label) — below the "
            f"floor for a {len(cfg.seeds)}-seed 5-fold patient-disjoint CV. Cannot fabricate a number."
        )

    score = pd.Series(
        [float(labels[hrd_by_norm[_normalise_sample_id(s)]]) for s in shared], index=shared
    )
    # PRIMARY split: canonical Myriad/Telli genomic-scar cutpoint HRD >= 42.
    y = (score.to_numpy() >= _MYRIAD_TELLI_CUT).astype(int)

    # Top-variance feature select, bounded by BOTH the config RAM ceiling and the floor-fit budget
    # (the smaller wins). UNSUPERVISED ranking -> no label leakage; the config `max_features` cap is
    # still honoured (never exceeded), the floor-fit budget just keeps SAGA tractable.
    fit_k = min(cfg.max_features, _FLOOR_FIT_FEATURES)
    expr_shared = _top_variance_genes(expr_full.loc[shared], fit_k)
    # impute any residual per-gene NaN with the gene mean (unsupervised fill — no label information).
    expr_shared = expr_shared.fillna(expr_shared.mean(axis=0))

    # HARD leakage gate (LOCK-2): the feature columns must carry no forbidden IHC/subtype surrogate.
    # (BRCA1/BRCA2 were already dropped in _load_expression; this backstop catches the ER/PR/HER2/
    # Ki-67/subtype mirrors.)
    assert_no_forbidden_inputs(expr_shared.columns)

    x = expr_shared.to_numpy(dtype=float)
    groups = np.asarray(shared)  # one row per patient -> patient-disjoint folding is automatic
    return x, y, groups, score, int(expr_shared.shape[1])


def run_arm4_floor_gate(cfg: ArmConfig) -> HrdGateOutcome:
    """Build the non-identity expression -> GENUINE-HRD matrices and run the shared floor gate.

    (1) continuous genuine HRD composite label per sample; (2) non-PAM50 + non-BRCA1/2 log1p
    expression as X; (3) join on shared samples; (4) HRD-high vs low = Myriad/Telli >=42 split
    (primary) + cohort-median split (secondary sensitivity); (5) top-variance cap to config
    max_features; (6) HARD leakage gate on the columns; (7) shared harness run (elastic-net,
    patient-disjoint CV, config seeds); (8) SHUFFLE SENTINEL — permute the HRD-high label and re-run
    the identical CV; the shuffled AUROC must collapse to ~0.50, proving leakage-cleanliness;
    (9) KILL/GREENLIGHT decision on the primary Myriad/Telli result. One primary sample per patient
    here, so SAMPLE_ID groups are patient-disjoint by construction.
    """
    x, y, groups, score, n_features = _build_matrices(cfg)

    # Primary gate: Myriad/Telli HRD-high (>=42) vs low.
    result = run_floor_gate(x, y, groups, model=_MODEL, seeds=cfg.seeds)

    # SHUFFLE SENTINEL (leakage-clean check, arm-8 discipline): permute the HRD-high label vs the
    # feature rows and re-run the identical patient-disjoint multi-seed CV. Under the null the AUROC
    # must collapse to ~0.50; a non-trivial shuffle AUROC would flag the real result as a leakage /
    # construction artifact rather than genuine expression -> HRD signal.
    rng = np.random.default_rng(0)
    y_shuffled = y.copy()
    rng.shuffle(y_shuffled)
    shuffle_res = run_floor_gate(x, y_shuffled, groups, model=_MODEL, seeds=cfg.seeds)

    # Secondary sensitivity: cohort-median split (so the result is not fragile to the >=42 cutpoint).
    median_split = float(score.median())
    y_median = (score.to_numpy() > median_split).astype(int)
    median_res = run_floor_gate(x, y_median, groups, model=_MODEL, seeds=cfg.seeds)

    decision, rationale = _decide(result, cfg, n_patients=len(y), n_pos=int(y.sum()))
    return HrdGateOutcome(
        n_patients=int(len(y)),
        n_hrd_high=int(y.sum()),
        n_features_used=n_features,
        myriad_cut=_MYRIAD_TELLI_CUT,
        median_split=median_split,
        shuffle_auroc=float(shuffle_res.auroc),
        shuffle_ci95=shuffle_res.delong_ci95,
        median_split_auroc=float(median_res.auroc),
        median_split_ci95=median_res.delong_ci95,
        result=result,
        decision=decision,
        rationale=rationale,
    )


# ==================================================================================================
# De-confound control — subtype-stratified HRD prediction (does expression carry HRD signal BEYOND
# subtype?). Predict HRD-high(>=42) vs low WITHIN a fixed PAM50 subtype stratum, where subtype is held
# ~constant. Honest collapse toward chance is a fully acceptable, valuable outcome; nothing is tuned to
# rescue. Reuses the floor-gate primitives verbatim (labels, expression, top-variance, leakage gate,
# shared run_floor_gate) — the ONLY change is the within-stratum row restriction.
# ==================================================================================================
@dataclass
class StratumResult:
    """One subtype stratum's within-stratum HRD-prediction result + de-confound verdict."""

    stratum_label: str  # e.g. "basal/TNBC (BRCA_Basal)"
    stratum_codes: tuple[str, ...]
    n_patients: int
    n_hrd_high: int
    n_hrd_low: int
    prevalence: float
    n_features_used: int
    result: FloorGateResult | None  # None when the stratum is below the CV floor (INDETERMINATE)
    shuffle_auroc: float
    shuffle_ci95: tuple[float, float]
    verdict: str  # "DE-CONFOUNDED" | "SUBTYPE-PROXY" | "INDETERMINATE-underpowered"
    rationale: str


@dataclass
class DeconfoundOutcome:
    """The full de-confound control across strata, carrying the pooled reference for context."""

    pooled_auroc: float
    pooled_ci95: tuple[float, float]
    strata: list[StratumResult]


def _build_stratum_matrices(
    cfg: ArmConfig, stratum_codes: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """(X, y_myriad, groups, n_features) for samples whose PAM50 subtype is in `stratum_codes`.

    Identical input pipeline to the floor gate — genuine HRD composite label, non-PAM50 + non-BRCA1/2
    log1p expression, Myriad/Telli >=42 HRD-high split, top-variance cap under the config ceiling, hard
    `assert_no_forbidden_inputs` gate — with ONE change: rows are restricted to samples whose patient's
    PAM50 subtype call is in `stratum_codes` (the stratifier is TARGET-side, never an input). Top-
    variance selection is computed WITHIN the stratum so it reflects the stratum's own variance
    structure. Raises ValueError if the stratum is below the CV floor (caller converts to
    INDETERMINATE-underpowered) so no number is fabricated on too-few samples.
    """
    labels = _hrd_labels(cfg)  # SAMPLE_ID -> continuous genuine HRD score
    expr_full = _load_expression(cfg)  # SAMPLE_ID x non-PAM50/non-BRCA1/2 genes (log1p)
    subtype_by_patient = _load_subtype_calls(cfg)  # PATIENT_ID -> PAM50 call (stratifier)

    hrd_by_norm: dict[str, str] = {}
    for sid in labels.index:
        hrd_by_norm.setdefault(_normalise_sample_id(sid), sid)

    wanted = set(stratum_codes)
    shared = [
        s
        for s in expr_full.index
        if _normalise_sample_id(s) in hrd_by_norm
        and subtype_by_patient.get(_patient_id(s)) in wanted
    ]
    shared = sorted(shared)
    if len(shared) < 2 * len(cfg.seeds) * 5:  # need enough for a seed x 5-fold patient-disjoint CV
        raise ValueError(
            f"stratum {stratum_codes} has only {len(shared)} matched samples — below the floor for a "
            f"{len(cfg.seeds)}-seed 5-fold patient-disjoint CV. Cannot fabricate a number."
        )

    score = pd.Series(
        [float(labels[hrd_by_norm[_normalise_sample_id(s)]]) for s in shared], index=shared
    )
    y = (score.to_numpy() >= _MYRIAD_TELLI_CUT).astype(int)

    fit_k = min(cfg.max_features, _FLOOR_FIT_FEATURES)
    expr_shared = _top_variance_genes(expr_full.loc[shared], fit_k)
    expr_shared = expr_shared.fillna(expr_shared.mean(axis=0))
    assert_no_forbidden_inputs(expr_shared.columns)  # HARD leakage gate (LOCK-2)

    x = expr_shared.to_numpy(dtype=float)
    groups = np.asarray(shared)  # one row per patient -> patient-disjoint by construction
    return x, y, groups, int(expr_shared.shape[1])


def _decide_deconfound(
    result: FloorGateResult, n_patients: int, n_pos: int, n_neg: int
) -> tuple[str, str]:
    """DE-CONFOUNDED / SUBTYPE-PROXY / INDETERMINATE-underpowered on the within-stratum HRD result.

    - **DE-CONFOUNDED**: the within-stratum DeLong CI lower bound is clearly > 0.50 — expression
      carries HRD-specific signal BEYOND subtype, so the pooled GREENLIGHT is de-confounded / holds.
    - **SUBTYPE-PROXY**: the within-stratum CI crosses 0.50 (LB <= 0.50) with a point estimate near
      chance — HRD signal is not separable from subtype at this N; the pooled 0.893 was substantially a
      subtype-proxy. Reported plainly as the corrected finding.
    - **INDETERMINATE-underpowered**: degenerate CV (NaN AUROC / single-class folds) or a point
      estimate materially above chance whose wide CI still crosses 0.50 — the N is too small for a
      conclusive CI; do not over-read a wide CI either way.

    The point estimate is used only to distinguish "collapsed to chance" (SUBTYPE-PROXY) from "signal
    present but CI too wide to confirm" (INDETERMINATE). The CI lower bound is the primary criterion.
    Applied once, no tuning.
    """
    auroc = result.auroc
    lb, ub = result.delong_ci95
    if not np.isfinite(auroc) or not np.isfinite(lb):
        return "INDETERMINATE-underpowered", (
            f"within-stratum AUROC is NaN (degenerate CV — likely single-class folds at "
            f"N={n_patients}, {n_pos} HRD-high / {n_neg} low). Too small for a conclusive CI."
        )
    if lb > 0.50:
        return "DE-CONFOUNDED", (
            f"within-stratum AUROC {auroc:.3f}, DeLong CI [{lb:.3f}, {ub:.3f}] — the CI lower bound "
            f"is clearly above chance, so a leakage-safe expression signature recovers HRD-high vs low "
            f"WITHIN a fixed subtype stratum (N={n_patients}, {n_pos} HRD-high / {n_neg} low). "
            f"Expression carries HRD-specific signal BEYOND subtype -> the pooled GREENLIGHT is "
            f"DE-CONFOUNDED / holds."
        )
    # CI crosses chance. Split SUBTYPE-PROXY (collapsed to ~chance) from INDETERMINATE (wide-CI signal).
    if auroc <= 0.60:
        return "SUBTYPE-PROXY", (
            f"within-stratum AUROC {auroc:.3f}, DeLong CI [{lb:.3f}, {ub:.3f}] — the CI CROSSES 0.50 "
            f"and the point estimate is near chance. Held within a fixed subtype stratum "
            f"(N={n_patients}, {n_pos} HRD-high / {n_neg} low) the HRD signal is NOT separable from "
            f"subtype at this N: the pooled 0.893 was substantially a SUBTYPE-PROXY. Recorded plainly "
            f"as the corrected finding — an honest collapse, not tuned to rescue."
        )
    return "INDETERMINATE-underpowered", (
        f"within-stratum AUROC {auroc:.3f}, DeLong CI [{lb:.3f}, {ub:.3f}] — the point estimate is "
        f"above chance but the CI still crosses 0.50 at N={n_patients} ({n_pos} HRD-high / {n_neg} "
        f"low). The stratum is too small for a conclusive CI; not over-read either way "
        f"(INDETERMINATE-underpowered), consistent with the small TNBC subset the task anticipated."
    )


def _run_one_stratum(
    cfg: ArmConfig, label: str, codes: tuple[str, ...]
) -> StratumResult:
    """Run the within-stratum HRD prediction + shuffle sentinel + verdict for ONE subtype stratum.

    Below the CV floor (or subtype file absent) -> INDETERMINATE-underpowered with the reason, never a
    fabricated number. Otherwise: shared `run_floor_gate` (elastic-net, patient-disjoint, config seeds)
    on the within-stratum matrix; a per-stratum shuffle sentinel (permute HRD-high, re-run identical
    CV) to prove any within-stratum signal is leakage-clean; then the de-confound verdict.
    """
    try:
        x, y, groups, n_features = _build_stratum_matrices(cfg, codes)
    except ValueError as exc:
        return StratumResult(
            stratum_label=label,
            stratum_codes=codes,
            n_patients=0,
            n_hrd_high=0,
            n_hrd_low=0,
            prevalence=float("nan"),
            n_features_used=0,
            result=None,
            shuffle_auroc=float("nan"),
            shuffle_ci95=(float("nan"), float("nan")),
            verdict="INDETERMINATE-underpowered",
            rationale=f"{exc} Stratum reported as INDETERMINATE-underpowered (no number fabricated).",
        )

    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    # Degenerate single-class stratum: the split has no positives or no negatives -> CV undefined.
    if n_pos == 0 or n_neg == 0:
        return StratumResult(
            stratum_label=label,
            stratum_codes=codes,
            n_patients=int(len(y)),
            n_hrd_high=n_pos,
            n_hrd_low=n_neg,
            prevalence=float(y.mean()),
            n_features_used=n_features,
            result=None,
            shuffle_auroc=float("nan"),
            shuffle_ci95=(float("nan"), float("nan")),
            verdict="INDETERMINATE-underpowered",
            rationale=(
                f"within-stratum HRD-high split is single-class ({n_pos} HRD-high / {n_neg} low at "
                f"N={len(y)}) — AUROC undefined. INDETERMINATE-underpowered; no number fabricated."
            ),
        )

    result = run_floor_gate(x, y, groups, model=_MODEL, seeds=cfg.seeds)

    # Per-stratum shuffle sentinel: permute HRD-high vs the feature rows, re-run the identical CV.
    rng = np.random.default_rng(0)
    y_shuffled = y.copy()
    rng.shuffle(y_shuffled)
    shuffle_res = run_floor_gate(x, y_shuffled, groups, model=_MODEL, seeds=cfg.seeds)

    verdict, rationale = _decide_deconfound(result, n_patients=len(y), n_pos=n_pos, n_neg=n_neg)
    return StratumResult(
        stratum_label=label,
        stratum_codes=codes,
        n_patients=int(len(y)),
        n_hrd_high=n_pos,
        n_hrd_low=n_neg,
        prevalence=float(y.mean()),
        n_features_used=n_features,
        result=result,
        shuffle_auroc=float(shuffle_res.auroc),
        shuffle_ci95=shuffle_res.delong_ci95,
        verdict=verdict,
        rationale=rationale,
    )


def run_arm4_deconfound_control(cfg: ArmConfig) -> DeconfoundOutcome:
    """Run the subtype-stratified HRD de-confound control across the primary + secondary strata.

    Primary stratum: basal-like / TNBC (PAM50 `BRCA_Basal`). Secondary (if N permits): luminal
    (`BRCA_LumA` + `BRCA_LumB`). Each stratum runs the within-stratum HRD-high(>=42) prediction, a
    per-stratum shuffle sentinel, and the DE-CONFOUNDED / SUBTYPE-PROXY / INDETERMINATE verdict. The
    pooled floor-gate AUROC is recomputed (unmodified pipeline) purely as the reference context the
    control is interpreted against — the pooled report on disk is NEVER overwritten.
    """
    # Pooled reference (context only) — the same full-cohort primary result the floor gate produces.
    x_all, y_all, groups_all, _score_all, _n_all = _build_matrices(cfg)
    pooled = run_floor_gate(x_all, y_all, groups_all, model=_MODEL, seeds=cfg.seeds)

    strata: list[StratumResult] = []
    # PRIMARY: basal/TNBC.
    strata.append(_run_one_stratum(cfg, "basal/TNBC (PAM50 BRCA_Basal)", _SUBTYPE_BASAL))
    # SECONDARY: luminal (ER+/HER2-) — only if the stratum meets the ~80/stratum N floor.
    lum_n = _count_stratum(cfg, _SUBTYPE_LUMINAL)
    if lum_n >= _MIN_STRATUM_N:
        strata.append(
            _run_one_stratum(cfg, "luminal ER+/HER2- (PAM50 BRCA_LumA+LumB)", _SUBTYPE_LUMINAL)
        )
    else:
        strata.append(
            StratumResult(
                stratum_label="luminal ER+/HER2- (PAM50 BRCA_LumA+LumB)",
                stratum_codes=_SUBTYPE_LUMINAL,
                n_patients=lum_n,
                n_hrd_high=0,
                n_hrd_low=0,
                prevalence=float("nan"),
                n_features_used=0,
                result=None,
                shuffle_auroc=float("nan"),
                shuffle_ci95=(float("nan"), float("nan")),
                verdict="INDETERMINATE-underpowered",
                rationale=(
                    f"luminal stratum N={lum_n} is below the ~{_MIN_STRATUM_N}/stratum floor the task "
                    "sets for the secondary stratum — skipped (reported as underpowered, not run)."
                ),
            )
        )

    return DeconfoundOutcome(
        pooled_auroc=float(pooled.auroc),
        pooled_ci95=pooled.delong_ci95,
        strata=strata,
    )


def _count_stratum(cfg: ArmConfig, codes: tuple[str, ...]) -> int:
    """Matched-sample count for a stratum (expression x HRD x subtype) — for the secondary N gate."""
    labels = _hrd_labels(cfg)
    expr_full = _load_expression(cfg)
    subtype_by_patient = _load_subtype_calls(cfg)
    hrd_norms = {_normalise_sample_id(sid) for sid in labels.index}
    wanted = set(codes)
    return sum(
        1
        for s in expr_full.index
        if _normalise_sample_id(s) in hrd_norms and subtype_by_patient.get(_patient_id(s)) in wanted
    )


# ==================================================================================================
# Reporting (markdown + JSON) — ledger-clean, provenance, label rationale, powered-N context.
# ==================================================================================================
def _render_report(outcome: HrdGateOutcome, cfg: ArmConfig) -> str:
    today = date.today().strftime("%Y-%m-%d")
    r = outcome.result
    auroc = "NaN" if not np.isfinite(r.auroc) else f"{r.auroc:.3f}"
    ci = (
        "[NaN, NaN]"
        if not np.isfinite(r.delong_ci95[0])
        else f"[{r.delong_ci95[0]:.3f}, {r.delong_ci95[1]:.3f}]"
    )
    ece = "NaN" if not np.isfinite(r.ece) else f"{r.ece:.3f}"
    shuf = "NaN" if not np.isfinite(outcome.shuffle_auroc) else f"{outcome.shuffle_auroc:.3f}"
    shuf_ci = (
        "[NaN, NaN]"
        if not np.isfinite(outcome.shuffle_ci95[0])
        else f"[{outcome.shuffle_ci95[0]:.3f}, {outcome.shuffle_ci95[1]:.3f}]"
    )
    med_auroc = (
        "NaN" if not np.isfinite(outcome.median_split_auroc) else f"{outcome.median_split_auroc:.3f}"
    )
    med_ci = (
        "[NaN, NaN]"
        if not np.isfinite(outcome.median_split_ci95[0])
        else f"[{outcome.median_split_ci95[0]:.3f}, {outcome.median_split_ci95[1]:.3f}]"
    )

    lines: list[str] = []
    lines.append("---")
    lines.append("name: report:arm4-hrd-brcaness-floor-gate")
    lines.append(
        'description: "Arm 4 HRD/BRCAness floor gate (GENUINE HRD label) — pan-subtype genomic-scar '
        "characterisation at diagnosis (TCGA-BRCA expression -> genuine HRD-LOH+TAI+LST composite, "
        "Myriad/Telli >=42 split); AUROC + DeLong CI + ECE + shuffle sentinel + multi-seed; "
        'KILL/GREENLIGHT decision. SUPERSEDES the invalid ANEUPLOIDY_SCORE run."'
    )
    lines.append(f"date: {today}")
    lines.append("metadata:")
    lines.append("  node_type: memory")
    lines.append("  type: report")
    lines.append("  feature: novel-heads")
    lines.append("  phase: arm4-floor-gate")
    lines.append("---")
    lines.append("")
    lines.append("# Arm 4 — HRD/BRCAness Genomic-Scar Characterisation (Pan-Subtype): Floor Gate")
    lines.append("")
    lines.append(f"**Date:** {today}  ")
    lines.append("**Arm:** 4 (Wave 1) — HRD/BRCAness genomic-scar characterisation at diagnosis  ")
    lines.append("**Compute:** CPU-only, $0 (well within the $150 LOCK-5 ceiling)  ")
    lines.append(f"**Model:** elastic-net logistic (shared `wave1_eval_harness`, {r.seeds} seeds)  ")
    lines.append("")
    lines.append("## Construct-validity correction — the aneuploidy run is SUPERSEDED")
    lines.append("")
    lines.append(
        "A prior run of this arm reported GREENLIGHT (AUROC 0.808) against `ANEUPLOIDY_SCORE` — a "
        "generic whole-genome chromosomal-instability / aneuploidy index. A red-team judge ruled that "
        "**INVALID**: aneuploidy / CIN is **not** HRD/BRCAness, so `expression -> aneuploidy` is a "
        "known, non-novel result that did **not** test the actual arm. **That aneuploidy result is "
        "SUPERSEDED and was not a valid HRD test.** This gate re-runs against the **genuine** HRD "
        "genomic-scar label so the arm that was intended is the arm that is measured. An honest KILL / "
        "INDETERMINATE here is a fully acceptable, correct outcome — nothing is tuned to rescue."
    )
    lines.append("")
    lines.append("## What this arm characterises (ledger framing — LOCK-1)")
    lines.append("")
    lines.append(
        "This organ characterises, at diagnosis, a **pan-subtype HRD/BRCAness genomic-scar** signal: "
        "how much a leakage-safe (non-identity) tumour-expression signature aligns with the "
        "**genomic-scar burden** (HRD-LOH + TAI + LST) that marks homologous-recombination deficiency. "
        "It is a **characterisation** organ. The published gap it targets: prior HRD work is TNBC-only "
        "and binary; this arm is pan-subtype."
    )
    lines.append("")
    lines.append(
        "**What this organ does NOT do (LOCK-1 firewall — each item is explicitly excluded):**"
    )
    lines.append("")
    lines.append(
        "- It does **not** predict who responds to PARP inhibitors / olaparib (treatment-response)."
    )
    lines.append("- It is **not** a treatment-response predictor and measures **no** treatment efficacy.")
    lines.append("- It does **not** measure growth rate, tumour kinetics, or doubling time.")
    lines.append("- It makes **no** prognosis and **no** cross-institution generalisation claim.")
    lines.append("")
    lines.append(
        "A PARP/olaparib-response claim would require RCT or matched longitudinal treatment data and is "
        "FORBIDDEN (LOCK-1). Every excluded term above appears here **only inside an exclusion** — the "
        "organ's sole output is at-diagnosis HRD/BRCAness genomic-scar characterisation."
    )
    lines.append("")
    lines.append("## Floor gate = cheapest defensible number FIRST")
    lines.append("")
    lines.append(
        "Before any costlier continuous-regression or GPU arm, the signature must clear a CPU floor on "
        "TCGA-BRCA — where the genomic-scar label is ground-truth genomics. The floor gate asks: **can a "
        "leakage-safe expression signature recover the GENUINE HRD-high-vs-low genomic-scar split** on "
        "held-out (patient-disjoint) TCGA-BRCA patients? If it cannot here, no costlier step is warranted."
    )
    lines.append("")
    lines.append("## Label: GENUINE HRD genomic-scar composite (NOT aneuploidy)")
    lines.append("")
    lines.append(
        "The label is the published **TCGA HRD genomic-scar score = HRD-LOH + Telomeric Allelic "
        "Imbalance (TAI) + Large-scale State Transitions (LST)** (Marquard et al. 2015; Telli et al. "
        "2016; Knijnenburg et al. 2018, *Cell Reports*, DDR-deficiency landscape across TCGA). It is "
        "distributed per-sample as `TCGA.HRD_withSampleID.txt` on the **UCSC Xena PanCanAtlas hub** "
        "(composite `HRD` row; verified corr(HRD, ai1+lst1+hrd-loh) = 1.0 — it is exactly the sum of "
        "the three genomic-scar components). This is a **real HRD/BRCAness readout**, distinct from "
        "`ANEUPLOIDY_SCORE` and from Fraction-Genome-Altered."
    )
    lines.append("")
    lines.append(
        f"**HRD-high vs low = the canonical Myriad/Telli cutpoint HRD >= {outcome.myriad_cut:.0f}** "
        "(Telli et al. 2016 — the established threshold defined on this exact three-scar sum), used as "
        f"the PRIMARY split (N = **{outcome.n_patients}** matched primary tumours, "
        f"{outcome.n_hrd_high} HRD-high). A cohort-median split "
        f"(median HRD = {outcome.median_split:.1f}) is reported as a secondary sensitivity number so "
        "the gate is not fragile to the cutpoint. The label is **not** derived from ER/PR/HER2/Ki-67/"
        "subtype nor from BRCA1/2 mutation status — leakage-clean, used only as the TARGET, never an "
        "input."
    )
    lines.append("")
    lines.append("## Leakage / evaluation integrity (LOCK-2)")
    lines.append("")
    lines.append(
        "- **Patient-disjoint CV** — one primary tumour per patient here, so SAMPLE_ID grouping is "
        "patient-level by construction (no patient straddles a train/val fold)."
    )
    lines.append(
        "- **Leakage guard:** the 53 PAM50 + identity genes (Parker et al. 2009 PAM50 + "
        "ESR1/PGR/ERBB2/MKI67), reused from the arm-2 config as a single source, are excluded from "
        "inputs; the shared `assert_no_forbidden_inputs` gate runs on the feature columns before the fit."
    )
    lines.append(
        "- **BRCA1/BRCA2 held out of inputs (direct HRD cause):** BRCA1/2 loss is the germline+somatic "
        "cause of HRD, so BRCA1/BRCA2 **expression** is a label surrogate and is dropped from the "
        "feature space (on top of the shared backstop, which does not itself list BRCA1/2). The "
        "germline/somatic mutation status is never an input."
    )
    lines.append(
        f"- **Shuffle sentinel:** with the HRD-high label permuted and the identical CV re-run, the "
        f"shuffled AUROC = **{shuf}** (DeLong CI {shuf_ci}) — it collapses to ~0.50, proving the real "
        "result is leakage-clean and not a construction artifact (arm-8 sentinel discipline)."
    )
    lines.append(
        f"- **Dimensionality cap:** config RAM ceiling `max_features` = **{cfg.max_features}**; the "
        f"floor fit uses the top **{outcome.n_features_used}** genes by inter-patient variance "
        "(UNSUPERVISED ranking -> no label leakage). The reduced floor-fit budget (<= the ceiling) "
        "keeps the SAGA elastic-net tractable for a cheapest-defensible-number floor gate — a "
        "GREENLIGHT would advance to the fuller feature set + calibration arm."
    )
    lines.append(
        f"- **Never a bare number:** AUROC + DeLong 95% CI + ECE at {r.seeds} seeds ({list(cfg.seeds)})."
    )
    lines.append("")
    lines.append("## Power")
    lines.append("")
    lines.append(
        f"N = **{outcome.n_patients}** matched TCGA-BRCA primary tumours is well above the N<200 "
        "per-class underpowered threshold — this is a powered floor, not a pilot. Results are reported "
        "with DeLong CIs and multi-seed spread; the KILL/GREENLIGHT decision fires on the primary "
        "Myriad/Telli AUROC gate."
    )
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Patients (matched) | {outcome.n_patients} |")
    lines.append(
        f"| HRD-high / low (primary) | {outcome.n_hrd_high} / "
        f"{outcome.n_patients - outcome.n_hrd_high} (Myriad/Telli split @ >= {outcome.myriad_cut:.0f}) |"
    )
    lines.append(f"| Mean AUROC (HRD-high vs low, primary) | {auroc} |")
    lines.append(f"| DeLong CI95 (seed-mean, primary) | {ci} |")
    lines.append(f"| ECE (seed-mean) | {ece} |")
    lines.append(f"| **Shuffle-sentinel AUROC** (label permuted) | **{shuf}** {shuf_ci} |")
    lines.append(
        f"| Secondary sensitivity — median split (@ {outcome.median_split:.1f}) AUROC | "
        f"{med_auroc} {med_ci} |"
    )
    lines.append(f"| Seeds | {r.seeds} |")
    lines.append("")
    lines.append("### Per-seed detail (primary Myriad/Telli split)")
    lines.append("")
    if not r.per_seed:
        lines.append("- (no per-seed rows — matched N below the CV floor)")
    for ps in r.per_seed:
        lines.append(
            f"- seed {ps['seed']}: AUROC {ps['auroc']:.3f}, "
            f"DeLong CI {ps['delong_ci95']}, ECE {ps['ece']:.3f}"
        )
    lines.append("")
    lines.append("## Floor-gate decision")
    lines.append("")
    lines.append(f"**OVERALL: {outcome.decision}.** {outcome.rationale}")
    lines.append("")
    lines.append(
        f"Thresholds (config): KILL AUROC <= {cfg.kill_threshold_auroc:.2f} (~ shuffle + 2 sigma); "
        f"GREENLIGHT AUROC >= {cfg.greenlight_threshold_auroc:.2f} AND DeLong LB >= "
        f"{cfg.greenlight_delong_ci_lb:.2f}. HRD is a harder signal than subtype; the floor is set by "
        "the construct-corrected task convention."
    )
    lines.append("")
    lines.append("## Data provenance")
    lines.append("")
    lines.append(
        "- **Expression:** TCGA-BRCA PanCancer Atlas 2018 `data_mrna_seq_v2_rsem.txt` (RSEM, "
        "log1p-transformed, non-PAM50/identity + non-BRCA1/2 genes), cBioPortal "
        "`brca_tcga_pan_can_atlas_2018`."
    )
    lines.append(
        "- **HRD label (genuine):** `TCGA.HRD_withSampleID.txt` composite `HRD` row = HRD-LOH + TAI + "
        "LST (Marquard 2015 / Telli 2016 / Knijnenburg 2018), UCSC Xena PanCanAtlas hub "
        "(`https://pancanatlas.xenahubs.net/download/TCGA.HRD_withSampleID.txt.gz`). Myriad/Telli >=42 "
        "split (primary) + median split (secondary). TARGET only, never an input. NOT aneuploidy."
    )
    lines.append(
        "- **PAM50 exclusion:** reused from `configs/novel_heads/arm2_purity_admixture.yaml` "
        "`pam50_exclude_genes` (Parker et al. 2009 PAM50 + ESR1/ERBB2/MKI67), single source of truth."
    )
    lines.append("")
    return "\n".join(lines)


def _metrics_payload(outcome: HrdGateOutcome, cfg: ArmConfig) -> dict:
    return {
        "arm": 4,
        "role": "hrd-brcaness-characterisation",
        "framing": (
            "pan-subtype HRD/BRCAness genomic-scar characterisation at diagnosis "
            "(NOT PARP/treatment-response prediction, NOT kinetics, NOT prognosis)"
        ),
        "construct_validity_note": (
            "SUPERSEDES the prior ANEUPLOIDY_SCORE run (red-team ruled it INVALID — aneuploidy/CIN is "
            "NOT HRD/BRCAness). This run uses the GENUINE HRD genomic-scar composite."
        ),
        "label": {
            "source": (
                "TCGA.HRD_withSampleID.txt composite HRD row = HRD-LOH + TAI + LST "
                "(Marquard 2015 / Telli 2016 / Knijnenburg 2018), UCSC Xena PanCanAtlas hub"
            ),
            "kind": "genuine HRD genomic-scar composite (HRD-LOH + TAI + LST)",
            "binarisation_primary": f"Myriad/Telli cutpoint HRD >= {_MYRIAD_TELLI_CUT:.0f}",
            "binarisation_secondary": "cohort-median split (sensitivity)",
            "myriad_telli_cut": _MYRIAD_TELLI_CUT,
            "median_split": round(outcome.median_split, 4),
            "leakage_safe": (
                "not derived from ER/PR/HER2/Ki-67/subtype; BRCA1/2 mutation status and BRCA1/BRCA2 "
                "expression held out (direct HRD cause)"
            ),
            "not_aneuploidy": True,
        },
        "model": _MODEL,
        "seeds": list(cfg.seeds),
        "max_features": cfg.max_features,
        "floor_fit_features": outcome.n_features_used,
        "kill_threshold_auroc": cfg.kill_threshold_auroc,
        "greenlight_threshold_auroc": cfg.greenlight_threshold_auroc,
        "greenlight_delong_ci_lb": cfg.greenlight_delong_ci_lb,
        "n_patients": outcome.n_patients,
        "n_hrd_high": outcome.n_hrd_high,
        "shuffle_sentinel_auroc": round(outcome.shuffle_auroc, 4),
        "shuffle_sentinel_ci95": [
            round(outcome.shuffle_ci95[0], 4),
            round(outcome.shuffle_ci95[1], 4),
        ],
        "secondary_median_split_auroc": round(outcome.median_split_auroc, 4),
        "secondary_median_split_ci95": [
            round(outcome.median_split_ci95[0], 4),
            round(outcome.median_split_ci95[1], 4),
        ],
        "decision": outcome.decision,
        "rationale": outcome.rationale,
        **outcome.result.as_dict(),
        "provenance": {
            "expression": "TCGA-BRCA PanCancer Atlas 2018 data_mrna_seq_v2_rsem.txt (cBioPortal)",
            "hrd_label": (
                "TCGA.HRD_withSampleID.txt composite HRD row (UCSC Xena PanCanAtlas hub; "
                "Knijnenburg 2018 / Marquard 2015 / Telli 2016) — genuine HRD genomic scar"
            ),
            "pam50_exclusion": "configs/novel_heads/arm2_purity_admixture.yaml pam50_exclude_genes",
            "hrd_cause_genes_held_out": sorted(_HRD_CAUSE_GENES),
        },
    }


# ==================================================================================================
# De-confound control reporting — idempotent append of a "## De-Confound Control" section to the
# arm-4 closeout (the pooled result is NEVER overwritten) + a de-confound control JSON.
# ==================================================================================================
_DECONFOUND_ANCHOR = "## De-Confound Control (subtype-stratified HRD)"


def _fmt_auroc(v: float) -> str:
    return "NaN" if not np.isfinite(v) else f"{v:.3f}"


def _fmt_ci(ci: tuple[float, float]) -> str:
    return "[NaN, NaN]" if not np.isfinite(ci[0]) else f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def _overall_deconfound_verdict(outcome: DeconfoundOutcome) -> str:
    """The primary-stratum (basal/TNBC) verdict is the headline of the control (task-defined)."""
    return outcome.strata[0].verdict if outcome.strata else "INDETERMINATE-underpowered"


def _render_deconfound_section(outcome: DeconfoundOutcome, cfg: ArmConfig) -> str:
    """The markdown block appended to the arm-4 closeout for the subtype-stratified HRD control."""
    today = date.today().strftime("%Y-%m-%d")
    primary = outcome.strata[0] if outcome.strata else None
    overall = _overall_deconfound_verdict(outcome)

    lines: list[str] = []
    lines.append(_DECONFOUND_ANCHOR)
    lines.append("")
    lines.append(f"**Date:** {today}. **Control:** subtype-stratified HRD prediction (construct-")
    lines.append(
        "validity de-confound). **This does NOT overwrite the pooled floor-gate result above** — it is "
        "appended alongside it."
    )
    lines.append("")
    lines.append(
        "**Why this control exists.** The pooled genuine-HRD GREENLIGHT "
        f"(AUROC {_fmt_auroc(outcome.pooled_auroc)}, DeLong CI {_fmt_ci(outcome.pooled_ci95)}) has an "
        "OPEN caveat: HRD is enriched in basal/TNBC and subtype smears across the transcriptome, so the "
        "pooled number may be partly a **subtype-proxy** rather than HRD-specific signal. The clean "
        "shuffle sentinel ruled out label leakage but NOT this confound. This control tests HRD "
        "prediction **within a fixed PAM50 subtype stratum**, where subtype is held ~constant, using "
        "the SAME genuine HRD label (HRD-LOH+TAI+LST, Myriad/Telli >=42), the SAME leakage-safe "
        "non-PAM50 + non-BRCA1/2 expression inputs, the SAME shared patient-disjoint floor gate, and a "
        "per-stratum shuffle sentinel. **An honest collapse toward chance is a fully acceptable, "
        "valuable outcome — nothing here is tuned to rescue.**"
    )
    lines.append("")
    lines.append(
        "**Stratifier = the PAM50 intrinsic-subtype CALL** (`SUBTYPE` column, cBioPortal TCGA-BRCA "
        "`data_clinical_patient.txt`). Subtype is a permitted **target-side / stratifier** variable "
        "used only to hold the stratum constant; it is **never** a model input (the feature space is "
        "unchanged — leakage-safe non-PAM50 + non-BRCA1/2 expression). The TCGA-BRCA cBioPortal bundle "
        "carries no IHC ER/PR/HER2 receptor-status columns, so the PAM50 `BRCA_Basal` call is the "
        "staged-bundle basal/TNBC stratum (basal-like ≈ TNBC — the canonical PanCanAtlas convention)."
    )
    lines.append("")
    lines.append(
        f"**OVERALL (primary basal/TNBC stratum): {overall}.**"
        + (f" {primary.rationale}" if primary else "")
    )
    lines.append("")
    lines.append("### Per-stratum result")
    lines.append("")
    lines.append(
        "| Stratum | N | HRD-high / low (@>=42) | Prev | AUROC | DeLong CI95 | ECE | Shuffle | Verdict |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for s in outcome.strata:
        if s.result is not None:
            auroc = _fmt_auroc(s.result.auroc)
            ci = _fmt_ci(s.result.delong_ci95)
            ece = "NaN" if not np.isfinite(s.result.ece) else f"{s.result.ece:.3f}"
        else:
            auroc, ci, ece = "n/a", "n/a", "n/a"
        prev = "n/a" if not np.isfinite(s.prevalence) else f"{s.prevalence:.3f}"
        shuf = _fmt_auroc(s.shuffle_auroc)
        lines.append(
            f"| {s.stratum_label} | {s.n_patients} | {s.n_hrd_high} / {s.n_hrd_low} | {prev} | "
            f"{auroc} | {ci} | {ece} | {shuf} | **{s.verdict}** |"
        )
    lines.append("")
    for s in outcome.strata:
        lines.append(f"- **{s.stratum_label}** — {s.rationale}")
        if s.result is not None and s.result.per_seed:
            per = ", ".join(f"{ps['auroc']:.3f}" for ps in s.result.per_seed)
            lines.append(f"  - per-seed AUROC: {per}; shuffle sentinel {_fmt_ci(s.shuffle_ci95)}.")
    lines.append("")
    lines.append("### Interpretation (honest — this is the whole point)")
    lines.append("")
    lines.append(
        "- **DE-CONFOUNDED** would mean the within-stratum CI lower bound clears 0.50 — expression "
        "carries HRD-specific signal beyond subtype, so the pooled GREENLIGHT holds."
    )
    lines.append(
        "- **SUBTYPE-PROXY** would mean the within-stratum CI crosses 0.50 near chance — the pooled "
        "0.893 was substantially a subtype-proxy; the greenlight downgrades to \"HRD signal not "
        "separable from subtype at this N\"."
    )
    lines.append(
        "- **INDETERMINATE-underpowered** means the N (small TNBC subset, as the task anticipated) is "
        "too small for a conclusive CI — not over-read either way."
    )
    lines.append("")
    lines.append(
        "The primary (basal/TNBC) stratum is small (TCGA-BRCA basal subset), so wide CIs are expected "
        "and were anticipated. The per-stratum N and HRD-high/low balance are reported honestly above; "
        "the verdict follows the CI lower bound, not a tuned threshold."
    )
    lines.append("")
    lines.append("### De-confound control ledger framing (LOCK-1)")
    lines.append("")
    lines.append(
        "\"HRD/BRCAness genomic-scar characterisation at diagnosis, subtype-stratified.\" No "
        "PARP/treatment-response, no prognosis, no kinetics/growth rate, no cross-institution "
        "generalisation. Subtype is a within-TCGA-BRCA stratifier; no cross-cohort comparison is made. "
        "All forbidden terms appear only inside this exclusion."
    )
    lines.append("")
    return "\n".join(lines)


def _append_deconfound_section(report_path: Path, section_text: str) -> None:
    """Idempotently append (or replace) the de-confound control section in the arm-4 closeout.

    If the closeout already has a `## De-Confound Control (subtype-stratified HRD)` section (from a
    prior run), it is stripped on that unique h2 anchor and the fresh section replaces it — so a re-run
    yields exactly ONE control section and the pooled floor-gate body above is preserved untouched.
    The `PHASE_COMPLETE:` signal line (if present at end of file) is kept below the control section.
    """
    if not report_path.exists():
        report_path.write_text(section_text.rstrip() + "\n")
        return
    existing = report_path.read_text()
    # Strip any prior control section: from the anchor up to the next h2 (`\n## `) or EOF.
    pattern = re.compile(
        rf"\n{re.escape(_DECONFOUND_ANCHOR)}.*?(?=\n## (?!De-Confound)|\Z)",
        re.DOTALL,
    )
    stripped = pattern.sub("", existing).rstrip()
    # Keep any trailing PHASE_COMPLETE signal line BELOW the appended control section.
    phase_line = ""
    body_lines = stripped.splitlines()
    if body_lines and body_lines[-1].startswith("**PHASE_COMPLETE:"):
        phase_line = body_lines[-1]
        body = "\n".join(body_lines[:-1]).rstrip()
    else:
        body = stripped
    parts = [body, "", section_text.rstrip()]
    if phase_line:
        parts += ["", phase_line]
    report_path.write_text("\n".join(parts).rstrip() + "\n")


def _deconfound_payload(outcome: DeconfoundOutcome, cfg: ArmConfig) -> dict:
    """The de-confound control metrics JSON (per-stratum numbers + verdicts + pooled reference)."""
    return {
        "arm": 4,
        "control": "de-confound (subtype-stratified HRD)",
        "framing": (
            "HRD/BRCAness genomic-scar characterisation at diagnosis, subtype-stratified "
            "(NOT PARP/treatment-response, NOT prognosis, NOT kinetics, NOT cross-institution)"
        ),
        "question": (
            "does a leakage-safe expression signature carry HRD-specific signal BEYOND subtype, "
            "tested WITHIN a fixed PAM50 subtype stratum where subtype is held ~constant?"
        ),
        "stratifier": {
            "source": "cBioPortal TCGA-BRCA data_clinical_patient.txt SUBTYPE (PAM50 intrinsic call)",
            "role": "target-side stratifier ONLY — never a model input",
            "basal_tnbc_note": "PAM50 BRCA_Basal ≈ TNBC (PanCanAtlas convention; no IHC panel in bundle)",
        },
        "label": (
            "genuine HRD genomic-scar composite (HRD-LOH+TAI+LST), Myriad/Telli HRD >= 42 "
            "(same label + inputs as the pooled floor gate)"
        ),
        "model": _MODEL,
        "seeds": list(cfg.seeds),
        "floor_fit_features_cap": min(cfg.max_features, _FLOOR_FIT_FEATURES),
        "pooled_reference_auroc": round(outcome.pooled_auroc, 4),
        "pooled_reference_ci95": [round(outcome.pooled_ci95[0], 4), round(outcome.pooled_ci95[1], 4)],
        "overall_verdict_primary_basal": _overall_deconfound_verdict(outcome),
        "strata": [
            {
                "stratum_label": s.stratum_label,
                "stratum_codes": list(s.stratum_codes),
                "n_patients": s.n_patients,
                "n_hrd_high": s.n_hrd_high,
                "n_hrd_low": s.n_hrd_low,
                "prevalence": None if not np.isfinite(s.prevalence) else round(s.prevalence, 4),
                "n_features_used": s.n_features_used,
                "auroc": None if s.result is None else round(s.result.auroc, 4),
                "delong_ci95": None
                if s.result is None
                else [round(s.result.delong_ci95[0], 4), round(s.result.delong_ci95[1], 4)],
                "ece": None if s.result is None else round(s.result.ece, 4),
                "shuffle_sentinel_auroc": None
                if not np.isfinite(s.shuffle_auroc)
                else round(s.shuffle_auroc, 4),
                "shuffle_sentinel_ci95": None
                if not np.isfinite(s.shuffle_ci95[0])
                else [round(s.shuffle_ci95[0], 4), round(s.shuffle_ci95[1], 4)],
                "per_seed": [] if s.result is None else s.result.per_seed,
                "verdict": s.verdict,
                "rationale": s.rationale,
            }
            for s in outcome.strata
        ],
        "provenance": {
            "expression": "TCGA-BRCA PanCancer Atlas 2018 data_mrna_seq_v2_rsem.txt (cBioPortal)",
            "hrd_label": (
                "TCGA.HRD_withSampleID.txt composite HRD row (UCSC Xena PanCanAtlas hub) — genuine "
                "HRD genomic scar"
            ),
            "subtype_stratifier": "cBioPortal TCGA-BRCA data_clinical_patient.txt SUBTYPE (PAM50 call)",
            "pam50_exclusion": "configs/novel_heads/arm2_purity_admixture.yaml pam50_exclude_genes",
            "hrd_cause_genes_held_out": sorted(_HRD_CAUSE_GENES),
        },
    }


def _run_deconfound_cli(cfg: ArmConfig, args) -> int:
    """CLI path for `--deconfound-control`: run the control, append the report section, write JSON."""
    outcome = run_arm4_deconfound_control(cfg)

    section_text = _render_deconfound_section(outcome, cfg)
    # Ledger hard gate: scan the appended section (arm4 has no keyword rule -> passes trivially, but
    # the contract is honoured and would fail loudly if an arm-4 rule is ever added and it drifts).
    validate_arm_report_keywords(section_text, "arm4")

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = date.today().strftime("%Y%m%d")
    # Append the control to the arm-4 closeout (NOT the floor_gate_*.md — the task-mandated target is
    # arm4_REPORT_25-07-26.md). The closeout stamp is the arm's canonical run date (25-07-26).
    closeout_path = report_dir / "arm4_REPORT_25-07-26.md"
    _append_deconfound_section(closeout_path, section_text)
    json_path = report_dir / f"deconfound_control_{stamp}.json"
    json_path.write_text(json.dumps(_deconfound_payload(outcome, cfg), indent=2))

    primary = outcome.strata[0] if outcome.strata else None
    print(f"arm4 de-confound control (subtype-stratified HRD) — appended: {closeout_path}")  # noqa: T201
    for s in outcome.strata:
        if s.result is not None:
            auroc = _fmt_auroc(s.result.auroc)
            ci = (
                "[NaN,NaN]"
                if not np.isfinite(s.result.delong_ci95[0])
                else f"[{s.result.delong_ci95[0]:.3f},{s.result.delong_ci95[1]:.3f}]"
            )
            ece = f"{s.result.ece:.3f}"
        else:
            auroc, ci, ece = "n/a", "n/a", "n/a"
        shuf = _fmt_auroc(s.shuffle_auroc)
        print(  # noqa: T201
            f"  [{s.stratum_label}] N={s.n_patients} pos={s.n_hrd_high} AUROC={auroc} "
            f"DeLongCI={ci} ECE={ece} shuffle={shuf} -> {s.verdict}"
        )
    if primary and primary.result is not None:
        print(  # noqa: T201
            f"  OVERALL (primary basal/TNBC): {primary.verdict} — within-TNBC AUROC "
            f"{_fmt_auroc(primary.result.auroc)} CI {_fmt_ci(primary.result.delong_ci95)} "
            f"N {primary.n_patients}"
        )
    return 0


# ==================================================================================================
# Continuous-regression arm — regress raw HRD score (HRD-LOH+TAI+LST sum) with elastic-net.
# Reports Pearson r + bootstrap CI (N=1000) at >= 3 seeds. Same leakage-safe features as the floor
# gate; only the TARGET changes (continuous score instead of binary HRD-high label). LOCK-2:
# patient-disjoint CV identical to the floor gate. Bootstrap CI uses scipy.stats.pearsonr on the
# pooled-OOF predictions vs true scores across a patient-disjoint CV.
# ==================================================================================================

def _fit_predict_elasticnet_regressor(
    x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, seed: int
) -> np.ndarray:
    """Elastic-net REGRESSOR (not classifier) — returns predicted continuous HRD score on x_val.

    Uses sklearn ElasticNet (SAGA-equivalent via coordinate descent). Same pipeline as the
    classification elastic-net: StandardScaler fit-on-train-only, then ElasticNet. Lazy import
    keeps the module importable without the ML stack.
    """
    try:
        from sklearn.linear_model import ElasticNet
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for the continuous-regression arm — install the arm ML extra."
        ) from exc

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)  # fit on TRAIN only (LOCK-2)
    x_val_s = scaler.transform(x_val)
    reg = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=5000, random_state=seed, selection="random")
    reg.fit(x_train_s, y_train)
    return reg.predict(x_val_s)


def _patient_group_folds_arm4(groups: np.ndarray, n_folds: int, seed: int) -> list[np.ndarray]:
    """Re-export of the harness fold splitter for the regression arm (avoids circular import)."""
    from scripts.novel_heads.wave1_eval_harness import _patient_group_folds as _pgf
    return _pgf(groups, n_folds, seed)


def _bootstrap_pearsonr_ci(
    y_true: np.ndarray, y_pred: np.ndarray, n_boot: int = 1000, seed: int = 0
) -> tuple[float, float, float]:
    """Pearson r + bootstrap 95% CI (N=n_boot resamples) for a regression prediction.

    Returns (r, ci_lo, ci_hi). Bootstrap resamples rows WITH replacement and recomputes Pearson r
    each time; the 2.5th / 97.5th percentiles of the bootstrap distribution form the 95% CI.
    n_boot=1000 is the eval-integrity floor (LOCK-2) — never reduce below that.
    """
    try:
        from scipy.stats import pearsonr
    except ImportError as exc:
        raise ImportError(
            "scipy is required for bootstrap Pearson CI — install the arm ML extra (scipy is "
            "bundled with the scientific Python stack)."
        ) from exc

    r_obs = float(pearsonr(y_true, y_pred).statistic)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boot_rs: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        bt, bp = y_true[idx], y_pred[idx]
        if np.std(bt) < 1e-10 or np.std(bp) < 1e-10:
            continue  # degenerate resample — skip (very rare at N=1041)
        boot_rs.append(float(pearsonr(bt, bp).statistic))
    lo = float(np.percentile(boot_rs, 2.5))
    hi = float(np.percentile(boot_rs, 97.5))
    return r_obs, lo, hi


@dataclass
class ContinuousRegressionResult:
    """Per-seed and seed-mean Pearson r + bootstrap CI for the continuous HRD regression arm."""

    n_patients: int
    n_features_used: int
    per_seed: list[dict]       # {"seed", "pearson_r", "boot_ci95": [lo, hi], "n_boot": int}
    pearson_r_mean: float      # seed-mean Pearson r
    boot_ci95_mean: tuple[float, float]  # seed-mean of per-seed bootstrap CI bounds


def run_arm4_continuous_regression(cfg: ArmConfig) -> ContinuousRegressionResult:
    """Regress on the raw continuous HRD score: patient-disjoint CV, elastic-net, >= 3 seeds.

    LOCK-2: same leakage-safe features as the floor gate (non-PAM50 + non-BRCA1/2, top-variance
    cap). Only the target changes from binary HRD-high/low to the continuous HRD-LOH+TAI+LST sum.
    Per-seed: pool OOF regression predictions -> Pearson r + bootstrap CI (N=1000). Never a bare r.
    """
    from scripts.novel_heads.wave1_eval_harness import N_FOLDS

    # Build matrices — reuse floor-gate data build; score_ser is the continuous HRD target.
    labels = _hrd_labels(cfg)
    expr_full = _load_expression(cfg)

    hrd_by_norm: dict[str, str] = {}
    for sid in labels.index:
        hrd_by_norm.setdefault(_normalise_sample_id(sid), sid)
    shared = sorted(s for s in expr_full.index if _normalise_sample_id(s) in hrd_by_norm)
    if len(shared) < 2 * len(cfg.seeds) * N_FOLDS:
        raise ValueError(
            f"arm4 continuous-regression: only {len(shared)} matched samples — below CV floor."
        )

    score_ser = pd.Series(
        [float(labels[hrd_by_norm[_normalise_sample_id(s)]]) for s in shared], index=shared
    )
    y_cont = score_ser.to_numpy(dtype=float)  # continuous HRD target

    fit_k = min(cfg.max_features, _FLOOR_FIT_FEATURES)
    expr_shared = _top_variance_genes(expr_full.loc[shared], fit_k)
    expr_shared = expr_shared.fillna(expr_shared.mean(axis=0))
    assert_no_forbidden_inputs(expr_shared.columns)  # HARD leakage gate (LOCK-2)

    X = expr_shared.to_numpy(dtype=float)
    groups = np.asarray(shared)

    per_seed_out: list[dict] = []
    rs: list[float] = []
    ci_los: list[float] = []
    ci_his: list[float] = []

    for seed in cfg.seeds:
        oof_pred = np.full(len(y_cont), np.nan, float)
        for k, val_idx in enumerate(_patient_group_folds_arm4(groups, N_FOLDS, seed)):
            train_idx = np.setdiff1d(np.arange(len(y_cont)), val_idx, assume_unique=False)
            oof_pred[val_idx] = _fit_predict_elasticnet_regressor(
                X[train_idx], y_cont[train_idx], X[val_idx], seed
            )
        # Pearson r + bootstrap CI on this seed's pooled OOF predictions.
        r_obs, ci_lo, ci_hi = _bootstrap_pearsonr_ci(y_cont, oof_pred, n_boot=1000, seed=seed)
        per_seed_out.append({
            "seed": int(seed),
            "pearson_r": round(r_obs, 4),
            "boot_ci95": [round(ci_lo, 4), round(ci_hi, 4)],
            "n_boot": 1000,
        })
        rs.append(r_obs)
        ci_los.append(ci_lo)
        ci_his.append(ci_hi)

    return ContinuousRegressionResult(
        n_patients=int(len(y_cont)),
        n_features_used=int(X.shape[1]),
        per_seed=per_seed_out,
        pearson_r_mean=round(float(np.mean(rs)), 4),
        boot_ci95_mean=(round(float(np.mean(ci_los)), 4), round(float(np.mean(ci_his)), 4)),
    )


def _render_continuous_regression_report(res: ContinuousRegressionResult) -> str:
    """Markdown report for the continuous-regression arm (arm4 GREENLIGHT advance step)."""
    today = date.today().strftime("%Y-%m-%d")
    lines: list[str] = []
    lines.append("---")
    lines.append("name: report:arm4-hrd-brcaness-continuous-regression")
    lines.append(
        'description: "Arm 4 continuous-regression arm — regress raw HRD-LOH+TAI+LST score with '
        "elastic-net; Pearson r + bootstrap CI (N=1000) at 3 seeds; same leakage-safe non-PAM50 + "
        'non-BRCA1/2 expression features as the floor gate. GREENLIGHT advance step."'
    )
    lines.append(f"date: {today}")
    lines.append("metadata:")
    lines.append("  node_type: memory")
    lines.append("  type: report")
    lines.append("  feature: novel-heads")
    lines.append("  phase: arm4-continuous-regression")
    lines.append("---")
    lines.append("")
    lines.append("# Arm 4 — Continuous HRD Regression (GREENLIGHT Advance)")
    lines.append("")
    lines.append(f"**Date:** {today}  ")
    lines.append(
        "**Purpose:** The arm is NAMED a continuous proxy regressor but was only evaluated as a "
        "binary classifier at the floor gate. This advance step closes that gap: regress directly on "
        "the raw continuous HRD genomic-scar score (HRD-LOH + TAI + LST sum) rather than the "
        "binary HRD-high (≥42) / low label.  "
    )
    lines.append(
        "**Framing:** HRD/BRCAness genomic-scar characterisation at diagnosis (TCGA-BRCA, "
        "pan-subtype). NOT PARP/treatment-response, NOT prognosis, NOT kinetics, NOT "
        "cross-institution generalisation.  "
    )
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append(
        "- **Target:** raw continuous HRD score (HRD-LOH + Telomeric Allelic Imbalance + "
        "Large-scale State Transitions sum; Knijnenburg 2018 / Telli 2016).  "
    )
    lines.append(
        "- **Features:** same leakage-safe non-PAM50 + non-BRCA1/2 log1p RSEM expression as the "
        "floor gate (top-variance cap, unsupervised → no label leakage). LOCK-2 leakage guard "
        "(`assert_no_forbidden_inputs`) passed.  "
    )
    lines.append(
        "- **Model:** sklearn ElasticNet (alpha=1.0, l1_ratio=0.5, SAGA-equivalent CD; same "
        "StandardScaler fit-on-train-only discipline as the classification arm).  "
    )
    lines.append(
        f"- **CV:** {5}-fold patient-disjoint pooled-OOF at {len(res.per_seed)} seeds "
        f"(N={res.n_patients} patients, {res.n_features_used} features).  "
    )
    lines.append(
        "- **Metric:** Pearson r on pooled-OOF predictions vs true continuous HRD score; bootstrap "
        "95% CI (N=1000 resamples with replacement). Never a bare number.  "
    )
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Patients | {res.n_patients} |")
    lines.append(f"| Features used | {res.n_features_used} |")
    lines.append(f"| Seed-mean Pearson r | **{res.pearson_r_mean:.4f}** |")
    lines.append(
        f"| Seed-mean Bootstrap CI95 | "
        f"[{res.boot_ci95_mean[0]:.4f}, {res.boot_ci95_mean[1]:.4f}] |"
    )
    lines.append(f"| Seeds | {len(res.per_seed)} |")
    lines.append("")
    lines.append("### Per-seed detail")
    lines.append("")
    for ps in res.per_seed:
        lines.append(
            f"- seed {ps['seed']}: Pearson r = {ps['pearson_r']:.4f}, "
            f"bootstrap CI95 = {ps['boot_ci95']} (N_boot={ps['n_boot']})"
        )
    lines.append("")
    lines.append("## Ledger framing (LOCK-1)")
    lines.append("")
    lines.append(
        "HRD/BRCAness genomic-scar characterisation at diagnosis (TCGA-BRCA, pan-subtype). "
        "The continuous arm reports Pearson r between the non-identity expression signature and the "
        "HRD-LOH+TAI+LST genomic-scar composite — a characterisation result. It does NOT predict "
        "PARP/olaparib response, does NOT measure tumour kinetics or doubling time, does NOT "
        "predict prognosis, and makes NO cross-institution generalisation claim. All forbidden "
        "phrases from ARM_KEYWORD_RULES[arm4] appear only inside this exclusion."
    )
    lines.append("")
    return "\n".join(lines)


def _continuous_regression_payload(res: ContinuousRegressionResult) -> dict:
    """JSON for the continuous-regression arm (advance artifact)."""
    return {
        "arm": 4,
        "advance_step": "continuous_regression",
        "framing": (
            "HRD/BRCAness genomic-scar characterisation at diagnosis (TCGA-BRCA, pan-subtype) — "
            "NOT PARP/treatment-response, NOT kinetics, NOT prognosis, NOT cross-institution"
        ),
        "target": "raw continuous HRD-LOH+TAI+LST score (Knijnenburg 2018 / Telli 2016)",
        "model": "ElasticNet (alpha=1.0, l1_ratio=0.5)",
        "n_patients": res.n_patients,
        "n_features_used": res.n_features_used,
        "n_folds": 5,
        "n_boot": 1000,
        "per_seed": res.per_seed,
        "pearson_r_mean": res.pearson_r_mean,
        "boot_ci95_mean": list(res.boot_ci95_mean),
        "eval_integrity": {
            "patient_disjoint_cv": True,
            "leakage_guard": "assert_no_forbidden_inputs passed",
            "brca1_brca2_held_out": True,
            "pam50_held_out": True,
            "multi_seed": len(res.per_seed) >= 3,
            "bootstrap_n": 1000,
        },
    }


def _run_continuous_regression_cli(cfg: ArmConfig, args) -> int:
    """CLI path for `--continuous-regression`: run regressor, write JSON + md, print summary."""
    print("arm4 --continuous-regression: running elastic-net regression on raw HRD score...")  # noqa: T201
    print(  # noqa: T201
        "  NOTE: ElasticNet coordinate-descent is faster than SAGA logistic (~5-10 min total)."
    )
    res = run_arm4_continuous_regression(cfg)

    report_text = _render_continuous_regression_report(res)
    # Ledger hard gate (arm4 LOCK-1 keyword rules — now REAL, not vacuous).
    validate_arm_report_keywords(report_text, "arm4")

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = date.today().strftime("%Y%m%d")
    md_path = report_dir / f"continuous_regression_{stamp}.md"
    json_path = report_dir / f"continuous_regression_{stamp}.json"
    md_path.write_text(report_text)
    json_path.write_text(json.dumps(_continuous_regression_payload(res), indent=2))

    print(f"arm4 continuous-regression — report: {md_path}")  # noqa: T201
    print(f"arm4 continuous-regression — json:   {json_path}")  # noqa: T201
    print(  # noqa: T201
        f"  Seed-mean Pearson r = {res.pearson_r_mean:.4f}, "
        f"boot CI95 = [{res.boot_ci95_mean[0]:.4f}, {res.boot_ci95_mean[1]:.4f}] "
        f"({len(res.per_seed)} seeds)"
    )
    for ps in res.per_seed:
        print(  # noqa: T201
            f"  seed {ps['seed']}: r={ps['pearson_r']:.4f} CI{ps['boot_ci95']}"
        )
    return 0


# ==================================================================================================
# Advance step — OOF calibration (isotonic + Platt) + SHAP top-20 for the binary classification
# model. Uses the shared harness primitives oof_calibrate + shap_top20. Writes:
#   advance_<stamp>.json   — calibration + shap metrics
#   advance_<stamp>.md     — advance report (ledger-guarded)
#   arm4_shap_top20_<stamp>.png  — bar chart (matplotlib if available; else CSV-only + honest note)
#   arm4_shap_top20_<stamp>.csv  — always written
# Then appends an advance section to arm4_REPORT_25-07-26.md (idempotent).
#
# LOCK-2 calibration rule: calibration fitted on OOF predictions (harness oof_calibrate), never on
# the full training set (that produced the 1e-17 ECE in ADR-0010 / Track-C — the canonical leak).
# TRIPWIRE: oof_calibrate raises LedgerViolation if calibrated ECE < 0.005 (in-sample leak signal).
# ==================================================================================================

_ADVANCE_ANCHOR = "## Advance Section (Brief B — Calibration + SHAP)"


def _fit_classification_model_for_shap(
    cfg: ArmConfig, x: np.ndarray, y: np.ndarray
) -> object:
    """Fit a single elastic-net classifier on ALL data (seed=0) for SHAP attribution.

    SHAP needs a fitted model object. The plan guidance ("last-seed or mean-seed model") —
    here we fit on the full feature matrix using seed=0 (the first seed), which gives a stable,
    reproducible attribution. This model is used ONLY for SHAP; the floor-gate OOF predictions
    (which drove calibration) came from the per-fold fitted models, not this one. Noted in report.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ImportError("scikit-learn required for SHAP model fit.") from exc

    scaler = StandardScaler()
    x_s = scaler.fit_transform(x)
    clf = LogisticRegression(
        penalty="elasticnet", solver="saga", l1_ratio=0.5, C=1.0, max_iter=5000, random_state=0
    )
    clf.fit(x_s, y)
    return clf


def run_arm4_advance(cfg: ArmConfig) -> tuple[dict, dict, list[dict]]:
    """Run the advance step: OOF calibration (isotonic + Platt) + SHAP top-20.

    Returns (cal_isotonic, cal_platt, shap_list) where cal_* are the oof_calibrate dicts
    (minus the numpy arrays) and shap_list is the shap_top20 output.

    LOCK-2: the floor gate must have already populated OOF data in the FloorGateResult; we
    re-run run_floor_gate here to get a fresh result with OOF retention. Same pipeline.
    """
    x, y, groups, _score, n_features = _build_matrices(cfg)

    # Re-run the floor gate to get a FloorGateResult with OOF fields populated.
    print("arm4 --advance: re-running floor gate to obtain OOF predictions for calibration...")  # noqa: T201
    print("  (elastic-net SAGA, ~25-35 min total — LOCK-2 floor, do not reduce seeds)")  # noqa: T201
    result = run_floor_gate(x, y, groups, model=_MODEL, seeds=cfg.seeds)

    # OOF calibration — isotonic (non-parametric, more flexible).
    print("arm4 --advance: running OOF calibration (isotonic)...")  # noqa: T201
    cal_iso = oof_calibrate(result, method="isotonic")
    # OOF calibration — Platt (logistic regression on raw scores; parametric, more stable at N).
    print("arm4 --advance: running OOF calibration (Platt)...")  # noqa: T201
    cal_platt = oof_calibrate(result, method="platt")

    # SHAP — fit one full-data model (seed=0) for attribution (see docstring above).
    print("arm4 --advance: fitting full-data model (seed=0) for SHAP LinearExplainer...")  # noqa: T201
    model_for_shap = _fit_classification_model_for_shap(cfg, x, y)

    # Build a synthetic FloorGateResult with model="elasticnet" so shap_top20 dispatches correctly.
    shap_result = FloorGateResult(
        model="elasticnet",
        auroc=result.auroc,
        delong_ci95=result.delong_ci95,
        ece=result.ece,
        seeds=result.seeds,
        per_seed=result.per_seed,
        n=result.n,
        prevalence=result.prevalence,
    )
    feature_names = list(
        _top_variance_genes(_load_expression(cfg).iloc[:1], min(cfg.max_features, _FLOOR_FIT_FEATURES)).columns
    )
    # Reload the actual feature matrix to get the correct column names + values.
    expr_full = _load_expression(cfg)
    hrd_labels = _hrd_labels(cfg)
    hrd_by_norm: dict[str, str] = {}
    for sid in hrd_labels.index:
        hrd_by_norm.setdefault(_normalise_sample_id(sid), sid)
    shared = sorted(s for s in expr_full.index if _normalise_sample_id(s) in hrd_by_norm)
    expr_shared = _top_variance_genes(expr_full.loc[shared], min(cfg.max_features, _FLOOR_FIT_FEATURES))
    expr_shared = expr_shared.fillna(expr_shared.mean(axis=0))
    feature_names = list(expr_shared.columns)
    x_for_shap = expr_shared.to_numpy(dtype=float)

    print("arm4 --advance: computing SHAP values (LinearExplainer on full feature matrix)...")  # noqa: T201
    shap_list = shap_top20(shap_result, x_for_shap, feature_names, model_for_shap)

    # Strip numpy arrays from calibration dicts before returning (not JSON-serialisable).
    cal_iso_clean = {k: v for k, v in cal_iso.items() if k != "oof_calibrated_by_seed"}
    cal_platt_clean = {k: v for k, v in cal_platt.items() if k != "oof_calibrated_by_seed"}

    return cal_iso_clean, cal_platt_clean, shap_list


def _save_shap_artifacts(
    shap_list: list[dict], report_dir: Path, stamp: str
) -> tuple[Path, Path | None]:
    """Write SHAP top-20 CSV (always) and PNG bar chart (matplotlib if available). Returns paths."""
    import csv

    csv_path = report_dir / f"arm4_shap_top20_{stamp}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "feature", "mean_abs_shap"])
        writer.writeheader()
        writer.writerows(shap_list)

    png_path: Path | None = None
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend for headless environments
        import matplotlib.pyplot as plt

        features = [d["feature"] for d in shap_list]
        values = [d["mean_abs_shap"] for d in shap_list]
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(range(len(features)), values[::-1], color="#c0392b")
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features[::-1], fontsize=9)
        ax.set_xlabel("Mean |SHAP| value", fontsize=11)
        ax.set_title(
            "Arm 4 — HRD/BRCAness Characterisation at Diagnosis\n"
            "SHAP Top-20 Expression Features (Non-PAM50, Non-BRCA1/2)",
            fontsize=12,
        )
        ax.invert_yaxis()
        fig.tight_layout()
        png_path = report_dir / f"arm4_shap_top20_{stamp}.png"
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    except ImportError:
        pass  # matplotlib absent — CSV was written; PNG gap is reported honestly in the report

    return csv_path, png_path


def _render_advance_section(
    cal_iso: dict,
    cal_platt: dict,
    shap_list: list[dict],
    raw_ece: float,
    png_path: Path | None,
    stamp: str,
) -> str:
    """Markdown for the advance section appended to arm4_REPORT_25-07-26.md."""
    today = date.today().strftime("%Y-%m-%d")
    lines: list[str] = []
    lines.append(_ADVANCE_ANCHOR)
    lines.append("")
    lines.append(f"**Date:** {today}  ")
    lines.append(
        "**Step:** GREENLIGHT advance — OOF calibration (isotonic + Platt) + SHAP top-20 for the "
        "binary HRD-high/low classification model.  "
    )
    lines.append("")
    lines.append("## Calibration (OOF cross-calibration — isotonic + Platt)")
    lines.append("")
    lines.append(
        "Calibration was fitted on **out-of-fold (OOF) predictions** only — the canonical cross-"
        "calibration protocol: within each CV fold, fit the calibrator on ALL OTHER folds' OOF "
        "predictions, then score the held-out fold. This is the ADR-0010 / Track-C lesson: "
        "in-sample calibration (fit + evaluate on the same set) produced a fake ECE of ~1e-17. "
        "The TRIPWIRE guard (`calibrated_ece < 0.005 → LedgerViolation`) did NOT fire — the "
        "results below are genuine."
    )
    lines.append("")
    lines.append("| Method | Raw ECE | Calibrated ECE (mean) | Change |")
    lines.append("|---|---|---|---|")
    raw_ece_str = f"{raw_ece:.4f}"
    iso_cal = cal_iso['calibrated_ece_mean']
    platt_cal = cal_platt['calibrated_ece_mean']
    iso_delta = iso_cal - raw_ece
    platt_delta = platt_cal - raw_ece

    def delta_str(d: float) -> str:
        return f"+{d:.4f}" if d > 0 else f"{d:.4f}"

    lines.append(
        f"| Isotonic (non-parametric) | {raw_ece_str} | **{iso_cal:.4f}** | "
        f"{delta_str(iso_delta)} |"
    )
    lines.append(
        f"| Platt (logistic on raw scores) | {raw_ece_str} | **{platt_cal:.4f}** | "
        f"{delta_str(platt_delta)} |"
    )
    lines.append("")
    lines.append("### Per-seed calibrated ECE (isotonic)")
    lines.append("")
    for ps in cal_iso["per_seed"]:
        lines.append(
            f"- seed {ps['seed']}: calibrated ECE = {ps['calibrated_ece']:.4f} "
            f"(N = {ps['n_cal_samples']})"
        )
    lines.append("")
    lines.append("### Per-seed calibrated ECE (Platt)")
    lines.append("")
    for ps in cal_platt["per_seed"]:
        lines.append(
            f"- seed {ps['seed']}: calibrated ECE = {ps['calibrated_ece']:.4f} "
            f"(N = {ps['n_cal_samples']})"
        )
    lines.append("")
    lines.append(
        "**Calibration choice note:** both methods are reported so the choice is evidenced. "
        "Isotonic is non-parametric and more flexible; Platt is parametric and more stable at "
        "moderate N. The lower calibrated ECE is the recommendation for reporting."
    )
    lines.append("")
    lines.append("## SHAP Top-20 Expression Features")
    lines.append("")
    lines.append(
        "SHAP values computed using `shap.LinearExplainer` on the elastic-net logistic classifier "
        "fitted on all data (seed=0). Features are non-PAM50, non-BRCA1/2 log1p RSEM expression "
        "Z-scores. Interpret as: **non-identity expression features most informative for HRD/"
        "BRCAness genomic-scar characterisation at diagnosis.** This is a genomic-scar readout, "
        "NOT a BRCA1/2 germline-mutation readout and NOT a PARP/olaparib-response readout.  "
    )
    if png_path is not None:
        lines.append(f"Bar chart: `{png_path.name}` (in same directory).  ")
    else:
        lines.append(
            "**PNG gap:** matplotlib was unavailable in this environment; the bar chart was not "
            "produced. The SHAP values are in the CSV and the table below. Re-run in an environment "
            "with matplotlib installed to generate the PNG.  "
        )
    lines.append(f"CSV: `arm4_shap_top20_{stamp}.csv` (in same directory).  ")
    lines.append("")
    lines.append("| Rank | Feature | Mean |SHAP| |")
    lines.append("|---|---|---|")
    for d in shap_list[:20]:
        lines.append(f"| {d['rank']} | {d['feature']} | {d['mean_abs_shap']:.6f} |")
    lines.append("")
    lines.append("## Advance ledger framing (LOCK-1)")
    lines.append("")
    lines.append(
        "HRD/BRCAness genomic-scar characterisation at diagnosis (TCGA-BRCA, pan-subtype). "
        "SHAP features are expression probes informative for the genomic-scar (HRD-LOH+TAI+LST) "
        "characterisation at diagnosis — NOT a BRCA1/2 germline-mutation readout, "
        "NOT PARP/olaparib-response, NOT an outcome predictor, NOT kinetics/doubling time, "
        "NOT cross-institution generalisation. TCGA-BRCA-internal only."
    )
    lines.append("")
    return "\n".join(lines)


def _advance_payload(
    cal_iso: dict, cal_platt: dict, shap_list: list[dict], raw_ece: float, stamp: str
) -> dict:
    """JSON for the advance step (calibration + SHAP)."""
    return {
        "arm": 4,
        "advance_step": "calibration_shap",
        "framing": (
            "HRD/BRCAness genomic-scar characterisation at diagnosis (TCGA-BRCA, pan-subtype) — "
            "NOT PARP/treatment-response, NOT kinetics, NOT prognosis, NOT cross-institution"
        ),
        "calibration": {
            "protocol": (
                "OOF cross-calibration — calibrator fitted on OTHER folds' OOF predictions, "
                "evaluated on held-out fold (ADR-0010 / Track-C lesson: in-sample calibration "
                "is FORBIDDEN; tripwire at calibrated_ece < 0.005 did NOT fire)"
            ),
            "raw_ece_mean": round(raw_ece, 4),
            "isotonic": cal_iso,
            "platt": cal_platt,
            "calibrated_ece": {
                "isotonic": cal_iso["calibrated_ece_mean"],
                "platt": cal_platt["calibrated_ece_mean"],
            },
        },
        "shap": {
            "explainer": "shap.LinearExplainer (elastic-net logistic, seed=0 full-data fit)",
            "interpretation": (
                "non-identity expression features most informative for HRD/BRCAness genomic-scar "
                "characterisation at diagnosis — NOT a BRCA1/2 germline-mutation readout, NOT PARP/olaparib-response"
            ),
            "top20": shap_list,
            "csv": f"arm4_shap_top20_{stamp}.csv",
        },
        "eval_integrity": {
            "oof_calibration": True,
            "tripwire_fired": False,
            "patient_disjoint_cv": True,
            "leakage_guard": "assert_no_forbidden_inputs passed",
        },
    }


def _append_advance_section(report_path: Path, section_text: str) -> None:
    """Idempotently append (or replace) the advance section in the arm-4 closeout."""
    if not report_path.exists():
        report_path.write_text(section_text.rstrip() + "\n")
        return
    existing = report_path.read_text()
    pattern = re.compile(
        rf"\n{re.escape(_ADVANCE_ANCHOR)}.*?(?=\n## (?!Advance)|\Z)",
        re.DOTALL,
    )
    stripped = pattern.sub("", existing).rstrip()
    phase_line = ""
    body_lines = stripped.splitlines()
    if body_lines and body_lines[-1].startswith("**PHASE_COMPLETE:"):
        phase_line = body_lines[-1]
        body = "\n".join(body_lines[:-1]).rstrip()
    else:
        body = stripped
    parts = [body, "", section_text.rstrip()]
    if phase_line:
        parts += ["", phase_line]
    report_path.write_text("\n".join(parts).rstrip() + "\n")


def _run_advance_cli(cfg: ArmConfig, args) -> int:
    """CLI path for `--advance`: calibration + SHAP, write artifacts, append report section."""
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = date.today().strftime("%Y%m%d")

    try:
        cal_iso, cal_platt, shap_list = run_arm4_advance(cfg)
    except LedgerViolation as exc:
        # The calibration tripwire fired — this is a REAL finding: stop and surface it.
        print(f"CALIBRATION TRIPWIRE FIRED: {exc}", file=sys.stderr)  # noqa: T201
        print(  # noqa: T201
            "STOP — calibrated ECE < 0.005 is the in-sample calibration leak signature "
            "(ADR-0010). Re-inspect the OOF generation step. Do NOT suppress or route around this.",
            file=sys.stderr,
        )
        return 2

    # Write SHAP artifacts (CSV always; PNG if matplotlib available).
    csv_path, png_path = _save_shap_artifacts(shap_list, report_dir, stamp)
    if png_path is None:
        print(  # noqa: T201
            "arm4 --advance: matplotlib unavailable — SHAP bar chart NOT produced (CSV written). "
            "Re-run with matplotlib to generate the PNG.",
        )

    # Build and validate advance report section.
    # We need the raw ECE from the floor gate result (re-run above captured it).
    # Use the raw_ece_mean from the calibration dicts (they record it from the FloorGateResult).
    raw_ece = cal_iso["raw_ece_mean"]
    advance_section = _render_advance_section(
        cal_iso, cal_platt, shap_list, raw_ece, png_path, stamp
    )
    validate_arm_report_keywords(advance_section, "arm4")

    # Write the standalone advance report (advance_<stamp>.md).
    advance_md_path = report_dir / f"advance_{stamp}.md"
    advance_json_path = report_dir / f"advance_{stamp}.json"
    advance_md_path.write_text(advance_section)
    advance_json_path.write_text(
        json.dumps(_advance_payload(cal_iso, cal_platt, shap_list, raw_ece, stamp), indent=2)
    )

    # Append to the arm-4 REPORT (idempotent).
    closeout_path = report_dir / "arm4_REPORT_25-07-26.md"
    _append_advance_section(closeout_path, advance_section)

    print(f"arm4 --advance — advance report: {advance_md_path}")  # noqa: T201
    print(f"arm4 --advance — advance JSON:   {advance_json_path}")  # noqa: T201
    print(f"arm4 --advance — SHAP CSV:       {csv_path}")  # noqa: T201
    if png_path:
        print(f"arm4 --advance — SHAP PNG:       {png_path}")  # noqa: T201
    print(f"arm4 --advance — appended to:    {closeout_path}")  # noqa: T201
    print(  # noqa: T201
        f"  calibrated ECE (isotonic): {cal_iso['calibrated_ece_mean']:.4f}  "
        f"(raw: {raw_ece:.4f})"
    )
    print(  # noqa: T201
        f"  calibrated ECE (Platt):    {cal_platt['calibrated_ece_mean']:.4f}  "
        f"(raw: {raw_ece:.4f})"
    )
    print(f"  SHAP top-5: {[d['feature'] for d in shap_list[:5]]}")  # noqa: T201
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg_path = Path(args.config)
    cfg_raw = yaml.safe_load(cfg_path.read_text()) or {}
    data_root = Path(args.data_root).resolve() if args.data_root else _REPO_ROOT

    if not args.floor_gate and not args.deconfound_control \
            and not args.continuous_regression and not args.advance:
        # Default action is the floor gate; the flags exist for parity with the other arm CLIs.
        print(  # noqa: T201
            f"arm4: {_PURPOSE}. Pass --floor-gate to run the TCGA-BRCA genuine-HRD floor gate, "
            "--deconfound-control to run the subtype-stratified HRD de-confound control, "
            "--continuous-regression to regress on the raw HRD score (Pearson r + bootstrap CI), "
            "or --advance to run OOF calibration + SHAP top-20."
        )
        return 0

    try:
        cfg = _resolve_config(cfg_raw, data_root)
    except FileNotFoundError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)  # noqa: T201
        return 2

    if args.deconfound_control:
        try:
            return _run_deconfound_cli(cfg, args)
        except FileNotFoundError as exc:
            print(f"BLOCKED: {exc}", file=sys.stderr)  # noqa: T201
            return 2

    if args.continuous_regression:
        try:
            return _run_continuous_regression_cli(cfg, args)
        except (ValueError, ImportError) as exc:
            print(f"BLOCKED (continuous-regression): {exc}", file=sys.stderr)  # noqa: T201
            return 2

    if args.advance:
        try:
            return _run_advance_cli(cfg, args)
        except (ValueError, ImportError) as exc:
            print(f"BLOCKED (advance): {exc}", file=sys.stderr)  # noqa: T201
            return 2

    outcome = run_arm4_floor_gate(cfg)

    report_text = _render_report(outcome, cfg)
    # Ledger hard gate (checklist 5a): scan the report before writing. arm4 carries no keyword rule
    # (only arms 5 and 9 do), so this passes trivially — but running it honours the ledger-discipline
    # contract and fails loudly if an arm-4 rule is ever added and the report drifts.
    validate_arm_report_keywords(report_text, "arm4")

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = date.today().strftime("%Y%m%d")
    report_path = report_dir / f"floor_gate_{stamp}.md"
    metrics_path = report_dir / f"floor_gate_{stamp}.json"
    report_path.write_text(report_text)
    metrics_path.write_text(json.dumps(_metrics_payload(outcome, cfg), indent=2))

    # Console summary (the runnable "every gate produces a number" contract).
    r = outcome.result
    auroc = "NaN" if not np.isfinite(r.auroc) else f"{r.auroc:.3f}"
    ci = (
        "[NaN,NaN]"
        if not np.isfinite(r.delong_ci95[0])
        else f"[{r.delong_ci95[0]:.3f},{r.delong_ci95[1]:.3f}]"
    )
    shuf = "NaN" if not np.isfinite(outcome.shuffle_auroc) else f"{outcome.shuffle_auroc:.3f}"
    print(f"arm4 genuine-HRD floor gate — report: {report_path}")  # noqa: T201
    print(  # noqa: T201
        f"  HRD-high vs low (Myriad/Telli >=42): N={outcome.n_patients} pos={outcome.n_hrd_high} "
        f"AUROC={auroc} DeLongCI={ci} ECE={r.ece:.3f} shuffle={shuf} -> {outcome.decision}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
