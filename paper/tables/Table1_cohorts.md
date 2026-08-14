# Table 1 — Cohorts and datasets

> Paper supplement. Cohort-level facts verified against `DATA_CARD.md` (G0 audit 2026-06-21,
> sha256 `8ef0945c`) + `decisions.md`. Classes are **luminal-like (Luminal A+B, unsplit) vs TNBC** —
> Luminal A alone is undetermined (Ki-67 N=0). Per-variable descriptive stats (age, Nottingham grade,
> stage) are computed from `data/manifest_v1.csv` on the train folds only at write-time — not shown here.

| Cohort | Role | Modality | N (analysis) | Class balance | Source | Split | Key notes |
|---|---|---|---|---|---|---|---|
| **Duke-Breast-Cancer-MRI** | Track A — headline | DCE-MRI + structured clinical | 759 in-scope; dev **624** (5-fold CV) / **613** (H6-basis) | luminal-like 595 / TNBC 164 (~3.6:1) | TCIA, single-institution | patient-level 5-fold; quasi-external (scanner/year) holdout carved **first** | 1 3D bounding box/patient (not pixel masks); demographically skewed corpus; Ki-67 absent → descriptive-only; Luminal A vs B undetermined |
| **ISPY2** | Quasi-external (G5) | structured clinical | **739** balanced (from clean pool 1215) | LumA 381 / TNBC 358 (~48% TNBC) | multi-site trial, via MAMA-MIA pool | sealed external; Duke ∩ external = 0 (de-dup asserted) | 48% vs 21% TNBC prevalence shift drives external ECE; LogReg estimator |
| **TCGA-BRCA** | Track B — co-result | H&E whole-slide (+ genomics) | **640** | LumA 475 / Basal 165 (~2.9:1) | TCGA | patient-level, 3 seeds; LOSO robustness | frozen TITAN / UNI2-h encoders; Duke ∩ TCGA ID overlap = 0; reported alongside Track A, never juxtaposed with it |
| **ISPY2 / MAMA-MIA (pCR)** | Pilot (ADR-0007) | DCE-MRI radiomics | **980** | pCR prevalence 0.322 | ISPY2 | patient-level, 3 seeds | distinct cohort from Duke; both gates NO-GO; no cross-cohort claim |
| **fastMRI-NYU** | Standalone (ADR-0016) | DCE-MRI | 300 (H-char 199 train / 50 test) | malignant 90 / benign 159 / normal 51 (quarantined) | NYU fastMRI Breast | shipped split 240/60 | NYU-only, never juxtaposed with Duke; H-char AUROC 0.599 NO-GO; anomaly head not trained |

**Leakage controls (all cohorts, LOCK-2):** patient-level sealed splits; the FORBIDDEN set
{ER, PR, HER2, Ki-67, Mol Subtype, Oncotype} ∩ classifier inputs = ∅ (CI-asserted); preprocessing
(Nyúl landmarks, scalers, imputers) fit on the train fold only; Duke ∩ MAMA-MIA = 291 shared patients
dropped before any external claim (clean external pool 1215).

*Source: `DATA_CARD.md`, `reports/audit_report.md`, `scripts/dedup_external.py`, `decisions.md`.*
