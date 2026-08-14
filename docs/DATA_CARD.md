# DATA_CARD.md — PinkSight cohorts

> The most important file in a medical-AI repo (manual §8). It is both memory and defence: it is
> what you show when asked "how do you know there is no leakage?". **No number is filled until the
> G0 audit (`scripts/audit_ki67.py`) produces it.** Do not invent N or the split.

## Cohort — Duke-Breast-Cancer-MRI (Track A, committed)
- Source: TCIA · License: TCIA (CC-BY-NC, attribute, no commercial redistribution)
- Modality: pre-op DCE-MRI (pre + post-contrast phases) + structured clinical table
- N (published): 922 patients with MRI + clinical labels · no whole-slide images

## N-waterfall (filled by G0 — 2026-06-21; see `reports/audit_report.md`)
```
922  total patients
 └─ has DCE-MRI (proxy) ......... 922   (true per-image DCE check deferred to P03 imaging manifest)
     └─ has subtype label ....... 922   (Mol Subtype: luminal-like 595 · TNBC 164 · ER/PR+HER2+ 104 · HER2 59)
         └─ in LOCK-3 scope ..... 759   (luminal-like 595 + TNBC 164; ~3.6:1, TNBC minority class)
             └─ Luminal A ....... UNDETERMINED   (luminal-like = A + B; the A/B split needs Ki-67)
     └─ USABLE NUMERIC Ki-67 .... N = 0   <-- Ki-67 absent from the clinical table; gates Head 2 (O-2)
```
Source sha256 `8ef0945c9f75`. Literature flags 922 vs ~457 after QC; image-level QC runs at P03.
O-2 / LOCK-3 outcome (Ki-67 head status + the Luminal-A-vs-luminal-like reframe) is PENDING the science
decision — not yet recorded in `decisions.md`.

## Labels
- Subtype: Luminal A vs Triple-Negative (from IHC ER/PR/HER2/Ki-67) — the LABEL, not an input
- Ki-67: continuous %, binary call at the fixed **14%** St. Gallen cutoff (contingent on G0 N)
- ROI annotation (D1, resolved 2026-06-22): Duke ships **one 3D bounding box per patient** (`Annotation_Boxes.xlsx`, 922 rows) — **boxes, not pixel masks**. XAI [6.3] scores box-hit/box-IoU; pixel-mask XAI needs the H0 segmenter (decisions.md D1 / PROPOSED-3).

## Split (carved at G0/G1 → `configs/split_v2.yaml`; does not exist yet — do NOT regenerate once made)
- Patient-level only (never image/slice-level)
- Quasi-external (scanner/year) holdout carved FIRST, sealed until the end
- 5-fold stratified CV on the remainder

## Leakage assertions (enforced in CI — `tests/test_leakage.py`)
- FORBIDDEN ∩ classifier inputs == ∅, where FORBIDDEN = {ER, PR, HER2, Ki-67, Mol Subtype, Oncotype}
- Duke ∩ MAMA-MIA: **291** shared patients (all MAMA-MIA `DUKE_*`; `DUKE_NNN↔Breast_MRI_NNN` maps 291/291, 0 mismatch). Drop `dataset==DUKE` before ANY "external"/H0 claim → clean **1215** external pool (ISPY2 980 + ISPY1 171 + NACT 64). Enforced in `tests/test_splits.py`; audited 2026-06-22.
- Preprocessing (Nyul landmarks, scalers, imputers) fit on the train fold only

## Known gaps / bias
- Single-institution, cross-sectional, all-cancer cohort (no healthy controls)
- Report performance by breast density + available demographics (Caucasian-skewed corpus)
- Ki-67 is NOT an enumerated Duke field (radiogenomics text only) — confirm count before committing
  Ki-67 as a primary head (decisions.md O-2).
- **No clinical Ki-67 anywhere on disk (2026-07-10, extends G0 N=0 to cell level).** The ONLY Ki-67
  signal in on-disk data is **`MKI67` gene expression** in the TCGA (1084 pts) + METABRIC (1985 pts)
  genomics matrices — a **molecular proxy, NOT clinical IHC %**, both gate-locked in Track B (LOCK-6).
  Confirmed absent from every on-disk clinical table (Duke / MAMA-MIA / METABRIC-clinical /
  TCGA-clinical). Implication: a real Ki-67 analysis is possible ONLY as a **continuous cross-cohort
  MKI67 proliferation sub-study in Track B** (never a Track-A head — no MRI overlap); continuous only
  ([1.2-R]), and MKI67 mRNA ≠ IHC Ki-67 → label the surrogate honestly. O-2 (descriptive) stands for
  Track A. Backlog / forward register (decisions.md EXT-3).

## G5 external de-dup (LOCK-2 leakage prep — 2026-07-09)
- Script: `scripts/dedup_external.py` (real data + `--selfcheck` assert-check). Real MAMA-MIA table
  WAS available locally (`data/mamma_mia/clinical_and_imaging_info.xlsx`, gitignored).
- Result (re-derived, matches the 2026-06-22 audit): total MAMA-MIA **1506** → dropped
  `dataset==DUKE` **291** → clean quasi-external pool **1215** (ISPY2 980 + ISPY1 171 + NACT 64).
  Post-dedup Duke∩external overlap = **0** (asserted). No `DUKE_*` id survives in the clean list.
- Output: `reports/external_clean_ids.txt` (1215 clean external patient_ids, one per line).
- Self-check: PASS (synthetic 2-overlap list → dedup removes exactly the 2; mislabeled-DUKE guard fires).
- This is leakage prep only — independent of any G3 headline. The clean list is the sealed
  external pool for G5; do not add Duke patients back in.

## External-pool add candidates (BACKLOG — G5, NOT acquired, NOT locked; decisions.md EXT-1/EXT-2)
> Flagged candidates to grow the sealed G5 external pool. Per-cohort calibration/robustness only —
> NOT a cross-institution generalisation claim (LOCK-1). IHC used for label derivation stays a
> FORBIDDEN model input (LOCK-2). Nothing acquired; count usable N before trusting any figure.
- **AMBL (Advanced-MRI-Breast-Lesions)** — TCIA, CC BY 4.0. T1 DCE-MRI (1 pre + 4 post), **632
  sessions**, structured clinical for **~200 patients**, re-labelable IHC ("receptor status + Ki-67
  if applicable"). The only genuinely-new free DCE-MRI cohort not already in the Duke/MAMA-MIA pool.
  Caveats: **~646 GB** download + H0-seg inference compute (heavy vs ~$0 budget, LOCK-5); IHC
  completeness **unverified** — open the AMBL supporting spreadsheet and count malignant patients
  with COMPLETE ER∧PR∧HER2 before trusting any usable N; relabel only to {luminal-like, TNBC} (Ki-67
  cutoff drift → no clean Luminal A; matches PROPOSED-2). Source:
  https://www.cancerimagingarchive.net/collection/advanced-mri-breast-lesions/ (decisions.md EXT-1)
- **TCGA-Breast-Radiogenomics** (TCGA Breast Phenotype group DCE-MRI) — **~84 patients**, full PAM50
  + ER/PR/HER2 + Oncotype/MammaPrint, **no Ki-67**. Free, CC BY 3.0. Second pick IF AMBL's usable N
  disappoints. **HARD de-dup gate (LOCK-2):** same patient IDs as the Track-B TCGA-BRCA cohort → MUST
  de-dup against Track-B TCGA patients (`scripts/dedup_external.py` logic) or exclude from any
  "external" claim — leakage-dressed-as-external otherwise. Source:
  https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=19039112 (decisions.md EXT-2)

## Cohort — fastMRI-NYU (standalone encoder, NYU-only; ADR-0016, EXECUTED 2026-08-05)
- Source: NYU fastMRI Breast · a **separate, standalone** cohort — NOT part of Track A/B, NOT pooled
  or joined with Duke/MAMA-MIA/TCGA in any trained weight (LOCK-1 firewall)
- Modality: DCE-MRI (pre + post-contrast phases), 96³ cached cubes
- N (total): **300 patients**, age 25–82 · negative/normal **51** (quarantined, hard-interlocked out of
  every H-char/H5 batch — see below) · malignant **90** · benign **159**
- Split: patient-level, **shipped** in the source xlsx `Data split` column (not carved by us) — 240
  train / 60 test; H-char cohort excludes the 51 normals → train **199** (72 malig/127 benign) / sealed
  test **50** (18 malig/32 benign, prevalence 0.36); H5 age head trains on all 249 non-normal patients
- Leakage: patient-disjoint train/test (asserted); Nyul intensity-normalization landmarks fit
  **train-fold only** (LOCK-2 assert — raises if any test-fold ID is consumed by the fit; verified PASS
  on the real run, 0/60 test IDs); images-only encoder (no biomarker/clinical columns as input)
- **51 verified-normal patients are quarantined behind a manifest** (`normals_quarantine_manifest.json`)
  that every H-char/H5 dataloader CI-asserts absent from its batches — mechanical, not honor-system.
  The anomaly/novelty head (H6) that would use them is **NOT trained** (early-detection risk; needs its
  own `/red-team` before the manifest unlocks).
- Result (NYU-internal only, NEVER juxtaposed with any Duke number): H-char malignant-vs-benign
  ensemble AUROC **0.599** [DeLong 0.4303, 0.7676] → **NO-GO**; H5 chronological-age MAE **11.68 yr**
  [9.02, 14.63]. See `docs/adr/0016-fastmri-nyu-standalone-encoder-deferred-fusion-slot.md`,
  `decisions.md [ADR-0016-EXECUTED]` (2026-08-05), and
  `process/general-plans/completed/fastmri-encoder-deferred-fusion_04-08-26/`.
