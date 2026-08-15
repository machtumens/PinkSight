# data/ — cohort acquisition & expected layout

This repository's git tree ships **no patient data and no weight binaries**. Every cohort below must
be obtained by you, directly from its source, **under that source's own data-use agreement (DUA)**.
Nothing here redistributes patient data, and this repo holds no DUA on your behalf. (The 15 G5
imaging-encoder weights are distributed separately as GitHub Release assets under CC-BY-NC-4.0 —
fetch + verify with `scripts/fetch_weights.py`; see `results/TRAINED_ARTIFACTS.md` and
`LICENSE-WEIGHTS.md`.)

## Firewall first (LOCK-1)

PinkSight is **not one model** — it is a set of **per-cohort organs**, each with its own number on
its own cohort. Duke, TCGA-BRCA, the ISPY2/MAMA-MIA companion, and fastMRI-NYU are
**patient-disjoint** (different institutions, different patients, no shared sample). A number from
one cohort is **never** pooled with, compared to, or juxtaposed against a number from another. Every
task is **characterisation at diagnosis** (subtype / aggressiveness). See
`docs/CLAIM_LEDGER.md` for the full allowed/forbidden boundary and `results/Table2_results.md` for
each within-cohort result.

## Cohorts — acquisition & DUA

### Track A — Duke-Breast-Cancer-MRI (committed)
- **Portal:** The Cancer Imaging Archive (TCIA).
- **License / DUA:** TCIA CC-BY-NC — attribute the source; no commercial redistribution. Obtain the
  collection directly from TCIA under its terms.
- **Modality:** pre-operative DCE-MRI (pre + post-contrast phases) + a structured clinical table.
- **Cohort size:** 922 patients with MRI + clinical labels (no whole-slide images).
- **Task:** subtype characterisation at diagnosis (Luminal A vs Triple-Negative). See
  `docs/DATA_CARD.md` for the N-waterfall, the patient-level split, and the CI leakage assertions.

### Track B — TCGA-BRCA (H&E histology; reported alongside, firewalled)
- **Portal:** NCI Genomic Data Commons (GDC) — the TCGA-BRCA diagnostic whole-slide images.
- **License / DUA:** follow the GDC / TCGA data-use and attribution policy for the open-access
  diagnostic slides.
- **Modality:** H&E diagnostic WSIs, encoded to **frozen** foundation-model slide features
  (TITAN, UNI2-h) — only small heads are trained on top; the encoders themselves are not retrained.
- **Cohort size:** 640 patients (Luminal A vs basal-like).
- **Firewall:** patient-disjoint from Duke. A Track-B histology number is never compared with a
  Track-A (Duke) number — see `results/TRAINED_ARTIFACTS.md` and `results/Table2_results.md`.

### Companion — ISPY2 / MAMA-MIA (quasi-external + pCR pilot)
- **Portal:** the MAMA-MIA public multi-cohort DCE-MRI release (which aggregates ISPY2, ISPY1, NACT,
  and Duke-overlapping cases); ISPY2 is likewise available through its own public release.
- **License / DUA:** obtain under the MAMA-MIA / ISPY2 release terms.
- **Modality:** DCE-MRI + clinical features.
- **Use:** per-cohort quasi-external calibration / robustness, and the ISPY2 pCR pilot — always
  within-cohort, never a cross-institution generalisation claim.
- **Leakage control (mandatory before any "external" use):** MAMA-MIA overlaps Duke by **291**
  patients (all `DUKE_*`). Drop `dataset==DUKE` first, leaving a clean external pool of **1215**
  (ISPY2 980 + ISPY1 171 + NACT 64) with zero Duke overlap. This de-duplication is enforced in CI
  (`tests/`); see `docs/DATA_CARD.md` for the audited derivation.

### Standalone — fastMRI-NYU (NYU-only encoder; ADR-0016)
- **Portal:** the NYU fastMRI Breast DCE-MRI release.
- **License / DUA:** register for, and agree to, the NYU fastMRI data-use agreement before download.
- **Modality:** DCE-MRI (pre + post-contrast phases).
- **Cohort size:** 300 patients; the 51 verified-normal patients are quarantined (hard manifest
  interlock), leaving malignant 90 / benign 159; the patient-level split is **shipped** in the source
  metadata (240 train / 60 test), not carved here.
- **Firewall:** a **separate, standalone** cohort — never pooled or joined with Duke / MAMA-MIA /
  TCGA in any trained weight (LOCK-1). Evaluation is NYU-internal only. See
  `docs/adr/0016-fastmri-nyu-standalone-encoder-deferred-fusion-slot.md`.

## Expected on-disk layout (what the ported code reads)

Paths are **relative to the repository root**. None of these files ship with the repo — you populate
them from the cohorts above. These are the exact locations the ported training scripts in `scripts/`
read (spot-checked against `scripts/train_g3_hierarchical.py`, `scripts/train_fastmri_nyu.py`, and
`scripts/trackb_mil_cv.py`):

```
data/
├── manifest_v1.csv                       # Track A — Duke DCE-MRI + clinical patient manifest
├── fastmri_processed_nyu/                # fastMRI-NYU — processed DCE-MRI cubes
├── fastmri_processed_nyu_masks/          # fastMRI-NYU — lesion masks
└── pathology/
    ├── bags/uni/                         # Track B — per-slide UNI2-h feature bags (MIL input)
    └── features/
        ├── TCGA_TITAN_features.pkl       # Track B — frozen TITAN slide embeddings (N=640)
        └── tcga_brca_titan_manifest.csv  # Track B — TITAN slide manifest
```

## Verifying your data & weights

The 15 **G5 imaging-encoder** weights (~1.9 GB) are distributed as **GitHub Release assets** (not in
the git tree). Fetch and verify them in one step:

```
python3 scripts/fetch_weights.py            # download from the release + SHA-256-verify each file
sha256sum -c scripts/g5_weights.sha256      # or verify files you already have, from the repo root
```

Every **other** artifact (cohort data you placed, frozen-feature inputs, the clinical-anchor and
Track-B heads) stays hash-manifest-only — verify each against its row in
**`results/TRAINED_ARTIFACTS.md`**:

```
sha256sum <path>          # compare the digest to the row in results/TRAINED_ARTIFACTS.md
```

`results/TRAINED_ARTIFACTS.md` (human-readable) and `scripts/g5_weights.sha256` (machine-readable)
carry the SHA-256 for each file. The G5 weights are licensed CC-BY-NC-4.0 (see `LICENSE-WEIGHTS.md`),
separately from the Apache-2.0 code license.
