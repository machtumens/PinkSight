# PinkSight — Real results (real data, verified)

This folder is the **real** side of the submission: verified PinkSight results on **real patient
cohorts**, with the frozen results tables and a checksummed manifest of the real trained artifacts.

It is deliberately separate from the notebooks and CLI in the parent folder, which run on **fake
(synthetic) data** and are wiring checks, **not** results.

## Read this first — the firewall (LOCK-1)

PinkSight is **not one model**. It is a set of **per-cohort organs**, each with its own number.
Two rules are absolute:

1. **Never pool or juxtapose numbers across cohorts.** Duke, TCGA-BRCA, ISPY2, and fastMRI-NYU are
   **patient-disjoint** — different institutions, different patients, no shared sample. A Track-B
   histology number (TCGA) is **never** compared to, contrasted with, or added to a Track-A number
   (Duke). Doing so would be a cross-institution generalisation claim, which is forbidden.
2. **Every number is characterisation at diagnosis** — subtype / aggressiveness. Never early
   detection, never growth-rate / kinetics, never a group-vs-group clinical claim.

There is **no single "stitched" model trained on real data**, and there cannot be one: no real
patient carries every modality, so there is no joint sample to train it on. The unified fusion
architecture is only *trainable-as-one* on synthetic data (that is the parent-folder wiring check).
On real data the modalities stay in separate, firewalled organs.

## What's here

| File | What it is |
|---|---|
| `Table1_cohorts.md` | The cohorts (Duke, ISPY2, TCGA-BRCA, fastMRI-NYU) — N, split, leakage controls. Frozen paper supplement. |
| `Table2_results.md` | **Every reportable number**, each with its cohort, estimator, DeLong CI, and verdict. Frozen paper supplement. |
| `TRAINED_ARTIFACTS.md` | Manifest of the **real trained artifacts on disk** — path, size, SHA-256, cohort, verdict. The large weights are *not* bundled (1.9 GB); this manifest is how you locate and verify them. |

The full manuscript lives at `reports/paper/pinksight_paper.md` (+ figures in `reports/paper/figures/`,
TRIPOD-AI checklist in `reports/paper/tables/Table3_tripod_ai.md`). It is not copied here to avoid a
drifting duplicate while it is under review — read it from source.

## The headline (Track A — Duke, within-cohort)

An **honest null**. On the Duke cohort, the **clinical stream alone** (LogReg, AUROC **0.708**) is the
only modality with meaningful subtype discrimination. Every imaging and imaging-fusion rung sits
below it (radiomics 0.567 → unimodal-MRI 0.518 → flat-fusion 0.636 → hierarchical #4 0.599 → MoE #7
0.650 ± 0.018), and the pre-registered ΔAUROC ≥ 0.03 fusion margin is **not demonstrated** (paired
DeLong vs the 0.708 anchor is negative for every fusion rung; the study is under-powered for +0.03 at
N=613, not that fusion is *rejected*). The contribution is the **leakage-safe fusion architecture**
and the **characterised imaging-fusion information ceiling** — not "imaging works". See
`Table2_results.md` for every number with its CI.

Track B (TCGA-BRCA H&E histology, standalone) and the companion pilots are reported **alongside**,
each behind its own ADR firewall — see `Table2_results.md`, always within-cohort.

_Sources: `reports/paper/tables/` (frozen tables), `decisions.md` (authority), the per-gate
`reports/` JSONs cited in each table row._
