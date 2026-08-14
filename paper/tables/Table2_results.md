# Table 2 — Results matrix (all reportable numbers)

> Paper supplement. Every number verified against its source JSON + `decisions.md`. **Always name the
> estimator** (0.708 = LogReg, not FT-Transformer 0.634). Every AUROC carries its DeLong CI95. Track B is
> reported alongside Track A and is **never** compared or juxtaposed with the Duke cohort (ADR-0015).
> `†` = pooled-OOF ensemble CI (3-seed per-patient-mean OOF, one DeLong CI) — replaces the earlier
> single-seed (ci_seed=0) CI for the #4/#7 fusion rungs (model-integrity-remediation, 12-08-26); the
> reported point stays the 3-seed mean.

## Track A — Duke subtype characterisation (luminal-like vs TNBC, N=613 MRI+clinical bi-modal intersection, patient-level 5-fold OOF, 3 seeds)

| Result | Estimator | AUROC [DeLong CI95] | ECE | Shuffle | Verdict | Source |
|---|---|---|---|---|---|---|
| Radiomics floor (G1) | LogReg (107 feat) | 0.567 [0.510, 0.624] | 0.068 | — | LOCK-4 floor | `G1_baseline/metrics.json` |
| **Clinical-alone (anchor)** | **LogReg (C=1.0)** | **0.708 [0.642, 0.747]** | 0.0196 | 0.505 | **only modality w/ meaningful discrimination** | `ablation_table.json` (0.708) · `G5_external.json` (CI, H6 anchor) · `G5_calibration.json` (ECE) |
| Unimodal MRI | 3D-ResNet-18 + probe | 0.518 [0.462, 0.575] | — | — | null | `G3…/ablation_table.json` |
| Flat fusion | concat cross-attn | 0.636 [0.580, 0.692] | 0.253 | 0.503 | null — descriptive Δ vs 0.708 = −0.072; paired Δ vs **estimator-matched** clinical (~0.638) **−0.0026, p=0.98** (CI incl. +0.03 → not demonstrated) | `G3…/delong_deltas.json` |
| Hierarchical fusion #4 | staged late-clinical | 0.599 (per-seed mean); pooled-OOF ensemble 0.6365 [0.582, 0.691]† | — | 0.494 | null — descriptive Δ vs clinical −0.109; **paired DeLong Δ vs 0.708 anchor −0.130** (Stouffer p≪0.001; aligned N=613); still < 0.708 | `G3…/hierarchical_oof.json` · `G3…/paired_vs_anchor_delong.json` |
| Biology-gated MoE #7 | grade-band routing | **0.650 ± 0.018 [0.615, 0.689]** (20-salt sweep; all 20/20 < 0.708, closest 0.689; md5-deterministic instance 0.6542; pooled-OOF ensemble 0.6682 [0.613, 0.723]†) | — | — | null (< clinical); **paired DeLong Δ vs 0.708 anchor −0.075** (Stouffer p≪0.001); expert class-purity e0 0.876±0.015 / e1 0.717±0.012 (routing purity, NOT a per-expert AUROC) | `G3…/moe7_corrected_reporting.json` · `G3…/moe_salt_sweep/` · `G3…/paired_vs_anchor_delong.json` |
| Imaging closing arm (G2) | 3D-ResNet-18 corrected | 0.491 [0.395, 0.587] | — | 0.490 | null (6-axis independent) | `G2…/run_r18_mn_corrected_smoke` |

> **Power / MDE (paired fusion-vs-clinical-anchor leg, N=613):** the pre-registered +0.03 ΔAUROC margin
> is NOT detectable at N=613 — minimum detectable effect at 80% power ≈ 0.066–0.073 (≈2× the margin;
> ~2,900–3,600 patients needed to detect +0.03). Every fusion rung is *not demonstrated*, not *rejected*.
> Source: `G3…/mde_power.json`.

## Track A — quasi-external, calibration, explainability (G5)

| Result | Estimator | Value [CI95] | Notes | Verdict | Source |
|---|---|---|---|---|---|
| ISPY2 quasi-external (9-feat, imputed) | LogReg | AUROC 0.5725 [0.531, 0.614] | internal 0.719; honest drop Δ=0.147; shuffle 0.494 | real-but-weak (CI LB > 0.50) | `G5_external/metrics.json` |
| ISPY2 quasi-external (4 ISPY2-native feat) | LogReg | AUROC 0.538 [0.496, 0.580] | isolates grade-imputation confound: internal 0.5916; 4-feat internal→external drop 0.054 vs 9-feat 0.147; shuffle 0.492 | real-but-weak; confound shrinks the external gap (some external signal is honest) | `G5_external/ispy2_4feature_matched.json` |
| Calibration — internal | LogReg | ECE 0.0196 raw → 0.0244 temp (T=1.095) → 0.0376 isotonic (held-out) | best method = **none** (raw); temp & held-out isotonic both worsen it; meets ≤0.05 "good" raw | pass — no scaling needed | `G5_calibration/calibration_compare.json` |
| Calibration — external | LogReg | ECE 0.391 → 0.373 | prevalence-shift driven, not leakage | target not met externally | `G5_calibration/metrics.json` |
| XAI IoU (null encoder) | Grad-CAM/HiResCAM | 0.123 ± 0.035 (3-seed) | < 0.30 gate on all seeds; encoder AUROC 0.5008 | honest-null corroboration | `G5_xai/metrics.json` |
| XAI pointing game | Grad-CAM/HiResCAM | 0.635 | < 0.70 gate; randomization sanity PASS ×3 | honest-null corroboration | `G5_xai/metrics.json` |

## Track B — H&E histology subtype characterisation (TCGA-BRCA, Luminal A vs basal-like, N=640) — reported alongside, firewalled

| Result | Estimator | AUROC [DeLong CI95] | ECE | Shuffle | Verdict | Source |
|---|---|---|---|---|---|---|
| **arm-3 histology (co-headline)** | frozen TITAN + LogReg | **0.9646 [0.943, 0.986]** | 0.042 | 0.503 | RATIFIED co-headline (ADR-0015); LOSO 0.9679 | `arm3…/metrics_20260728.json` |
| Track B MIL (confirmation) | frozen UNI2-h + ABMIL | 0.9675 [0.9479, 0.9871] (3-seed 0.9622 ± 0.0038) | 0.0428 | 0.4309 | RATIFIED (ADR-0012) — reported alongside arm-3; **same 640-patient cohort, different encoder (UNI2-h vs TITAN) → encoder-robustness, NOT independent corroboration**; never juxtaposed with Duke | `trackb/mil_cv_uni.json` |

## Companions and pilots (distinct cohorts; no cross-cohort claim)

| Result | Cohort (N) | Value [CI95] | Shuffle | Verdict | Source |
|---|---|---|---|---|---|
| pCR Phase D — radiomics floor | ISPY2/MAMA-MIA (980) | AUROC 0.599 [0.561, 0.637] | 0.498 | NO-GO (gate 0.65) | `pcr_pilot/metrics_floor.json` |
| pCR Phase E — 3D-CNN | ISPY2/MAMA-MIA (980) | AUROC 0.4874 (3-seed; per-seed CIs cross 0.50) | 0.502 | genuine null, NO-GO (0.62) | `pcr_pilot/metrics_cnn.json` · `…_cnn_shuffle.json` (shuffle) |
| At-diagnosis recurrence stratification (ADR-0006) | Duke (920) | AUROC 0.577 [0.482, 0.614] | 0.468 | disclosed near-null; ECE 0.256→0.007 | `decisions.md:995` |
| Track-C ensemble panel | Coimbra 116 / BCSC 2.39M / METABRIC 1917 | 0.806 [0.724, 0.887] / 0.634 [0.625, 0.642] / 0.744 [0.717, 0.771] | — | independent per-cohort DeLong CIs (BCSC = strata-cluster bootstrap); ensemble, not fusion | `trackc/trackc_cis.json` |
| fastMRI-NYU H-char (standalone) | NYU (249) | AUROC 0.599 [0.430, 0.768] | 0.370 | NO-GO; NYU-only, never vs Duke | `decisions.md:1047` |

**Not citable until logged (LAW L-1):** Track B MIL 0.9675 (RATIFIED, ADR-0012, 2026-08-12) and the
Wave-1/2 arm set — the MIL number is now logged; the Wave-1/2 arms remain in `JOURNAL.md`/memory
and must be logged before citing. arm-5/arm-7 have no reportable number.
The numerical coincidence of fastMRI-NYU H-char (0.599) with the Duke hierarchical-#4 figure (0.599) is pure
coincidence across distinct cohorts/tasks — co-locate only as a firewall reminder, never as a comparison.
