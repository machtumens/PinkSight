# Claim Ledger — PinkSight (Team TestEin, OPSI 2026)

> The scientific constitution for this submission: the allowed/forbidden claim boundary,
> the evaluation-integrity rules, the gate spine, and the target metrics. Extracted verbatim
> from the project's governing `CLAUDE.md` (the CLAIM LEDGER, Evaluation integrity, Gates, and
> Targets sections). Authority order: `decisions.md` > this ledger. Research posture — OPSI 2026
> competition, not a clinical product.

## CLAIM LEDGER — the constitution (NEVER violate; if asked to, STOP and flag)
**ALLOWED:** subtype characterisation · Ki-67 stratification AT DIAGNOSIS · explainable multimodal fusion.

**FORBIDDEN:**
- pre-detection in healthy tissue / "early detection"
- growth-rate / tumour kinetics / doubling time (Ki-67 is a snapshot, not kinetics)
- clinical-trial-grade false-positive / false-negative reduction
- cross-institution generalisation claims

**Wording:** say "characterisation / localisation", never "early detection". Say "Ki-67 / aggressiveness", never "growth rate". This forbidden framing is the *natural* way to describe the project, so it drifts back in constantly — guard it actively.
## Evaluation integrity (non-negotiable)
- Patient-level splits only; never image/slice-level.
- Quasi-external (scanner/year) holdout carved FIRST, before any tuning.
- Leakage assertion in CI: ER/PR/HER2/Ki-67/Mol-Subtype/Oncotype are EXCLUDED from classifier inputs.
- No Duke ↔ MAMA-MIA patient overlap (de-duplicate before any "external" claim).
- If mask-gated: train AND test on PREDICTED masks (never ground-truth masks at test).
- Report AUROC with DeLong CI + ECE (calibration) + multi-seed spread (3 min / 5 target) — never a bare number.
## Gates (the spine — each must produce its number)
`G0` audit/repo → `G1` replicate Duke baseline → `G2` single-modality MVP + Ki-67 head → **`G3` fusion + ablation + stats + XAI (committed floor)** → `G4` Track B → `G5` external + calibration + XAI-validation → `G6` clinician pretest + closeout.
## Targets (min / target / stretch)
subtype AUROC 0.75 / 0.80 / 0.85 · Ki-67 Pearson r 0.40 / 0.55 / 0.70 · MCC 0.40 / 0.55 / 0.70 · ECE ≤ 0.05 (good) / ≤ 0.10 (acceptable) · XAI IoU ≥ 0.30 (pointing ≥ 0.70) · ΔAUC fusion vs unimodal ≥ 0.03 (p<0.05) · Cohen's κ 0.61 / 0.71 / 0.81 · SUS 68 / 73 / 80.
