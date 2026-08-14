# ADR-0005: Head-2 target pivot — Ki-67 (data-blocked) → Nottingham grade (NHG1-vs-NHG3)
Date: 2026-07-11   Status: ACCEPTED — ratified by PI (Richard) 2026-07-11
Supersedes (in scope): ADR-0002 (Ki-67 head framing) for the predictive head design; ADR-0002 remains the authority on Ki-67 descriptive treatment.

## Context

ADR-0002 committed Head-2 to a **Ki-67 continuous regression** head (Huber loss, fixed 14% St. Gallen cutoff),
contingent on the G0 usable-N count clearing the O-2 threshold. The G0 audit (`scripts/audit_ki67.py`) found
**usable numeric Ki-67 N = 0** in the Duke cohort (Ki-67 appears in radiogenomics free text only, no structured
open field). The AMBL external cohort carries Ki-67 for **N = 36** patients — materially below the pre-registered
O-2 floor of 40. Ki-67 as a *predictive* Head-2 target is therefore **DATA-BLOCKED** on all available open DCE-MRI
datasets. ADR-0002's O-2 condition fired in the null direction.

Meanwhile, an independent published result emerged that re-opens an imaging head on the same cohort via a
different target: **Hadidchi et al. (2025)** (DeepRadGrade, *Eur Radiol*, PMID 41405689, PMC free full text
fetched 2026-07-10) trained a CNN on **Duke (n=877) + AMBL (n=37 external)**, radiologist-box crop,
multi-phase pre-op DCE (1 pre + 3–4 post), predicting **binary Nottingham grade NHG1-vs-NHG3 (NHG2 dropped)**,
and achieved **test AUC 0.82 [0.71–0.91] / train 0.84 / external 0.84 [0.69–0.96]**.

This is the *same cohort, same modality, same input style* where our LumA-vs-TNBC subtype task nulled
6-axis-independently (see `[G2-RECIPE-CORRECTED]` and H4 ceiling ≤ 0.624 DeLong UB). Grade ≠ subtype:
subtype is molecular receptor status (weak/absent imaging correlate at N~600); grade is a morphological/
proliferation phenotype (tumour architecture, nuclear pleomorphism, mitotic activity) with direct imaging
correlates — which is why the same cohort that nulls on subtype trains on grade (DeepRadGrade is external proof).
**This opens a new, verified head; it does not relitigate or reopen the spent 6-axis subtype null.**

Two corrections to the published claim, recorded (2026-07-10):
1. "Single-timepoint" in informal summaries is imprecise — the paper uses single-**visit** pre-op but
   **multi-phase** (1 pre + 3–4 post-contrast), which is exactly what Duke provides. Grade does not require
   during-treatment scans (that constraint bites pCR, not grade).
2. The grade ↔ Ki-67 pathological link rests on **pathology** (mitotic count is a Nottingham component),
   NOT on this paper's data. All Ki-67-substitute framing must be worded that way.

**How this respects the locked [1.16]/[8.3] caveat (no violation).** The locked caveat in decisions.md
reads: *"grade's mitotic component ∝ proliferation → do NOT reuse grade naively as an INPUT to Head-2."*
That constraint bites when grade is a **feature** feeding a proliferation **prediction** (circular input).
Here grade is the **target** and DCE-MRI is the input — grade is absent from the imaging, so there is no
circularity. The caveat is satisfied by construction.

**Our empirical evidence for grade (as of 2026-07-11, states honestly):**
- `[GRADE_CLINICAL]` — clinical→grade: **AUROC 0.687** [0.613, 0.761], shuffle 0.479, dev N=198
  (82 NHG1 / 116 NHG3, 3-way clinical∩radiomics∩grade∈{1,3} intersection). Grade is clinically
  predictable — this is the **confirmed floor** to beat.
- `[H6]` amendment — re-baselined clinical subtype anchor 0.634 → **0.719** (Richard-ratified);
  grade-ablation shows clinical-full 0.708 / grade-only 0.691 / minus-grade 0.590 (ΔAUROC +0.118,
  p≈0): **clinical is grade-anchored** (grade-only recovers ~97% of clinical-full signal for subtype).
- `[HEAD2-GRADE_SMOKE]` (S27, Arm A) — imaging→grade first smoke: OOF **0.522**, DeLong [0.392, 0.652],
  shuffle 0.424. LB spans 0.5 → pre-registered NULL. Fired the gate-closed rule; full 3-seed not paid.
- `[HEAD2-GRADE_RESMOKE]` (S27b) — after recovering 56/57 missing patients (trainable N 257→313) AND
  switching to DeepRadGrade-matched input (`fixed4` 4-ch + `crop_mode=lesion` tight tumour crop): OOF
  **0.5023**, DeLong [0.412, 0.593], shuffle 0.528. Pre/crop hypothesis FALSIFIED; null is stronger.
  Full 3-seed NOT run.
- `[SPEC_06]` — frozen DINOv2-B/L + BiomedCLIP linear probe on grade: secondary reads 0.494–0.585,
  shuffle floors 0.418–0.533. Faint encoder-scaling trend (0.494→0.585 as models grow), below the 0.60
  threshold and contaminated by low shuffle floor on BiomedCLIP. **Suggestive, not signal.**
- ComBat-radiomics anchor→grade: **~0.569** (near-null); clinical carries the grade signal, not radiomics.

**Bottom line on our own evidence:** clinical→grade WORKS (0.687, floor confirmed). Imaging→grade is
**UNPROVEN-TO-NULL on our pipeline** at two independent smoke points. We have NOT executed a full, careful
DeepRadGrade-method replication (full N=313, 3-seed, SPEC_01 lineage). The 0.82 is neither confirmed nor
refuted by our pipeline — it is the external precedent driving the bet.

## Options considered

1. **Head-2 = imaging→grade replication (DeepRadGrade method, GPU required)** — pro: the headline
   imaging bet; directly replicates the external precedent on our cohort; if it works, the first live
   imaging prediction in the project. Con: two smoke probes at the correct N+crop returned NULL; N=313 vs
   their 431 (~26% less); replication is plausible, not guaranteed. Compute ≤ $25 incremental (LOCK-5).
2. **Head-2 = clinical→grade only (0.687 floor, already in hand)** — pro: real signal, no GPU, immediately
   usable; con: not imaging, so the "multimodal imaging head" contribution is weaker; clinical is already
   grade-anchored (ADR double-count risk in the narrative).
3. **Head-2 = descriptive only (AMBL Ki-67 + clinical grade-floor)** — pro: safest, no null risk; con:
   abandons the sole credible imaging precedent on our cohort, weakening the submission versus the
   honest-null narrative.
4. **Head-2 = Ki-67 predictive (original ADR-0002 plan)** — rejected: DATA-BLOCKED (N=0 Duke open,
   N=36 AMBL < O-2 floor). Infeasible as a trainable predictive head. Closed.

## Decision

**Head-2 target pivots to binary Nottingham grade (NHG1-vs-NHG3).** The execution plan is:

1. **Floor established (done):** clinical→grade baseline **AUROC 0.687** [0.613, 0.761] — the number
   any imaging replication must beat to claim added imaging value.
2. **Imaging→grade replication bet (the headline):** run a proper **DeepRadGrade-method replication**
   (GPU; SPEC_01 lineage; ADR-0001 backbone 3D-ResNet-18 + MedicalNet + corrected optimizer; full N=313;
   3-seed; `fixed4` 4-ch + `crop_mode=lesion`; binary NHG1-vs-NHG3 NHG2-dropped; patient-level
   StratifiedGroupKFold; DeLong + shuffle + ECE + 3-seed spread). Gated on a cheap-first smoke that
   clears the GREENLIGHT rule before any GPU spend.
3. **clinical+imaging fusion for grade (conditional):** attempt only if step 2 imaging result LB > 0.5.
   Disclose the grade-double-count dependency explicitly (see Integrity section).

Ki-67 moves to **descriptive companion only** (AMBL N=36, `[KI67-DESCRIPTIVE-AMBL]`). It is not a
predictive head. The pathology link (mitotic count ∈ Nottingham grade) is the framing bridge between
Ki-67 and grade in the narrative — never "we predict Ki-67."

## Claim scope (mandatory — this ADR records a claim-ledger change)

**ALLOWED wording (use exactly these forms):**
- "Head-2 characterises Nottingham grade (binary NHG1-vs-NHG3) at diagnosis."
- "Head-2 characterises the proliferation/aggressiveness axis at diagnosis via Nottingham grade."
- "Ki-67 (AMBL, N=36) is reported descriptively as a companion via the grade ↔ proliferation pathology link."

**FORBIDDEN wording (must never appear in any output or submission artefact):**
- "We predict Ki-67" — Head-2 is a grade head; Ki-67 is a descriptive companion.
- "Growth rate", "tumour kinetics", "doubling time" — grade is a cross-sectional at-diagnosis
  snapshot; it characterises proliferation status, not a temporal rate.
- "Cross-institution generalisation" — Duke is the primary cohort; AMBL (N=37) is an external probe,
  not a generalisation claim. Do not pool Duke + AMBL for a grade generalisation headline.
- "Early detection" or "screening" — Head-2 is a characterisation of an already-diagnosed tumour.
  CLAIM-LEDGER FORBIDDEN.

**This is a claim-scope change to a proposal-core deliverable.** The OPSI 2026 proposal promises
Ki-67 proliferation stratification (Row 3). Head-2 = Nottingham grade is a defensible scientific
substitution (Nottingham grade is the clinical-standard at-diagnosis aggressiveness axis, and its
mitotic component is the pathology-level proxy for Ki-67), but it is **not Ki-67**. The PI must
sign this scope change consciously. Ratifying this ADR constitutes that sign-off.

## Integrity caveats

**Grade-double-count in fusion (NEW flag, E6).** The clinical subtype head is grade-anchored:
grade-only 0.691 ≈ clinical-full 0.708 for LumA-vs-TNBC. If a future G3 fusion combines the
clinical stream (grade as input to the subtype head) with THIS imaging-grade Head-2, grade appears
on both sides of the fusion. Do not present that configuration as independent multimodal signal.
Options: (a) drop grade from the clinical subtype features when fusing; (b) disclose the shared-grade
dependence explicitly in the write-up. This is not a blocker but must be stated every time fusion
results for grade are discussed.

**Patient-level splits only.** No image-level or slice-level AUROC is a substitute for the
patient-level pooled-OOF metric. Every reported AUROC must be patient-disjoint.

**Predicted masks only at test.** If H0 (MAMA-MIA nnU-Net) masks are used at any stage, the same
mask source (predicted, never ground-truth) is used at train AND test. No GT-mask leakage at test.

**The inner-val-peak trap.** Multiple smoking sessions produced inner-val peaks 0.71–0.775 that
collapsed to OOF chance. Judge on **pooled OOF only** (DeLong + shuffle + 3-seed spread), never on
per-fold inner-val curves. A result is real only when the OOF LB clears 0.5.

**N honest anchor.** Our trainable contrast is **NHG1=113 + NHG3=207 = 320** (universe); preprocessed
N=313 (7 phase-stack-unrecoverable). DeepRadGrade trained on NHG1=162 + NHG3=269 = 431 (~26% more).
Report our N honestly in every table and write-up. The external 0.84 rests on only 37 AMBL patients —
fragile; the trustworthy external precedent is **test 0.82 [0.71–0.91]**, not the external 0.84.

## Pre-registered kill-gate (judge on pooled OOF only)

These thresholds are fixed before any GPU replication run and are non-negotiable post-hoc:

| Outcome | OOF AUROC criterion | Consequence |
|---|---|---|
| **GREENLIGHT (smoke gate)** | Smoke grade OOF clearly off chance (DeLong LB > 0.5, folds hold) | Proceed to full 3-seed run (GPU, ≤$25 incremental) |
| **SMOKE-NULL** | OOF AUROC LB spans 0.5 (DeLong lower bound ≤ 0.5) | Do NOT launch full run; Head-2 stays descriptive |
| **HEAD-2 ALIVE (imaging success)** | Full-run pooled-OOF AUROC ≥ 0.70 with DeLong LB > 0.5, shuffle ≤ 0.53, 3-seed spread < 0.05 | Imaging-grade head is a live result; target ~0.82 [0.71–0.91] band |
| **CONDITIONAL** | Full-run pooled-OOF 0.60 ≤ AUROC < 0.70, LB > 0.5, shuffle-clean | Usable but below target; report honestly, cite clinical floor (0.687) as context |
| **IMAGING-NULL** | Full-run pooled-OOF AUROC < 0.60 OR DeLong LB ≤ 0.5 after shuffle-clean check | Declare imaging→grade NULL at our N; Head-2 = clinical→grade floor (0.687) only; no further GPU spend on imaging-grade without a new pre-registered ADR |

Kill-gate summary: proper replication **pooled-OOF < 0.60 with LB ≤ 0.5 and shuffle-clean → IMAGING-NULL**;
**≥ 0.70 → headline achieved**; **0.60–0.69 → CONDITIONAL**. A shuffle-contaminated result (OOF ≈ shuffle)
is always NULL regardless of AUROC absolute value.

## Consequences

- **Architecture:** Head-2 loss retargets to binary cross-entropy (NHG1=0, NHG3=1, NHG2 dropped from
  training). The Huber-regression head from ADR-0002 is retired for the predictive stream.
- **XAI:** Head-2 Grad-CAM + SHAP targets grade. IoU validation uses the same tumour-box annotations
  as the imaging input, assessing morphological agreement with grade-defining features.
- **G3 fusion:** the grade head joins the G3 fusion ablation. The grade-double-count flag (see Integrity)
  must be disclosed in the G3 methods section.
- **Ki-67:** retained as a descriptive association only — AMBL N=36, framed via the mitotic-count pathology
  link. Never a model output; never a predictive headline.
- **LOCK-5 ($150 cap, ~$142 remaining):** cheap-first smoke gates every GPU spend. Hard stop ≤ $25
  incremental for this head. Full replication ~$3–6 RunPod / free Kaggle.
- **ADR-0002 status:** ADR-0002 closes its O-2 open-lock in the null direction. The descriptive Ki-67
  treatment (14% St. Gallen, AMBL companion) in ADR-0002 remains authoritative for the companion analysis.
  This ADR governs the *predictive head* design only.

---

## Ratification block — PI sign-off required

> Ratifying this ADR constitutes conscious sign-off on the claim-scope change (Ki-67 predictive → Nottingham
> grade predictive). On ratification, append the `[HEAD2-GRADE-PIVOT]` Changelog block from
> `reports/G2_imaging/HEAD2-NOTTINGHAM-GRADE_PREREG_10-07-26.md` verbatim to `decisions.md`. Do not
> modify decisions.md before ratification.

| Field | Value |
|---|---|
| Decision | Head-2 target pivots from Ki-67 (data-blocked) to binary Nottingham grade (NHG1-vs-NHG3). Ki-67 becomes descriptive companion (AMBL N=36). Clinical floor 0.687. Imaging→grade replication of DeepRadGrade (test AUC 0.82) is the headline bet, gated by the pre-registered kill-gate above. |
| Ratified-by | Richard (PI) |
| Date | 2026-07-11 |
| Signature | Ratified in-session (chat) 2026-07-11 |

After signing: append `[HEAD2-GRADE-PIVOT] LOCKED` to `decisions.md` (dated entry), referencing this ADR
number (ADR-0005). The staging file `reports/G2_imaging/HEAD2-NOTTINGHAM-GRADE_PREREG_10-07-26.md` and its
Changelog block are the verbatim source for the decisions.md entry.

---

*Citations resolve against `decisions.md` → Invariant & Reference Register. Template: `docs/templates/ADR_TEMPLATE.md`.
Staging file: `reports/G2_imaging/HEAD2-NOTTINGHAM-GRADE_PREREG_10-07-26.md`. Session records: JOURNAL.md S26–S30.*
