# ADR-0006: Admit an at-diagnosis recurrence-stratification organ (proposes narrowing LOCK-1)
Date: 2026-07-13   Status: accepted (amends **LOCK-1** additively; ratified by Richard 2026-07-13 — see Ratification block)

> **Ratified 2026-07-13 by Richard's explicit instruction this session, with the near-null result (AUROC 0.577, CI [0.482, 0.614]) and the #1-integrity-exposure risk recorded and accepted.**

> **This ADR proposes a change to a LOCKED invariant (LOCK-1).** Per **LAW L-1** (single source of
> truth + change control), a locked item is reopened only via a dated `decisions.md` entry with
> reason · downstream impact · approver. This file is the case-law draft that *requests* that entry;
> it does **not** itself move the LOCK. Until Richard ratifies, LOCK-1 stands unchanged and the
> recurrence work stays OFF-LEDGER exploration (`explore/tabular_duke/`).

## Context

Two things are true at once, and this ADR exists to reconcile them honestly rather than paper over
either:

1. **A "Duke-native organ" is buildable and methods-validated.** The tabular risk pipeline in
   `explore/tabular_risk/` — LightGBM + a leakage-free nested-CV sigmoid recalibration, with
   DeLong-slot bootstrap CIs, PR-AUC, multi-seed spread, and a permuted-label negative control — was
   validated on the public Coimbra / BCSC / METABRIC trio (M1 Coimbra verified). Porting that exact,
   reused-verbatim machinery to Duke's own cohort (`explore/tabular_duke/`, EVL-verified) yields an
   at-diagnosis clinical → recurrence stratifier with no new data acquisition and no new modelling
   surface.

2. **LOCK-1 currently FORBIDS the framing this organ lives under.** LOCK-1's FORBIDDEN list includes
   "growth-rate / doubling-time" and disease-course / kinetics framing, and recurrence prediction is
   naturally read as disease-course. So an at-diagnosis recurrence stratifier cannot enter the
   architecture *as currently worded*. LOCK-1 (verbatim, `decisions.md` Invariant Register):

   > **LOCK-1 — Claim & reporting discipline.** ALLOWED framings (subtype characterisation; Ki-67
   > proliferation/aggressiveness AT diagnosis; explainable fusion) vs FORBIDDEN (early/pre-detection,
   > growth-rate/doubling-time, clinical-trial-grade FP/FN, cross-institution generalisation); always
   > report the continuous value + calibration, never a bare threshold/number.

This ADR proposes to **narrow** LOCK-1 by ONE carve-out: permit an **at-diagnosis recurrence
*stratification*** component as an architectural organ, under a strict framing guard (below). It does
NOT touch any other part of LOCK-1 — early/pre-detection, growth-rate/kinetics, clinical-trial-grade
FP/FN, and cross-institution generalisation all stay FORBIDDEN.

**The binding constraint is the number, and the number is weak.** Richard is being asked to amend a
LOCK with the result already known — that is the responsible order, and it is what makes this ADR
defensible to an OPSI judge. The result is recorded prominently and un-spun in the next section.

## The empirical result (recorded honestly, up front)

Off-ledger exploration `explore/tabular_duke/` (EVL-verified; `results/duke_recurrence_metrics.json`),
at-diagnosis clinical → binary recurrence, Duke cohort:

| metric | value |
|---|---|
| N total / in-scope / events / dropped | 922 / 920 / **87 events** (9.5% positive) / 2 (missing target) |
| **AUROC** (multi-seed mean, seeds 42/1337/2024) | **0.577** ± 0.023 (min 0.550, max 0.607) |
| **AUROC 95% CI** (seed 42, DeLong-slot bootstrap) | **[0.482, 0.614]** — **WIDE, and it CROSSES 0.50** |
| PR-AUC (seed 42) | 0.133 (vs 0.095 prevalence baseline) |
| ECE raw → sigmoid (leakage-free nested-CV) | 0.256 → 0.007 (bootstrap CIs separate; AUROC move 0.009 ≤ 0.05 ✓) |
| negative control (shuffled labels) | 0.468 ≈ 0.50 ✓ (no wiring leak) |
| leakage guard | 9 at-diagnosis features **disjoint** from 33 forbidden/treatment/outcome cols (asserted in code) |

**State it plainly:** this is a **WEAK, near-chance** signal. The DeLong CI **touches 0.50**, so at
the primary seed the result is not statistically distinguishable from a coin flip; the multi-seed mean
(0.577) sits just above chance because two of three seeds land higher, but the honest headline is "a
barely-above-chance recurrence signal in this cohort." The organ is proposed for admission **as an
honest, low-signal stratification component — explicitly NOT as a strong or clinical predictor.** The
one thing that IS solid is calibration: the leakage-free monotonic recal restores ECE (0.256 → 0.007)
without changing rank, and the negative control confirms the pipeline is leak-free. A well-calibrated
weak stratifier is still a legitimate architectural organ; a strong predictor it is not, and this ADR
does not claim otherwise.

## Decision (proposed)

**Amend LOCK-1 to permit an at-diagnosis recurrence-*stratification* organ**, subject to the framing
guard below. Concretely: add a narrow ALLOWED carve-out — "at-diagnosis recurrence *stratification* /
baseline-feature risk *characterisation*" — while leaving every existing FORBIDDEN framing in LOCK-1
intact. The organ is admitted as a **standalone, calibrated, low-signal clinical-tabular stratifier**,
reported always with its DeLong CI and the near-null caveat, Duke-cohort-only.

### LOCK-1 text amendment (diff-style, before → after)

The change is **additive to the ALLOWED list only** — not one word of the FORBIDDEN list is removed.
`decisions.md` Invariant Register row for **LOCK-1** (line 43) changes as follows:

```diff
- **LOCK-1** | Claim & reporting discipline | ALLOWED framings (subtype characterisation; Ki-67
-   proliferation/aggressiveness AT diagnosis; explainable fusion) vs FORBIDDEN (early/pre-detection,
-   growth-rate/doubling-time, clinical-trial-grade FP/FN, cross-institution generalisation); always
-   report the continuous value + calibration, never a bare threshold/number.
+ **LOCK-1** | Claim & reporting discipline | ALLOWED framings (subtype characterisation; Ki-67
+   proliferation/aggressiveness AT diagnosis; explainable fusion; **at-diagnosis recurrence-risk
+   *stratification* / baseline-feature risk *characterisation* — Duke-cohort-only, ADR-0006,
+   reported with DeLong CI + the near-null caveat, NEVER as a longitudinal/kinetics/predictive
+   claim**) vs FORBIDDEN (early/pre-detection, growth-rate/doubling-time, clinical-trial-grade
+   FP/FN, cross-institution generalisation); always report the continuous value + calibration,
+   never a bare threshold/number.
```

**What is unchanged (verbatim):** every FORBIDDEN entry — early/pre-detection, growth-rate/doubling-time,
clinical-trial-grade FP/FN, cross-institution generalisation — and the "continuous value + calibration,
never a bare number" reporting rule. The amendment ADDS one guarded ALLOWED clause; it SUBTRACTS
nothing.

## Framing guard (verbatim rules the organ's every mention must follow)

Every mention of this organ — in code, docs, figures, the report, and any presentation — MUST obey:

- **ALLOWED wording:** "at-diagnosis recurrence *stratification*", "baseline-feature risk
  *characterisation*".
- **FORBIDDEN wording:** "growth rate", "tumour kinetics", "doubling time", "progression", "early
  detection", and "predicts recurrence" *as a strong or clinical claim*.  <!-- # allow-ledger: names the bans in order to forbid them -->
- **Every reported number carries its DeLong CI** and the "CI crosses 0.50 / near-null" caveat — never
  a bare AUROC, never a bare threshold accuracy (this is also the LOCK-1 reporting rule).
- **Duke-cohort result only.** No cross-institution generalisation claim — that stays FORBIDDEN under
  LOCK-1 and is unaffected by this carve-out.

The guard exists because the FORBIDDEN framings are the *natural* way to describe recurrence work, so
they will drift back in constantly. Guard it actively, exactly as the Ki-67 head guards its
proliferation-vs-kinetics boundary ([1.2-R], ADR-0002).

## Architecture integration spec (how the organ attaches)

- **What it is:** a standalone **clinical-tabular LightGBM stratification head** — the exact
  `explore/tabular_duke/` pipeline (`src/duke_recurrence_loader.py` +
  `demo_duke_recurrence.py`), with `StratifiedKFold(5)` OOF and `scale_pos_weight` for the ~9.5%
  positive class.
- **Calibration:** a **leakage-free nested-CV sigmoid** recalibrator (`src/nested_calibration.py`) —
  within each outer fold the calibrator is fit only on inner-OOF training-fold probabilities and
  applied to the held-out fold; no point is scored by a calibrator that saw it. Reporting always pairs
  the calibrated probability with its reliability diagram + ECE.
- **Where it sits:** **alongside** the existing FT-Transformer clinical branch
  (`src/pinksight/models/clinical_encoder.py`). It reuses that branch's canonical at-diagnosis feature
  set — tumour size [T], nodes [N], Nottingham grade, age at diagnosis, menopause, race/ethnicity,
  multicentric/multifocal, metastatic-at-presentation, lymphadenopathy — so there is one feature
  contract, not two.
- **Recommended staging: standalone G2-companion FIRST, optional G3-fusion stream LATER.** Admit it
  as a standalone characterisation organ now (it needs no fusion to be reported honestly). Only
  consider wiring it in as an optional modality-dropout stream at G3 *after* the standalone number and
  its calibration are on record — and note the grade-double-count caution below before doing so.
- **Leakage rules (identical to the sandbox, non-negotiable):** the feature set is asserted **disjoint**
  from three excluded families before any matrix is built —
  (a) the FORBIDDEN biomarker set (LOCK-2: ER/PR/HER2/Ki-67/Mol-Subtype/Oncotype/ESR1/PGR/ERBB2/MKI67),
  (b) ALL post-diagnosis treatment fields, and
  (c) ALL outcome-derived fields (days-to-recurrence, days-to-death, response, restaging, the target
  itself) — with a regex family guard catching renamed variants. 33 excluded columns total; the
  disjointness assertion is auditable, not a comment.
- **Eval standard (already satisfied by the sandbox):** patient-level splits (patient IDs asserted
  unique), DeLong CI + ECE + 3-seed spread + a permuted-label negative control. This organ meets the
  project's evaluation-integrity bar (LOCK-2) as built.

## Consequences / risks (honest)

- **Top integrity exposure, by explicit acknowledgement.** Both the local council and the red-team
  flagged recurrence/kinetics + cross-cohort framing as the project's #1 integrity exposure with OPSI
  judges. Admitting a recurrence organ walks *toward* that exposure on purpose. The mitigation is the
  framing guard + CI-always + Duke-only — the same discipline that keeps the Ki-67 head on the right
  side of the kinetics line. If the guard slips, the organ becomes the single most attackable claim in
  the submission.
- **The signal is near-null.** AUROC ~0.58 with a CI that includes 0.50 means a hostile judge can
  legitimately say "this is not distinguishable from chance at your primary seed." The honest defence
  is that it is admitted *as* a weak, well-calibrated stratifier and always reported as such — never
  as a predictor. If that framing is ever dropped, the claim is indefensible; that is the risk this
  guard bounds.
- **Grade-double-count caution if fused (G3).** Nottingham grade carries essentially all of the Duke
  clinical AUROC ([G2 leakage Tier-1], 2026-07-10) and is already a feature of the FT-Transformer
  clinical branch. Wiring the recurrence organ into fusion risks counting grade twice (cf.
  `[G2-CLIN-LEAK]`). Keep it standalone until this is designed out.
- **What could go wrong, and how the guard bounds it:** (i) drift into "predicts recurrence" phrasing
  → the framing guard's FORBIDDEN list + the CI-always rule catch it; (ii) a reader reads 0.577 as a
  headline → the mandatory "CI crosses 0.50 / near-null" caveat reframes it; (iii) someone reads it as
  cross-institution → Duke-only wording + LOCK-1's untouched cross-institution ban forbid it. None of
  these are eliminated — they are bounded by discipline, which is why the guard is verbatim and
  active.
- **What becomes easier:** an honest, calibrated, rigor-forward organ that fits the project's
  characterisation-of-the-information-ceiling narrative (the same posture as the imaging honest-null
  and the FVA standard) without acquiring any new data or building any new modelling machinery.

### What stays in force (unchanged by this amendment)

- **LOCK-2 (evaluation integrity) — UNCHANGED.** Patient-level splits only; the 9-feature at-diagnosis
  input set stays asserted **disjoint** from the FORBIDDEN biomarker set + all treatment fields + all
  outcome-derived fields (33 excluded cols, code-enforced). This ADR touches LOCK-1's *claim/reporting*
  clause only — it does NOT relax any leakage rule.
- **Eval gates required BEFORE any submission claim** (all already met by the sandbox, re-required on
  the on-ledger organ): (1) patient-level split, IDs asserted unique; (2) AUROC reported with DeLong
  95% CI; (3) ECE + reliability diagram via the leakage-free nested-CV recalibrator; (4) multi-seed
  spread ≥3 (5 target); (5) a permuted-label negative control that collapses to ~0.50. A submission
  mention that drops ANY of these five, or drops the "CI crosses 0.50 / near-null" caveat, is a
  framing-guard breach and is not shippable.
- **Spine files DEFERRED to ratification.** `decisions.md`, `JOURNAL.md`, `docs/architecture/pinksight_system_architecture.md`,
  and the claim ledger are NOT edited by this ADR. They wire in only via the Ratification block below,
  and only after Richard signs. Until then LOCK-1 stands as written and `explore/tabular_duke/` remains
  OFF-LEDGER with no LOCK moved.

## Ratification block

**Ratified-by: [x] Richard — date: 2026-07-13**  *(SIGNED — ADR is ACCEPTED; LOCK-1 amended additively in `decisions.md` this session)*

> Ratified 2026-07-13 by Richard's explicit instruction this session, with the near-null result (AUROC 0.577, CI [0.482, 0.614]) and the #1-integrity-exposure risk recorded and accepted.


**STATUS flips to `accepted` and LOCK-1 amends in `decisions.md` ONLY upon Richard's explicit
ratification.** On ratification, wire (in this order):

1. **`decisions.md`** — a dated changelog append (LAW L-1: reason · downstream impact · approver)
   referencing this ADR, narrowing LOCK-1's ALLOWED framings to include "at-diagnosis recurrence
   stratification / baseline-feature risk characterisation" and recording the framing guard verbatim.
2. **Architecture section** (`docs/architecture/pinksight_system_architecture.md` / decisions.md architecture) — add the standalone
   clinical-tabular recurrence-stratification organ alongside the FT-Transformer clinical branch, with
   the standalone-first / optional-G3-stream staging.
3. **`JOURNAL.md`** — a dated entry recording the amendment and the organ's admission.
4. **Move / cite the evidence** — promote the `explore/tabular_duke/` result from off-ledger
   exploration to the organ's methods evidence (metrics JSON + reliability PNG + leakage-guard
   provenance), updating the OFF-LEDGER banner to "ratified organ (ADR-0006)".

Until all four are done, this ADR is a PROPOSED draft only, LOCK-1 stands as written, and
`explore/tabular_duke/` remains OFF-LEDGER with no LOCK moved.

## Alternatives considered

1. **Keep LOCK-1 as-is; recurrence stays off-ledger exploration.** — pro: zero new integrity exposure;
   the project's #1 judge-risk stays fully closed. Con: forgoes a buildable, methods-validated,
   no-new-data organ that fits the honest-null / characterisation narrative. This is the safe default
   and the fallback if Richard declines the amendment.
2. **A calibrated-subtype organ instead.** — pro: fully inside LOCK-1's ALLOWED framings (subtype
   characterisation), no LOCK change needed. Con: the imaging→subtype signal is a committed honest-null
   (pooled-OOF ≤ 0.567, sealed 6-axis-independent, 2026-07-13); a subtype organ adds little beyond the
   already-characterised clinical 0.708 / imaging-null story. Does not require this ADR.
3. **A pCR (pathologic complete response) organ.** — pro: also derivable from the Duke table; a
   legitimate at-diagnosis-predicts-treatment-response question. Con: pCR is a post-treatment outcome
   entangled with the treatment fields the leakage guard forbids as inputs, so an honest at-diagnosis
   pCR stratifier is a separate, non-trivial leakage-design problem; deferred, not chosen here.

---

## Status update — 2026-07-13 (draft written; awaiting Richard)

Written as a PROPOSED decision record with the Duke number KNOWN and recorded un-spun (AUROC 0.577,
CI [0.482, 0.614] crossing 0.50, 87/920 events). No live spine file touched — `decisions.md`,
`JOURNAL.md`, `docs/architecture/pinksight_system_architecture.md`, and the claim ledger all await Richard's explicit ratification per the
Ratification block. The `explore/tabular_duke/` sandbox remains OFF-LEDGER; no LOCK has moved.

## Status update — 2026-07-13 (RATIFIED)

**Status flipped to `accepted`.** Richard ratified this ADR by explicit instruction this session,
with the near-null result (AUROC 0.577, CI [0.482, 0.614] crossing 0.50, 87/920 events) and the
#1-integrity-exposure risk (recurrence/kinetics framing, flagged by council + red-team) recorded
and accepted with eyes open. Wiring completed on ratification: (1) `decisions.md` — dated changelog
append additively amending LOCK-1's ALLOWED list (framing guard recorded verbatim; no FORBIDDEN
entry removed; LOCK-2 untouched); (2) `docs/architecture/pinksight_system_architecture.md` — the standalone clinical-tabular
recurrence-stratification organ added as a clinical-stream companion head (NOT fused into the
imaging encoder), standalone-first per this ADR; (3) `JOURNAL.md` — dated entry recording the
amendment + the organ's admission. Origin sandbox `explore/tabular_duke/` is now the organ's methods
evidence. The framing guard (this file, "Framing guard" section) is binding on every mention of the
organ.

---
*Citations resolve against `decisions.md` → Invariant & Reference Register (v1.1). Template: `docs/templates/ADR_TEMPLATE.md`. Evidence: `explore/tabular_duke/` (metrics/calibration/negative-control JSON + reliability PNG).*
