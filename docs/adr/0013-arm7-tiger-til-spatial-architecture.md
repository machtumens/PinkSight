# ADR-0013: Arm 7 — TIL spatial-architecture characterisation from TIGER (outcome labels hard-excluded)

Date: 2026-07-28   Status: **PROPOSED** (drafted in a CPU-only execute session; requires PI sign-off before any arm-7 GPU pass, and a SEPARATE Gate-2 authorisation for the spend itself)

> **This ADR records a NEW additive companion-organ arm; it does NOT propose changing any LOCKED invariant.**
> It fits the **existing** LOCK-1 ALLOWED framing ("characterisation", "localisation") with no amendment —
> see §Claim-ledger compatibility. Per **LAW L-1** the operative go/no-go is a dated `decisions.md` entry.
> Arm 7 is a standalone companion organ: it is **NOT** fused into the Duke imaging encoder and **no arm-7
> result moves any LOCK**.

## Context

Arm 7 characterises the **spatial architecture of tumour-infiltrating lymphocytes (TILs)** at diagnosis
using **TIGER** (Tumor InfiltratinG lymphocytes in breast cancER, public challenge dataset). TIGER ships
per-slide sTIL (stromal TIL percentage) annotations, which makes a cheap CPU floor gate possible: the
scalar sTIL % is already computed and shipped, so the floor gate is a table read, not a segmentation run.

**The binding constraint is what else TIGER ships.** TIGER's challenge design includes a survival-analysis
leaderboard, and the distribution therefore carries **overall-survival and disease-free-survival outcome
columns**. Those columns are physically present in the same tables as the admissible features.

This is a materially different hazard from arms 3 and 5. For those arms the forbidden quantity is absent
from the data and the risk is purely one of wording. Here the forbidden quantity is **sitting in the
dataframe**, one careless `df.drop` omission away from becoming a target or, worse, a feature. A wording
guard alone is insufficient: a model could be trained on a survival column and reported in immaculate
prose. The guard has to bite at the **data layer**, before any model sees the matrix.

Outcome prediction is forbidden here for the same reason it is forbidden across PinkSight: the project
characterises tumours **at diagnosis**. A survival model makes a claim about the future of a patient, which
this cross-sectional design cannot support and which the claim ledger excludes.

## Options considered

1. **Use TIGER including its survival endpoints** — pro: richest use of the distribution; a survival number
   is the kind of result reviewers reward. con: **LOCK-1 FORBIDDEN** (prognosis / outcome prediction).
   Rejected outright.
2. **Use TIGER for sTIL spatial characterisation, with outcome columns dropped by convention** — pro:
   simple; the analyst "just doesn't use them". con: convention is not enforcement. One inherited notebook
   or one wildcard column selector reintroduces the leak silently. Rejected as insufficient.
3. **Use TIGER for sTIL spatial characterisation, with outcome columns excluded by a programmatic
   pre-flight guard that must exit 0 before any model fit** — pro: the exclusion is mechanical, auditable,
   and fails loudly; ledger-clean. con: a little extra plumbing. **Chosen.**
4. **Skip arm 7** — pro: zero risk. con: forfeits the only Wave-3 arm addressing immune spatial
   architecture, when the hazard is fully mitigable by option 3. Rejected.

## Decision

Arm 7 characterises **immune spatial architecture at diagnosis** from TIGER: a scalar sTIL-percentage CPU
floor gate first, and only on GREENLIGHT (and only after Gate 2) a GPU cell-segmentation pass that adds
genuinely spatial features (clustering, dispersion, tumour-stroma interface organisation).

**TIGER's OS/DFS outcome labels are hard-excluded by a mandatory pre-flight guard.** See §Forbidden target.

## Forbidden target

**Survival and outcome endpoints are FORBIDDEN as arm-7 targets, auxiliary targets, or input features.**
Named and rejected: overall survival, disease-free survival, progression-free survival, recurrence-free
survival, event/censoring indicators, vital status, and any time-to-event column.

**Enforcement is mechanical, not editorial.** `scripts/novel_heads/arm7_til_spatial_architecture.py`
exposes `--check-no-survival-labels`, which loads the TIGER feature matrix and calls
`wave1_eval_harness.assert_no_survival_columns(df)`. It scans column names for
`survival, os_, dfs_, pfs_, rfs_, event, censored, deceased, died, death, months_to, days_to, outcome`
and raises `LedgerViolation` (exit 1) on any match.

**This guard runs as floor-gate step 0 — before any model fit.** A model fit that occurs before the guard
has exited 0 is a protocol violation regardless of the result it produces. The guard is covered by a
poisoned-fixture test: a matrix with an injected `os_months` column MUST raise.

## Out of scope

Explicitly out of scope for arm 7, now and without a NEW ADR:

- **Prognosis, survival, or any time-to-event modelling** — see §Forbidden target.
- **Treatment-response prediction.** Arm 7 uses no treatment labels and makes no response claim.
- **Kinetics of any kind.** No growth rate, no doubling time, no tumour kinetics. Arm 7 describes spatial
  organisation at one timepoint.
- **Early detection / screening / pre-detection framing.** Arm 7 operates on slides from diagnosed tumours.
- **Cross-institution generalisation.** Arm 7 is TIGER-internal. No Duke comparison, no Duke validation,
  no transfer claim.
- **Fusion into the Duke imaging encoder.** Standalone companion organ; attachment needs a NEW ADR plus a
  fresh red-team pass.
- **Any GPU cell-segmentation pass** until Gate 2 is granted by the PI in the literal required form.

## Consequences

**Easier.** The sTIL scalar is shipped with the dataset, so the floor gate is minutes of CPU rather than a
segmentation job — the expensive spatial question is only asked if the cheap scalar justifies it. The
outcome-column exclusion is machine-checkable and testable, so it survives contributor turnover and
context loss.

**Harder.** We forgo TIGER's survival leaderboard entirely, so arm-7 results are not comparable to the
challenge's headline task. The floor gate deliberately uses a single scalar and may therefore return an
honest null that a richer spatial feature set would not — the plan mitigates this by making the GPU spatial
pass conditional on the floor, not a replacement for it.

**Commits / forbids downstream.**

- `--check-no-survival-labels` MUST exit 0 before arm 7's first model fit. Non-negotiable, no exceptions.
- The wording guard `validate_arm_report_keywords(report_text, "arm7")` bans "survival", "prognosis",
  "overall survival", "disease-free survival", " os ", " dfs ", "growth rate" and requires "spatial
  architecture at diagnosis" or "immune spatial architecture". Arm-7 reports therefore state the exclusion
  **positively** ("outcome labels excluded from the feature matrix") rather than as a disclaimer using the
  banned words.
- No arm-7 GPU pass may run without this ADR ratified AND Gate 2 granted. Both, not either.

## Claim-ledger compatibility

LOCK-1 ALLOWED includes "characterisation" and "localisation". Immune spatial architecture at diagnosis is
a morphological characterisation of a slide at a single timepoint — no amendment required. LOCK-1 FORBIDDEN
includes outcome/prognosis framing, which is why the survival endpoints are excluded at the data layer
above rather than merely avoided in prose.

**No LOCK is moved by this ADR.** LOCK-1 through LOCK-6 are unchanged. No arm-7 result — GREENLIGHT, KILL,
or null — moves a gate target or changes a headline number.

## Required framing (verbatim, for any arm-7 artifact)

- SAY: "spatial architecture at diagnosis", "immune spatial architecture", "sTIL characterisation",
  "characterisation", "localisation", "outcome labels excluded from the feature matrix".
- NEVER SAY: "survival prediction", "prognosis", "overall survival", "disease-free survival", "growth
  rate", "kinetics", "early detection", "pre-detection", "cross-institution generalisation".

## Status / sign-off

- **PROPOSED 2026-07-28.** Drafted during a CPU-only Wave-3 execute session.
- **PI sign-off: PENDING.** No arm-7 GPU pass may proceed until a dated `decisions.md` entry records
  acceptance (LAW L-1). The CPU floor gate is permitted under the plan's Wave-3 CPU scope.
- **Gate 2 (GPU spend): NOT GRANTED.** Separate and additional to the sign-off above. Requires the PI to
  write the literal phrase "authorize Wave 3 GPU spend".
