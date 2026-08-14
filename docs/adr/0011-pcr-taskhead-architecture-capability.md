# ADR-0011: pCR task-head as a DEFERRED pluggable fusion-architecture capability (documented slot, no shared-cohort gradient, no cross-cohort claim)
Date: 2026-07-23   Status: accepted (deferred capability) — ratified after /red-team BLOCK→PASS on the 5 folded fixes

## Context

The pCR pilot (ADR-0007) is closed: a pre-registered ISPY2 (MAMA-MIA, N=980)
standalone pipeline. Radiomics floor AUROC **0.599** [DeLong 0.561–0.637] — NO-GO
on the ≥0.65 greenlight. Phase-E CNN **0.4874** (all 3 DeLong CIs cross 0.50) —
genuine null; shuffle sentinel **0.5019** [0.4634, 0.5404] — leakage-clean. The
result is a real-but-weak floor + a genuine imaging null, on a cohort **disjoint
from Duke** (Track A). Duke carries **zero pCR labels**; there is no patient with
both a Duke encoder input and a pCR target, so joint training is not definable.

The PI directed that the PinkSight fusion architecture should be able to host a
pCR head. This ADR admits that ONLY as a **deferred, documented architecture slot**
— hardened by a hostile /red-team pass (2026-07-23) that BLOCKED the first draft
for three leak paths (shared weights, honor-system synthetic fixtures, unfalsifiable
capability). All five required fixes are folded below. This ADR deliberately does
NOT amend LOCK-1.

## Options considered

1. **Reject** — refuse any pCR wiring. Rejected: PI has authority over the architecture.
2. **Fuse a pCR head into the Duke model and report a combined metric.** Rejected:
   impossible (no shared patients) and, if forced via synthetic pairing, a fabricated
   cross-institution result — a LOCK-1 and OPSI-integrity violation.
3. **Amend LOCK-1 to permit a cross-institution pCR claim.** Rejected: unjustified —
   the pCR arm is a genuine null; there is no result worth generalising.
4. **(CHOSEN, hardened) Deferred architecture capability.** Document a task-head slot;
   no weights attach until a matched cohort exists; no cross-cohort metric is ever
   reported; LOCK-1's cross-institution FORBIDDEN clause stays fully in force.

## Decision

Admit a **deferred pluggable pCR task-head slot** in the fusion architecture, bound by
five enforceable rules (the folded red-team fixes):

- **(Fix 1 — no cross-cohort gradient).** No trainable parameter may receive gradient
  from more than one cohort. Any future pCR branch attaches only to a **frozen** Duke
  trunk OR uses fully separate weights. Duke and ISPY2 samples never co-occur in a batch.
- **(Fix 5 — deferred).** No ISPY2 weights attach to any Duke-fed trunk until a
  **matched cohort** exists (same patients carrying both the encoder input and a pCR
  label). Until then the architecture shows a **greyed "task-head slot (untrained, no
  data)"** and nothing more. No pCR training code lands under this ADR.
- **(Fix 2 — poisoned synthetic fixtures).** Any synthetic/artificial array used in
  development writes its labels as `NaN`/poison sentinel so any AUROC/metric call
  **errors out**, not merely "should not be reported." A CI assertion rejects any file
  on a pCR-eval path unless it traces to a registered real-cohort manifest hash. These
  artifacts are named `plumbing_smoke_*`, **never** `val`. Their sole purpose is proving
  shapes match and gradients flow.
- **(Fix 3 — falsifiable, Duke-only).** The capability's ONLY admissible evaluation is a
  **Duke-only ablation**: attaching the (empty) slot must not degrade Duke subtype AUROC
  by >0.005. No ISPY2 pCR number may ever be produced to "demonstrate the capability."
- **(Fix 4 — co-location with the null).** Every mention of the pCR capability, in any
  artifact, must appear within one sentence of: "the pCR arm is a genuine null (0.487;
  ADR-0007)." Capability and null are quoted together or not at all.

The contribution claimed is **architectural design extensibility only** (methods note),
NOT performance, NOT generalisation.

## Consequences

- **Easier:** the architecture documents extensibility (a methods design note) without
  new data or new cohorts; consistent with ADR-0008 (architecture-forward) and ADR-0006
  (standalone-first).
- **Committed:** the CI synthetic-poison + manifest-hash gate must exist before any pCR
  fixture is created; the co-location-with-null rule binds every writeup mention.
- **Blocked downstream (needs a NEW ADR + /red-team):** attaching real weights; any
  matched-cohort pCR evaluation; any reported pCR metric from a fused/attached config;
  any "predicts pCR" or cross-institution transfer statement.

## What this ADR does NOT do (honesty firewall)

- Does **NOT** amend LOCK-1. "cross-institution generalisation" stays verbatim FORBIDDEN.
- Does **NOT** license any reported pCR number from a fused/cross-cohort configuration.
- Does **NOT** license synthetic val sets as evidence — poisoned, un-scoreable, un-reportable.
- Does **NOT** land pCR training code — the slot is documented and deferred, not trained.
- Does **NOT** claim the pilot was under-powered or "would work with more data." It is a
  genuine null (ADR-0007); that stands.
- Does **NOT** move any LOCK, gate verdict, or the ADR-0007 NO-GO.

## Framing guard

- ALLOWED: "the fusion architecture documents a pluggable task-head slot (deferred, no
  data); the pCR arm is a genuine null (0.487; ADR-0007)."
- FORBIDDEN: "imaging predicts pCR", "the model generalises across institutions", "pCR
  AUROC of the fused model is X", "Duke pCR", or any combined/cross-cohort metric.

## What stays in force (unchanged)

- LOCK-1 (claim discipline, incl. cross-institution FORBIDDEN) — unchanged.
- LOCK-2 (leakage/eval integrity, patient-level sealed splits) — unchanged.
- ADR-0007 pCR NO-GO (0.599 floor / 0.487 null) — unchanged.
- ADR-0008 fusion architecture-forward reframe — extended in posture, not in numbers.

## Ratification block

Proposed 2026-07-23. /red-team verdict: **BLOCK → PASS** conditional on 5 fixes — all
folded into the Decision above. PI ratified 2026-07-23 (session directive: "fold 5 fixes
and ratify for pinksight integration"). On acceptance: decisions.md appended, CLAUDE.md
architecture line updated with the deferred slot, memory captured. Reopening to attach
real weights requires a NEW ADR + fresh /red-team + a matched cohort.

Signed: Richard (PI), 2026-07-23   Red-team verdict: PASS (post-fix), 2026-07-23
