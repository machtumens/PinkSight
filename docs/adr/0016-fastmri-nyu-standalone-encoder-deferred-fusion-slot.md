# ADR-0016: fastMRI-NYU as a STANDALONE encoder + a DEFERRED Duke-only-ablation fusion slot
(separate weights, no cross-institution gradient, no pooled NYU+Duke claim)

Date: 2026-08-04   Status: **ACCEPTED** (deferred capability) — /red-team CONDITIONAL PASS + PI ratification recorded
Supersedes: none. Related: ADR-0011 (pCR deferred task-head), ADR-0012 (Track-B deferred), ADR-0008 (G3 imaging-fusion null), ADR-0001 (imaging encoder / O-1).

## Context

fastMRI Breast (NYU, 300 patients: neg 51 / malig 90 / benign 159; official shipped split 240/60) is
on disk (acquired 2026-07-13; schema/split/counts verified 2026-08-04). The PI directs training a
fastMRI encoder and, later, fusing it into the PinkSight model. PinkSight's headline is Duke (Track A).
fastMRI is a DIFFERENT institution — so, unlike the pCR case (ADR-0011, where Duke carries zero pCR
labels and joint training is undefinable), here BOTH cohorts carry DCE-MRI and the temptation to share
a gradient is real. That makes an explicit firewall necessary, not incidental. This ADR admits the work
ONLY as (1) a standalone NYU organ and (2) a deferred, Duke-only-evaluated frozen feature slot. It
deliberately does NOT amend LOCK-1.

## Options considered

1. Reject any fastMRI wiring. Rejected: PI has authority over the architecture; the standalone organ is
   fully ledger-safe.
2. Train one shared encoder on NYU+Duke and report a combined metric. Rejected: cross-institution
   generalisation — a LOCK-1 and OPSI-integrity violation.
3. Amend LOCK-1 to permit a NYU→Duke transfer claim. Rejected: unjustified; no result worth
   generalising, and it would relax a constitutional commitment.
4. (CHOSEN) Standalone NYU encoder + DEFERRED frozen-feature slot, evaluated Duke-only. No shared
   gradient; no pooled/juxtaposed NYU+Duke number; LOCK-1 stays fully in force.

## Decision

Admit (a) a STANDALONE fastMRI-NYU encoder and (b) a DEFERRED pluggable fusion slot, bound by
enforceable rules (mirroring the ADR-0011 red-team fixes, hardened by the 2026-08-04 red-team):

- **(No cross-cohort gradient.)** No trainable parameter may receive gradient from more than one cohort.
  The NYU encoder trains on NYU only. Any future Duke attachment uses the NYU encoder FROZEN (or fully
  separate weights). NYU and Duke samples never co-occur in a batch.
- **(Deferred fusion.)** No fusion code lands under this ADR. The trained NYU encoder is frozen; the Duke
  attachment is documented as a slot, not run. Running the attachment ablation is a separate go-ahead;
  unfreezing NYU weights into the Duke trunk needs a NEW ADR + fresh /red-team.
- **(Duke-only, falsifiable eval + OUTCOME-GATE.)** The slot's ONLY admissible evaluation is a Duke-only
  ablation on Duke's own subtype task and sealed split — report Δ(Duke subtype AUROC). **ONLY a
  null/bounded result (|Δ| ≤ 0.005, "no separable effect") is reportable under this ADR.** A positive
  frozen-transfer Δ beyond the ±0.005 null band is a cross-institution transfer finding — it is
  NON-REPORTABLE and requires a NEW ADR + fresh /red-team before it enters any artifact. No NYU test
  number may ever be reported as validating Duke; no pooled/juxtaposed NYU+Duke metric is ever produced.
- **(H6 double-gate — HARD interlock.)** The anomaly head (which includes the 51 verified-normal scans)
  is NOT trained under this ADR. The 51 verified-normal patient IDs are quarantined behind a manifest
  that the H-char/H5 loader CI-asserts is absent from every batch (mechanical, not honor-system). H6 is
  admissible only as characterisation within an already-referred diagnostic population, and requires its
  OWN additional /red-team on the early-detection surface before the manifest unlocks or any H6 code lands.
- **(NYU-internal reporting.)** Every fastMRI number is reported NYU-internal, with AUROC + DeLong CI +
  ECE + shuffle sentinel + ≥3 seeds; the H-char go/no-go binds to the DeLong lower bound, not the point
  estimate (n=50 sealed test / 18 positive → wide CI). No NYU metric is placed in comparison (same
  table/sentence, as if measuring the same thing) with a Duke number; co-locating the ADR-0008 null as a
  firewall reminder is explicitly permitted.

The contribution claimed is a standalone NYU characterisation organ + architectural extensibility
(methods note), NOT cross-institution performance, NOT generalisation.

## Scope admitted (the three heads)

- **H-char** malignant-vs-benign characterisation (malig 90 vs benign 159; drop the 51 normals) —
  ALLOWED, primary head, trainable now behind the Unlock Gate.
- **H5** chronological age regression (Huber), forward-register, NYU-internal MAE only — ALLOWED,
  secondary head, trainable now. NOT a biological-age/physiological-progression proxy.
- **H6** anomaly/novelty (incl. the 51 verified-normal) — DEFERRED, documented design only, HARD-
  interlocked (see Decision), needs its own early-detection /red-team before any code.

## Consequences

- Easier: a standalone NYU organ (characterisation + a forward-register age head) and a documented
  extensibility slot, consistent with ADR-0006 (standalone-first) and ADR-0008 (architecture-forward).
- Committed: the NYU-only training discipline, the frozen-encoder attachment, the Duke-only ablation
  eval, the ±0.005 outcome-gate, and the H6 hard manifest interlock are binding on any code that lands.
- Blocked downstream (NEW ADR + /red-team): unfreezing NYU weights into Duke; any NYU→Duke fine-tune;
  any slot Δ beyond the ±0.005 null band; any pooled/juxtaposed NYU+Duke metric; any "generalises across
  institutions" statement; training H6.

## What this ADR does NOT do (honesty firewall)

- Does NOT amend LOCK-1. "cross-institution generalisation" stays verbatim FORBIDDEN.
- Does NOT license any pooled or juxtaposed NYU+Duke metric, or any "NYU→Duke transfer" claim.
- Does NOT license the H6 anomaly head as early detection / screening / "cancer in a normal breast."
- Does NOT read H5 age as a biological-age / physiological-progression signal — it is chronological-age
  regression, forward-register, NYU-internal MAE only.
- Does NOT land fusion code — the slot is documented and deferred (frozen-transfer, Duke-only-eval).
- Does NOT claim fastMRI improves, rescues, or reopens the Duke imaging-fusion ceiling (ADR-0008 null —
  clinical 0.708; hierarchical #4 0.599 [0.495, 0.610] — stands).
- Does NOT move any LOCK, gate verdict, or prior ADR.

## Framing guard

- ALLOWED: "a standalone fastMRI-NYU encoder characterises malignant-vs-benign at diagnosis (NYU-internal
  AUROC X [CI]); the architecture documents a deferred, Duke-only-evaluated frozen feature slot."
- FORBIDDEN: "the model generalises from NYU to Duke", "NYU validates Duke", "fastMRI improves the Duke
  headline", any pooled/juxtaposed NYU+Duke number, "detects cancer in healthy tissue", "early
  detection", growth-rate/kinetics.

## What stays in force (unchanged)

- LOCK-1 (claim discipline, incl. cross-institution + early-detection FORBIDDEN) — unchanged.
- LOCK-2 (leakage/eval integrity, patient-level sealed splits, train-only preprocessing) — unchanged.
- ADR-0008 Duke imaging-fusion null — unchanged; co-locates with any slot result.
- LOCK-5 ($150 compute cap) — unchanged.

## Ratification block

Proposed 2026-08-04. /red-team verdict: **CONDITIONAL PASS 2026-08-04** (hostile, fresh-context) — 6
findings, all folded before ratification (slot Δ ±0.005 outcome-gate; H6 hard manifest interlock; H5
chronological-only; comparison-vs-firewall juxtaposition rule; DeLong-lower-bound go/no-go; images-only
CI assert + NYU biomarker cols in FORBIDDEN_FEATURES). PI budget sign-off: **$0-local primary (RTX 4060);
≤$5 RunPod fallback; within LOCK-5.** PI ratification: **ACCEPTED — Richard, 2026-08-04.** Reopening to
attach real (unfrozen) weights or to train H6 requires a NEW ADR + fresh /red-team.

Plan: process/general-plans/active/fastmri-encoder-deferred-fusion_04-08-26/fastmri-encoder-deferred-fusion_PLAN_04-08-26.md
