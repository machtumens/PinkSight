# ADR-0008: Reframe G3 imaging-fusion from "honest-null" headline to architecture-contribution + v2.0 data-hypothesis (positioning only — asserts no imaging signal)

**Status:** ACCEPTED — ratified by Richard 2026-07-16
**Date drafted:** 2026-07-15
**Supersedes/relates:** the `[G3-FUSION-ARCH-BUNDLE]` decisions.md OUTCOME entry (2026-07-15, committed 496b600), which currently headlines the result as HONEST-NULL. Relates to ADR-0001 (imaging encoder), ADR-0006 (eyes-open integrity-risk precedent). Governed by LOCK-1 (claim discipline) and LOCK-2 (leakage) — **neither is amended by this ADR.**

---

## Context

Richard's instruction (2026-07-15): *"I don't think it's a null. Put it as part of the model's architecture — it just needs better data later for v2.0. Nothing is a null here."*

The G3 fusion bundle (hierarchical staged fusion #4 + biology-gated MoE #7 + counterfactual XAI #3) ran end-to-end and is committed. The committed decisions.md entry headlines it as an HONEST-NULL. This ADR asks whether that headline can be changed to position G3 as a **delivered architecture contribution with a forward v2.0 data-hypothesis**, rather than as a null.

The claim ledger (LOCK-1) and the guard note in docs/CLAIM_LEDGER.md warn that the *natural* way to describe imaging work drifts toward forbidden framing ("needs better data" → implies signal exists → implies detection). So this reframe is a claim-ledger surface and gets an ADR + red-team pass before any record changes — exactly the ADR-0006 discipline.

---

## The empirical result (recorded honestly, up front)

Frozen split_v2.yaml, patient-level, 3 seeds (s0–s2), leakage CI green throughout. Full ablation ladder:

| Rung | AUROC | DeLong 95% CI | vs clinical-alone 0.708 |
|---|---|---|---|
| radiomics (G1 floor) | 0.567 | [0.510, 0.624] | −0.141 |
| unimodal MRI (G2 encoder) | 0.518 | [0.462, 0.575] | −0.190 |
| flat fusion (G3 floor) | 0.636 | [0.580, 0.692] | −0.072 |
| **hierarchical staged fusion (#4)** | **0.599** | **[0.495, 0.610]** | **−0.109** |
| **biology-gated MoE (#7, grade-band)** | **0.645** | [0.560, 0.668] (single-seed) | **−0.063** |
| clinical-alone (H6 anchor) | 0.708 | — (anchor; shuffle 0.505) | — |

**State it plainly:**
- **#4's DeLong CI lower bound is 0.495 — it crosses 0.50.** At the primary seed, hierarchical fusion is **not statistically distinguishable from chance** for imaging→subtype.
- **#4 (0.599) is below the pre-existing flat-fusion floor (0.636)** and below clinical-alone (0.708). The sophisticated architecture did **not** beat the simpler baseline.
- **#7 (0.645) is single-seed** and its routing variable (Nottingham grade) partially correlates with subtype; it still sits below clinical-alone. DeLong Δ(hierarchical − clinical) = **−0.109**.
- This is **on top of** G2's 6-axis-independent imaging→subtype null (H4 information ceiling DeLong UB ≤ 0.624; H6 MRI non-additive, Shapley −0.025, clinical the sole significant modality).

**The one thing that is solid:** the *architecture* is real, leakage-controlled, and reusable — the HR-status routing leakage (fake AUROC 1.0000) was caught and rejected in favour of leakage-safe grade-band routing (max expert purity 0.875). A working, leakage-safe fusion scaffold is a legitimate methods contribution. A demonstrated imaging-signal lift it is not, and this ADR does not claim otherwise.

---

## Decision (proposed)

**Change the G3 headline from "HONEST-NULL" to "architecture delivered; imaging-fusion signal data-limited on the Duke cohort (characterised ceiling); v2.0 forward-hypothesis."** This is a **positioning/emphasis change only.** Concretely, the submission and decisions.md may lead with:

1. **The architecture as a methods contribution** — a novel hierarchical late-clinical fusion firewall + a leakage-safe biology-gated MoE, with a documented, caught-and-rejected leakage failure mode. This is claimed as *design/method*, never as *performance*.
2. **The imaging-fusion outcome as a characterised information ceiling** — reported with every number carrying its DeLong CI (LOCK-1), consistent with G2. Clinical is the sole significant carrier.
3. **A v2.0 forward-hypothesis** — *"whether this architecture exploits imaging signal in a cohort where such signal is demonstrable is untested — an explicit v2.0 hypothesis, not a claim that the current result is under-powered."*

No LOCK is amended. No FORBIDDEN framing is removed. The word "null" is demoted from the *headline*; the null *finding* (imaging adds no separable signal) remains stated in the body, with its CIs.

---

## What this ADR does NOT do (the honesty firewall — read before ratifying)

This is the part that makes the reframe survivable. Ratifying this ADR does **NOT** license:

- ❌ Any claim that **imaging fusion works**, produces a signal, or beats a baseline. It does not (0.599 < 0.636 floor; CI crosses 0.50).
- ❌ Presenting **0.599 or 0.645 as a "respectable imaging result."** They are reported only with their CIs and only as characterisation of the ceiling.
- ❌ Asserting the result is **"under-powered" or "just needs more data"** as a *fact*. That contradicts G2's characterised information ceiling. v2.0 is a **hypothesis**, phrased as such, or it is forbidden-framing drift.
- ❌ Removing or weakening any **FORBIDDEN** framing (early/pre-detection, growth-rate/kinetics, clinical-grade FP/FN, cross-institution generalisation) — all remain in force verbatim.
- ❌ Overturning the **statistical fact** that #4 is not separable from chance at the primary seed. No signature can move a confidence interval.

If a mention violates any of the above, it is out of compliance with this ADR, not licensed by it.

---

## Framing guard (verbatim rules every G3 mention must follow)

- **ALLOWED wording:** "hierarchical late-clinical fusion architecture", "leakage-safe biology-gated MoE", "characterised imaging-fusion information ceiling", "modality redundancy — clinical is the sole significant carrier", "v2.0 hypothesis: test the architecture where imaging signal is demonstrable".
- **FORBIDDEN wording:** "imaging fusion works", "respectable imaging result", "0.60 subtype signal", "needs better data" *stated as fact that signal exists*, "the null is wrong / there is no null", plus all standing LOCK-1 bans (early detection, growth rate, kinetics, generalisation).
- **Every reported number carries its DeLong CI** and, for #4, the "CI crosses 0.50 / not separable from chance at the primary seed" caveat — never a bare AUROC (LOCK-1 / LAW L-2).
- **Duke-cohort result only.** No cross-institution claim (stays FORBIDDEN under LOCK-1).

The guard exists because "it's not a null, it just needs data" is the *natural, seductive* way to describe this result, and it is precisely the drift a hostile OPSI judge is trained to exploit.

---

## Consequences / risks (honest — red-team baked in)

A hostile-OPSI-judge red-team pass (2026-07-15) attacked the reframe. The findings, ranked, and how this ADR bounds each:

1. **CRITICAL — the CI kills "not a null."** 0.599, CI [0.495, 0.610], lower bound < 0.50. *Bounded by:* this ADR forbids asserting signal and requires the CI + "not separable from chance" caveat on #4 everywhere. The reframe is positioning, not a statistical claim.
2. **CRITICAL — "needs better data" implies signal exists**, contradicting G2's information ceiling. *Bounded by:* v2.0 is recorded strictly as a hypothesis; the ceiling finding stays in the body.
3. **HIGH — #4 (0.599) is below the flat-fusion floor (0.636).** The new architecture underperforms the baseline. *Bounded by:* the architecture is claimed as method/design (leakage-safe scaffold), never as a performance win; the ablation ladder is shown in full, floor included.
4. **HIGH — MoE 0.645 is single-seed + grade-correlated.** *Bounded by:* 0.645 is never headlined; grade-band leakage inspection (purity 0.875) and multi-seed status stay on record.
5. **MEDIUM — graph attack:** the knowledge graph has no evidence path to an "imaging works" node, only to ceiling/forbidden-claim nodes. *Bounded by:* the ADR's claims connect only to the characterisation/ceiling nodes they are actually supported by.

**Residual risk Richard accepts by ratifying:** demoting "null" from the headline moves the submission's language *toward* the integrity line (same class of exposure as ADR-0006). If the framing guard ever slips — if any mention drops the CI or lets "needs data" read as "signal exists" — this becomes an attackable over-claim. The mitigation is the framing guard + CI-always + the explicit "does NOT do" firewall above. **The honest-null finding is not deleted; it is reframed as a characterised ceiling with an architecture contribution in front of it.**

---

## What stays in force (unchanged by this ADR)

- LOCK-1 FORBIDDEN list (early/pre-detection, growth-rate/kinetics, clinical-grade FP/FN, cross-institution generalisation) — verbatim.
- LOCK-2 leakage & evaluation integrity — patient-level splits, FORBIDDEN inputs excluded, DeLong CI + shuffle sentinel on every number.
- The G2 imaging→subtype null (H4/H6) and its 6-axis-independent characterisation.
- The committed `[G3-FUSION-ARCH-BUNDLE]` decisions.md numbers (this ADR reframes their *headline*, not the numbers).
- #3 counterfactual XAI remains BLOCKED/deferred to G5 (no trained spatial encoder).

---

## Alternatives considered

- **A — Report-framing tweak only (no ADR).** Rejected: Richard asked to formally change the recorded designation; a silent report tweak would not be on the decisions.md record and would not carry a red-team pass. B is the disciplined path.
- **C — Keep "HONEST-NULL" as the headline.** The status quo. Defensible and maximally safe, but does not reflect Richard's judgment that the architecture is a first-class deliverable. This ADR is the middle path: architecture-forward headline, null finding preserved in the body with CIs.
- **D — Assert "imaging fusion works, needs better data."** REJECTED as unratifiable: contradicts the DeLong CI and the G2 information ceiling; a hostile judge dismantles it with our own numbers. No ADR can license it. This is the boundary the "does NOT do" firewall enforces.

---

## Ratification block

By signing, Richard confirms he has read **"What this ADR does NOT do"** and the **red-team consequences**, and accepts the residual integrity risk of an architecture-forward headline, bounded by the framing guard.

- [x] Richard — 2026-07-16 — **ratified: reframe G3 headline per this ADR (positioning only; asserts no imaging signal)**

On ratification: flip Status to ACCEPTED; append a dated status block; update the `[G3-FUSION-ARCH-BUNDLE]` decisions.md entry's *headline* (not its numbers) to the architecture-forward framing; update the project's current-gate status accordingly.

---

## Status update — 2026-07-15 (draft written; awaiting Richard)

Drafted with red-team pass folded in. The maximum defensible claim is: *a validated, leakage-safe fusion architecture (methods contribution); imaging adds no separable subtype signal beyond clinical on the Duke cohort (characterised ceiling, #4 CI crosses 0.50); v2.0 hypothesis to test where imaging signal is demonstrable.* Not ratified. No decisions.md / docs/CLAIM_LEDGER.md / gate edits made pending Richard's signature.

## Status update — 2026-07-16 (RATIFIED)

Richard ratified as drafted. Status ACCEPTED. Applied this session: (a) decisions.md `[G3-FUSION-ARCH-BUNDLE]` headline reframed via a dated append (LAW L-1 — original honest-null entry preserved byte-for-byte, numbers unchanged); (b) the project's current-gate status updated to architecture-forward framing with the #4 CI-crosses-0.50 caveat retained. The "What this ADR does NOT do" firewall and the framing guard are in force for every G3 mention.
