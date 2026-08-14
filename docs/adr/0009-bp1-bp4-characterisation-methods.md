# ADR-0009: Record BP1 (BPE augmentation) + BP4 (selective characterisation) as delivered characterisation-METHOD contributions (positioning only — asserts no imaging/BPE signal)

**Status:** DRAFT — pending `/red-team` pass + Richard ratification
**Date drafted:** 2026-07-16
**Supersedes/relates:** the BP1+BP4 code commit `c930c44` (`feat(explore): BP1 BPE-augmentation + BP4 selective-characterisation methods`). Relates to ADR-0008 (G3 architecture-forward reframe — the direct precedent for this framing move) and ADR-0001 (imaging encoder). Governed by LOCK-1 (claim discipline) and LOCK-2 (leakage) — **neither is amended by this ADR.**

---

## Context

Richard's instruction (2026-07-16): *"commit the BP4 and BP1, as part of the architecture. it's not a null, as what we wanted is the overall model innovation, not just the AUROC."*

BP1 and BP4 are two research-synthesis blueprints (`reports/research_synthesis/BLUEPRINTS_16-07-26.md`) built and run this session as **exploration** (off-ledger — no `decisions.md` gate entry). The code is committed as architecture (`c930c44`). Both produced null/uniform-null empirical outcomes. This ADR asks whether they can be recorded as **delivered method/architecture contributions** — the reusable BPE-augmentation pipeline and the selective-characterisation framework — with the AUROC outcomes recorded as characterisation of the ceiling, rather than filed as "two more nulls."

This is the same claim-ledger surface ADR-0008 governs: "it's not a null, it's the method" is the *natural, seductive* way to describe this, and it drifts toward forbidden framing ("BPE helps" / "imaging has a confident subpopulation" / "needs better data"). So the reframe gets an ADR + red-team pass before any durable record designates it a contribution — the ADR-0006/0008 discipline.

---

## The empirical result (recorded honestly, up front)

Frozen `split_v2.yaml`, patient-level, leakage-safe throughout. Both are exploration numbers on the Duke cohort.

### BP1 — contralateral-BPE clinical-stream augmentation (908/922 patients extracted)

Paired DeLong, clinical-only vs clinical+BPE, same fold partitions (3 seeds):

| Stratum | clinical-only | clinical+BPE | paired ΔAUROC | paired p (2-sided) | seeds p<0.05 |
|---|---|---|---|---|---|
| **Postmenopausal (pre-registered PRIMARY)** | 0.621 | 0.612 | **−0.009** | **0.82** | 0/3 |
| Full dev cohort | 0.634 | 0.657 | +0.023 | 0.37 (range 0.07–0.83) | 0/3 |

- **F-B1.1 redundancy:** BPE is **non-redundant** with age/menopause (|r| < 0.15 for all three features) — a genuine *non-additive-signal* null, not a redundancy artifact.
- **Calibration:** base ECE ≈ 0.23 (both arms poorly calibrated); BPE's ECE effect is negligible (full 0.232→0.224).

### BP4 — calibrated selective characterisation (imaging-only, frozen-embedding probe, n=613)

- Imaging-only OOF **AUROC 0.539, DeLong [0.480, 0.598]** — CI crosses 0.50 (reproduces the ~0.518 imaging null).
- **Coverage–AUROC curve falls** as coverage tightens: 100%→0.539, 80%→0.511, 40%→0.472, 20%→0.442. **Lift = −0.067.**
- **F-B4.2 easy-case audit is empty:** confident-subset selection uncorrelated with age (r=+0.02), tumour size (r=−0.02), scanner (χ² p=0.44), BPE (r=−0.08).
- Temperature scaling did **not** help (ECE 0.099→0.115).

**State it plainly:**
- **BP1's pre-registered primary endpoint is a clean null** (postmenopausal Δ−0.009, p=0.82). The full-cohort +0.023 is **not significant** (p=0.37, 0/3 seeds) and is the premenopausal BPE↔age↔TNBC confound the postmenopausal-primary design was built to exclude.
- **BP4's imaging null is UNIFORM** — no reliable confident subpopulation, and confidence *anti*-correlates with accuracy (negative lift). Not a heterogeneous mixture, not an easy-case artifact — structureless.
- Both sit **on top of** the G2 6-axis imaging→subtype null and the G3 fusion-layer null (ADR-0008).

**The one thing that is solid:** the *methods* are real, leakage-controlled, reusable, and now part of the stack — a contralateral-BPE extraction pipeline (LOCK-1/2 guarded) + a leakage-checked `load_xy(bpe_npz=…)` augmentation hook + a predictor-agnostic selective-characterisation framework (`eval/selective.py`) that *measures whether a null is uniform or heterogeneous*. A working, reusable characterisation toolkit is a legitimate methods contribution. A demonstrated BPE or imaging signal it is not, and this ADR does not claim otherwise.

---

## Decision (proposed)

**Record BP1 and BP4 as delivered characterisation-METHOD contributions, with their empirical outcomes as characterisation of the imaging/BPE ceiling — not as performance results.** Positioning/emphasis only. The submission and any exploration record may lead with:

1. **The methods as contributions** — (a) a physics-principled contralateral-BPE augmentation pipeline with a data-driven tissue floor and a leakage-safe clinical-stream hook; (b) a reusable selective-characterisation framework (coverage–AUROC + easy-case drivers) that *diagnoses the structure of a null*. Claimed as *design/method/tooling*, never as *performance*.
2. **The outcomes as characterised ceilings** — BPE is non-additive to clinical for subtype in the confound-free stratum; the imaging null is uniform (no gate-able subpopulation). Every number carries its CI / significance (LOCK-1).
3. **No forward-hypothesis inflation** — unlike ADR-0008's v2.0 clause, BP1/BP4 add *no* "needs better data" hypothesis. They are recorded as they are: methods that characterised the ceiling and found it flat.

No LOCK is amended. No FORBIDDEN framing is removed. "Null" is demoted from the *headline* (the methods lead); the null *findings* remain stated in the body with their CIs/p-values.

---

## What this ADR does NOT do (the honesty firewall — read before ratifying)

Ratifying this ADR does **NOT** license:

- ❌ Any claim that **BPE improves subtype characterisation.** It does not (postmenopausal Δ−0.009, p=0.82; full-cohort +0.023 not significant).
- ❌ Any claim that **a confident imaging subpopulation exists** or that imaging is "selectively informative." It is not (coverage-AUROC lift −0.067, uniform null).
- ❌ Presenting **0.657 (BP1) or 0.539 (BP4) as respectable results.** They are reported only with CIs and only as ceiling characterisation.
- ❌ Quoting BP1's Δ **against the 0.708 headline.** This runner's clinical-only baseline is **0.634** (pooled-OOF, ≈ the P05 0.626), a different metric/config than the G3 ablation's 0.708. The baseline reconciliation is an OPEN gap; no Δ-vs-0.708 claim until it is closed.
- ❌ Asserting the results are **"under-powered" or "just need more data"** as fact. That contradicts the G2/G3 characterised ceiling.
- ❌ Removing or weakening any **FORBIDDEN** framing (early/pre-detection, growth-rate/kinetics, clinical-grade FP/FN, cross-institution) — all remain verbatim. (BPE = systemic hormonal context at diagnosis, never contralateral early-detection.)
- ❌ Overturning the **statistical facts** — BP1's primary null (p=0.82) and BP4's negative lift. No method framing moves a p-value or a CI.

If a mention violates any of the above, it is out of compliance with this ADR, not licensed by it.

---

## Framing guard (verbatim rules every BP1/BP4 mention must follow)

- **ALLOWED wording:** "contralateral-BPE clinical-stream augmentation method", "leakage-safe BPE→FT-Transformer hook", "selective-characterisation framework", "characterised: BPE is non-additive to clinical for subtype", "characterised: the imaging null is uniform (no gate-able subpopulation)", "the method/tooling is the contribution".
- **FORBIDDEN wording:** "BPE improves subtype / adds signal", "imaging has a confident subpopulation", "selective prediction rescues the null", "0.66 subtype signal", "needs better data" *stated as fact that signal exists*, "not a null / the null is wrong", plus all standing LOCK-1 bans.
- **Every reported number carries its CI/significance** — BP1 with "Δ not significant (p=0.82 primary)"; BP4 with "lift −0.067 / CI crosses 0.50" — never a bare AUROC (LOCK-1 / LAW L-2).
- **Duke-cohort exploration only.** No cross-institution claim; no `decisions.md` gate promotion without a separate gate.

The guard exists because "the methods are the innovation, the AUROC doesn't matter" can *slide* into "so the null doesn't matter" — and dropping the null is exactly the drift a hostile OPSI judge exploits.

---

## Consequences / risks (preliminary self-adversarial pass — **formal `/red-team` PENDING**)

A hostile-OPSI-judge lens, pre-empting the formal pass, ranked:

1. **CRITICAL — "it's not a null" is false at the number level.** BP1 primary p=0.82; BP4 lift −0.067. *Bounded by:* this ADR reframes only the *headline emphasis* (methods lead), forbids asserting signal, and requires the p-value/CI on every mention. The null findings stay in the body.
2. **CRITICAL — baseline mismatch (0.634 vs 0.708) invites an accidental over-claim.** *Bounded by:* the firewall forbids any Δ-vs-0.708 statement and flags the reconciliation as an open gap.
3. **HIGH — "method contribution" with a null result reads as padding** unless the tooling's reusability is concrete. *Bounded by:* the claim is scoped to the *reusable, leakage-guarded pipeline + framework* (committed, tested), never to a finding of signal.
4. **MEDIUM — BP4 confidence = probability margin, not ensemble variance** (BP4 §5 specifies MC-Dropout / epistemic variance). *Bounded by:* recorded as a known caveat; the uniform-null conclusion is robust given zero covariate structure, but the variance-based version is untested.
5. **MEDIUM — BP4 Step 6 (fusion selective prediction) not run** (G3 saved no per-sample OOF). *Bounded by:* recorded as a documented gap, not a silent omission.

**Residual risk Richard accepts by ratifying:** leading with "methods contribution" over "null" moves the language toward the integrity line (same class as ADR-0006/0008). If the framing guard slips — if any mention drops the p-value/CI or lets "the innovation is the method" read as "so imaging/BPE works" — it becomes an attackable over-claim. Mitigation: the framing guard + CI/p-always + the "does NOT do" firewall. **The null findings are not deleted; they are reframed as characterised ceilings with the delivered methods in front of them.**

---

## What stays in force (unchanged by this ADR)

- LOCK-1 FORBIDDEN list (early/pre-detection, growth-rate/kinetics, clinical-grade FP/FN, cross-institution) — verbatim.
- LOCK-2 leakage & evaluation integrity — patient-level splits, FORBIDDEN inputs excluded, DeLong CI + significance on every number.
- The G2 imaging→subtype null and the G3 fusion null (ADR-0008).
- BP1 and BP4 remain **exploration** — this ADR does not promote them to a gate outcome or a `decisions.md` claim; it records their designation as method contributions.
- Open gap: BP1 baseline reconciliation (0.634 pooled-OOF vs 0.708 ablation) — must be closed before any absolute-Δ statement.

---

## Alternatives considered

- **A — Commit-message framing only (no ADR).** Rejected: the commit body carries the firewall, but a "methods contribution" designation is a claim-ledger surface; it belongs on the durable ADR record with a red-team pass, per ADR-0008.
- **B — This ADR (methods-forward designation, null findings preserved with CIs).** Chosen path: architecture-forward, honest body, firewalled.
- **C — File both as plain nulls, no contribution framing.** Maximally safe; does not reflect Richard's judgment that the reusable pipeline + framework are first-class deliverables.
- **D — Assert "BPE helps / imaging has a gate-able subpopulation, needs better data."** REJECTED as unratifiable: contradicts the p-value, the CI, and the G2/G3 ceiling; a hostile judge dismantles it with our own numbers. No ADR can license it — the boundary the "does NOT do" firewall enforces.

---

## Ratification block

By signing, Richard confirms he has read **"What this ADR does NOT do"** and the **red-team consequences**, accepts the residual integrity risk of a methods-forward designation bounded by the framing guard, and confirms the formal `/red-team` pass has been run.

- [ ] `/red-team` pass run and findings folded in
- [ ] Richard — 2026-07-__ — **ratified: record BP1+BP4 as characterisation-method contributions per this ADR (positioning only; asserts no imaging/BPE signal)**

On ratification: flip Status to ACCEPTED; append a dated status block; (optionally) add a firewalled BP1/BP4 note to `JOURNAL.md`; keep both as exploration (no `decisions.md` gate entry) unless a separate gate is opened.

---
