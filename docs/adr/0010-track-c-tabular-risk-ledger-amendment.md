# ADR-0010: LOCK-1 path-scoped carve-out admitting a Track-C public-benchmark tabular-risk suite (detection / incidence / prognosis), as an ENSEMBLE companion panel — not cross-attention fusion

**Status:** ACCEPTED (Scope C) — Richard ratified via in-session scope decision 2026-07-16; fixes-first gate satisfied (EVL green, below). Operative LOCK-1 amendment recorded as the dated LAW-L-1 change-control entry in `decisions.md`.
**Date drafted / accepted:** 2026-07-16
**Amends:** **LOCK-1** (Claim & reporting discipline) — path-scoped exemption of the FORBIDDEN framings for Track-C artifacts only.
**Relates:** ADR-0006 (recurrence-stratification organ — the standalone-companion precedent + the direct LOCK-1-amendment template), ADR-0008 (G3 architecture-forward reframe), the off-ledger tabular exploration (`explore/tabular_risk/`).
**Does NOT amend:** LOCK-2 (leakage/eval integrity), LOCK-3..LOCK-6. Track-A (Duke) keeps un-amended LOCK-1 **verbatim**.

---

## Context

Richard's instruction (2026-07-16): *"commit these to the whole model, and integrate it to the actual architecture"* → scope call (after being shown the two collisions): **"Amend the ledger"** → after red-team: **"Ratify Scope C + fixes."**

Four public-benchmark tabular models were built off-ledger this session (`explore/tabular_risk/`, all `off_ledger: true`). Two collisions were surfaced (STOP-and-FLAG) before any commit; this ADR is the change-control instrument that resolves them:

1. **Claim-ledger collision (LOCK-1).** Coimbra = *detection* ("has cancer?"); BCSC = *incidence risk* (screening) — both FORBIDDEN "early/pre-detection" verbatim. The four together are a *cross-institution* surface (FORBIDDEN).
2. **Architecture collision (physical).** Coimbra (N=116, PT), BCSC (2.39 M women, US), METABRIC (N=1 917, UK/CA), Duke-recurrence (N=920, US) share **zero patients**. Cross-attention fusion binds modalities *of the same patient*; with no shared patients these can only be an **ensemble of independent per-cohort stratifiers**. The amendment removes collision (1) by carve-out; it cannot remove collision (2) — so the committed form is an **ensemble companion panel**, never fusion, never "integrated into the imaging encoder."

---

## The empirical result (honest, up front — sources: `explore/tabular_risk/results/*_metrics.json`, EVL-confirmed 2026-07-16)

All 5-fold OOF, LightGBM depth-3, patient/strata-level. Each on its **own** cohort — no cross-cohort transfer tested or claimed. **Raw OOF ECE (all fail the ≤0.10 bar) is shown alongside the honest held-out calibrated ECE** — the calibrators were re-fit nested/held-out this session after a leak was found (see §Fixes).

| Model | Task (label class) | Cohort | AUROC (seed 42) | CI95 | raw OOF ECE | held-out calibrated ECE | notes |
|---|---|---|---|---|---:|---:|---|
| **Coimbra** | blood biomarkers → **cancer vs healthy (detection)** | UCI, N=116 (64/52) | **0.806** | [0.721, 0.887] | 0.118 | **0.069** (sigmoid) ✅≤0.10 | POC only; N=116; CI width 0.166 — quote DeLong CI, never the ±0.019 reseed spread |
| **BCSC** | **1-yr incidence risk** (invasive+DCIS) | 2.39 M women / 280 660 strata | **0.634** | [0.625, 0.642] | 0.117 | **6.2e-06** (isotonic) † | PR-AUC **0.0083** @ 0.486% prev (~1.7× chance) — the honest headline; shuffle→0.50 |
| **METABRIC** | **5-yr overall-survival (prognosis)** | cBioPortal, N=1 917 (22.3% pos) | **0.744** | [0.719, 0.770] | 0.186 | **0.018** (sigmoid) ✅≤0.05 | features include ER/PR/HER2/CLAUDIN_SUBTYPE — **LOCK-2 leak if ever fed to the subtype head** (CI now blocks it) |
| **Duke recurrence** | at-diagnosis recurrence stratification | Duke, N=920 (87 events) | **0.577** | [0.482, 0.614] **crosses 0.50** | 0.256→0.007 | (nested recal) | **already the ADR-0006 organ** — no new ledger scope needed |

**†** BCSC's held-out calibrated ECE 6.2e-06 is a **documented base-rate-deflation artifact** at 0.486% prevalence (`small_ece_is_base_rate_artifact_not_leak: true`, `leak_check.passed`, AUROC unmoved, Brier 0.0214→0.0048) — NOT the in-sample leak class. **Must always be reported with the base-rate caveat, never as standalone "perfect calibration."**

**State it plainly:** Coimbra is detection by label construction (you cannot re-frame the training label); its 0.806 is a small-N POC, not clinical utility. BCSC is a screening/incidence model; its honest headline is PR-AUC-at-prevalence + calibration, not AUROC. METABRIC (prognosis) is the cleanest and maps onto the ADR-0006 at-diagnosis family, but is partly carried by IHC features Track-A forbids. Duke-recurrence is a near-null already covered by ADR-0006.

---

## Decision (ratified — Scope C)

Amend LOCK-1 by a **named, walled, path-scoped exemption** (the FORBIDDEN list is NOT deleted). Introduce:

### Track C — public-benchmark tabular-risk suite (committed companion panel; ADR-0006 standalone-organ family)

1. **Amendment scope (path-scoped).** Within **Track-C artifacts only** (`explore/tabular_risk/` + `reports/**/track_c*`/`TRACK_C*`), the framings *detection / case-control* (Coimbra), *incidence / screening-population risk* (BCSC), *prognosis / survival* (METABRIC) are **ALLOWED**, each bound to its cohort and reported **per-cohort as an independent public benchmark**. The suite may be discussed together **only as a panel of independent benchmarks**, never as evidence one model transfers to another's population.
2. **Form = ENSEMBLE companion panel, NOT fusion (binding).** Track C is four independent per-cohort LightGBM stratifiers attached as a **separately-reported companion panel** (ADR-0006 family). It is **NOT** cross-attention-fused with the Duke DCE-MRI encoder, shares **no patients** with Track A, has **no** modality-dropout stream and **no** shared latent. The words "fusion" and "integrated into the architecture" are **banned** for Track C.
3. **Reporting discipline survives the amendment.** Every number carries its CI + calibration; BCSC additionally carries PR-AUC-at-prevalence + the base-rate caveat; the label-shuffle sentinel stays on every model. **Cohorts are never tabled together in one figure** (each gets its own task-labelled figure with a "single-cohort OOF; no cross-cohort transfer tested" caption).

### What LOCK-1 STILL FORBIDS after this amendment (unchanged, everywhere including Track C)
- ❌ **growth-rate / tumour kinetics / doubling-time** — forbidden verbatim, all tracks.
- ❌ **clinical-trial-grade FP/FN reduction** claims — forbidden, all tracks.
- ❌ **cross-institution generalisation / transfer** claims — the panel is independent benchmarks, never a transfer result.
- ❌ Any Track-A (Duke subtype/Ki-67) artifact using detection/incidence framing — Track A keeps **un-amended** LOCK-1 verbatim.
- ❌ Letting a Track-C model's framing bleed onto the Duke headline ("PinkSight detects cancer" from Coimbra) — the drift this firewall stops.

---

## What this ADR does NOT do (the honesty firewall)

Ratifying this ADR does **NOT** license: claiming the **Track-A model** does detection/incidence; any **cross-institution generalisation**; **fusing** Track C into the imaging encoder; feeding **METABRIC ER/PR/HER2/subtype into the Duke subtype head** (LOCK-2 leak, now CI-blocked); presenting **Coimbra 0.806 as clinical utility** (N=116) or **BCSC 0.634 without its PR-AUC + calibration + base-rate caveat**; removing the growth-rate/kinetics or FP/FN bans; or overturning any statistical fact (Duke-recurrence CI crosses 0.50; BCSC PR-AUC ~1.7× chance; Coimbra CI width 0.166). A mention that violates any of these is out of compliance with this ADR, not licensed by it.

---

## Fixes folded in from the `/red-team` pass (2026-07-16 — EVL-confirmed)

| Red-team finding | Resolution | Evidence |
|---|---|---|
| **C2 (CRITICAL) — calibration leak** (Coimbra ECE ~1e-17 = in-sample `fit_isotonic(y,oof,oof)`; 3 raw ECEs fail bar) | **FIXED.** Re-fit calibrators nested/held-out. Honest OOF calibrated ECE: Coimbra 0.069, METABRIC 0.018, BCSC 6.2e-06 (base-rate-flagged). 1e-17 deleted. Raw AUROCs unchanged. | `explore/tabular_risk/results/*_metrics.json`; independently reproduced by vc-tester (0.06862 exact) |
| **H1 (HIGH) — METABRIC IHC → subtype leak path** | **FIXED (guard live).** 2 CI tests block METABRIC ER/PR/HER2/CLAUDIN_SUBTYPE from Track-A `clinical_encoder.FEATURES`. | `tests/test_leakage.py` 8/8 → **10/10 green** |
| **M1 (MEDIUM) — linter contradicts amendment** | **FIXED.** `ledger_lint.py` path-scopes detection/incidence to Track-C; growth-rate/kinetics/FP-FN/cross-institution still banned everywhere; Track-A violation still fails. | `ci/ledger_lint.py --selfcheck` PASS (4 ADR-0010 branches) |
| **C3 (CRITICAL) — "architecture/fusion" over-read** | **BOUND.** Ensemble-companion-panel framing; "fusion"/"integrated" banned for Track C (§Decision 2). | this ADR |
| **H2 (HIGH) — panel reads as generalisation** | **BOUND.** No cohorts tabled together; per-cohort caption mandatory (§Decision 3). | this ADR |
| **H3 (HIGH) — Coimbra small-N CI** | **BOUND.** Quote DeLong [0.721, 0.887]; never the ±0.019 reseed spread as uncertainty. | this ADR |
| **C1 (CRITICAL) — goalpost-move optics** | **ACCEPTED as residual risk** (unfixable by code — only framing discipline). | §Residual risk |

---

## Residual risk Richard accepts by ratifying

This **amends LOCK-1, the project's #1 invariant.** Even scoped to Track C, it is a materially larger move than ADR-0006/0008 (which preserved LOCK-1). The single worst optic (red-team C1): the amendment is dated *after* the G2/G3 imaging null, so a hostile OPSI judge can read it as *moving the integrity goalposts once the imaging thesis failed*, and the mere presence of a detection model (Coimbra) invites "so you DO do detection." **No code fix removes this — only framing discipline does.** Mitigation: the Track-C wall, the ensemble-only + no-fusion-language rule, the LOCK-2 leak CI, the per-cohort-no-shared-table rule, and the "does NOT do" firewall. The honest re-fit *strengthens* the case (calibration genuinely holds up held-out — METABRIC 0.018 clears the good bar — so the suite's differentiator is real, not leaked), but does not touch C1. Ratified with C1 on record.

Rejected: **Scope D** (unscoped LOCK-1 repeal for the Duke headline) — red-team called it indefensible; not adopted.

---

## What stays in force (unchanged)
- LOCK-1 FORBIDDEN list for **Track A** (Duke subtype/Ki-67) — verbatim. growth-rate/kinetics + clinical-grade FP/FN + cross-institution — verbatim, **all tracks**.
- LOCK-2 — patient/strata-level splits, forbidden-input exclusion, DeLong/bootstrap CI + calibration on every number; the new METABRIC→subtype CI extends it.
- The G2 imaging→subtype null and the G3 fusion null (ADR-0008).

## Ratification block
- [x] `/red-team` pass run; C2/H1/M1 fixed + EVL-confirmed; C3/H2/H3 bound; C1 accepted as residual.
- [x] **Scope C** — Richard — 2026-07-16 — ratified via in-session scope decision ("Ratify Scope C + fixes"), fixes-first gate satisfied: amend LOCK-1 with a **Track-C-scoped** carve-out (detection/incidence/prognosis for public benchmarks; ensemble not fusion; Duke headline keeps un-amended LOCK-1). Countersign in the changelog when convenient.
- [ ] Scope D (unscoped repeal) — NOT adopted.

Operative amendment recorded as the dated LAW-L-1 entry in `decisions.md` (2026-07-16, `[TRACK-C-TABULAR-RISK]`).
