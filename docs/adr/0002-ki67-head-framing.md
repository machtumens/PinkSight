# ADR-0002: Ki-67 head — fixed 14% cutoff, continuous-primary, contingent on the G0 count
Date: 2026-06-20   Status: proposed (records OPEN-LOCK **O-2**; ratify at G0 once usable N is known)

## Context
decisions.md OPEN-LOCK **O-2**: Ki-67 is NOT an enumerated field in the Duke
Clinical_and_Other_Features table — it appears only in radiogenomics free text — so the usable
numeric N is unknown until the G0 audit (`scripts/audit_ki67.py`) counts it. Two forces shape the
head's design:

- **Circularity risk.** Ki-67 is one of the IHC markers that define the molecular-subtype LABEL. A
  data-driven cutoff (e.g. Youden-optimised against subtype) would let the label leak into its own
  threshold. The St. Gallen consensus **14%** cutoff is fixed and external, keeping Head 2
  independent of the subtype head and of the FORBIDDEN inputs (**LOCK-2**; the [1.16] leak-safe list).
- **Sample size.** If the usable numeric N is small, a continuous regression head is underpowered and
  Ki-67 must become a clearly-labelled sub-analysis, with subtype kept as the primary endpoint.

This head is the *statistical* kind of regression — it outputs a proliferation percentage at the time
of diagnosis. It is a cross-sectional, at-diagnosis index and must only ever be described with the
ALLOWED framings of **LOCK-1** (proliferation / aggressiveness), never with any framing on LOCK-1's
FORBIDDEN list (growth rate / doubling time / kinetics).  <!-- # allow-ledger: names the bans to forbid them -->

## Options considered
1. **Fixed 14% St. Gallen cutoff, continuous-primary (recommended)** — model Ki-67 as a continuous
   value (Huber loss) with a binary low/high call at 14% for reporting. Pro: no circularity; matches
   clinical convention; one defensible threshold. Con: 14% is a soft consensus, not a constant —
   report sensitivity around it.
2. **Data-driven / Youden cutoff** — rejected: tunes the threshold against the very label Ki-67 helps
   define → leakage + overfitting (a **LOCK-2** violation); exactly what a hostile judge probes for.
3. **Demote Ki-67 to a sub-analysis now** — premature: the G0 count may well support a primary head;
   deciding before the number exists violates "every gate produces a number" (**LAW L-2**).

## Decision
<PENDING the G0 usable-N number. Recommended: Option 1, with an explicit demote-to-sub-analysis rule
if usable numeric N falls below the pre-registered threshold (`docs/pre_registration.md`, tagged
ILLUSTRATIVE until ratified at G0). Aligns with [1.1] (head form contingent on the G0 count) and
[1.2-R] / [8.3] (fixed 14% St. Gallen cutoff).>

## Consequences
- The 14% cutoff is frozen in config when the pipeline lands; it is NOT re-tuned per fold.
- `DATA_CARD.md`'s N-waterfall row "has numeric Ki-67" is the gating number for this ADR.
- Reporting always pairs the binary call with the continuous value + calibration, never a bare
  threshold accuracy (**LOCK-1** reporting rule).

---
*Citations resolve against `decisions.md` → Invariant & Reference Register (v1.1). Template: `docs/templates/ADR_TEMPLATE.md`.*
