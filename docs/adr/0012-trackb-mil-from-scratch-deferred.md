# ADR-0012: Track-B MIL from frozen UNI2-h — ratify the ABMIL multiple-instance-learning confirmation arm as a reportable methods-rigour result for OPSI 2026

Date: 2026-08-12   Status: **ACCEPTED / RATIFIED (2026-08-12)** (ratified by a dated `decisions.md` entry by Richard — see `[ADR-0012-RATIFIED]`, 2026-08-12, LAW L-1). Bound by the firewall below; no LOCK moved.

> **This ADR ratifies a methods-rigour confirmation arm, not a new capability, headline, or data source.**
> No LOCK is amended. The MIL result moves no gate target, no Duke headline number, and no
> LOCK-1 forbidden claim. The claim ledger is unchanged. Per **LAW L-1** the operative go/no-go
> is a dated `decisions.md` entry, not this file.

---

## Context

Track-B MIL characterises **LumA vs Basal (Triple-Negative) subtype from TCGA-BRCA H&E
whole-slide images** using **frozen UNI2-h** foundation-model tile embeddings aggregated by an
attention-based multiple-instance-learning head (ABMIL). It is the hardened successor to the
degenerate 2026-07-11 TITAN POC (607/640 single-tile bags) proposed in the `[ADR-0012-PROPOSED]`
entry (2026-07-25).

The result is:

| Metric | Value |
|---|---|
| AUROC | **0.9675** [DeLong 95% CI 0.9479, 0.9871] |
| 3-seed mean ± SD | **0.9622 ± 0.0038** (per-seed 0.9618 / 0.9670 / 0.9578) |
| ECE | 0.0428 |
| Shuffle sentinel | 0.4309 — leakage clean (real ≫ shuffle) |
| Cohort | TCGA-BRCA H&E WSI, **n = 640** (LumA 475 / Basal 165), patient-level split |
| Encoder | frozen UNI2-h (`uni2h-h5-1536d`, 1536-d tile features) + ABMIL |
| Relationship to arm-3 | **same 640-patient cohort, DIFFERENT encoder** (UNI2-h vs arm-3's TITAN) → encoder-robustness, NOT independent corroboration, NOT generalisation |

The result was produced under the same leakage discipline as arm-3 (patient-level split, 3 seeds,
DeLong CI + shuffle sentinel on every number). Duke ∩ TCGA ID overlap = 0 (`dedup_report.json`,
`direct_overlap_n == 0`, re-confirmed 2026-08-12). Track-B MIL is a **confirmation arm**: it
reproduces arm-3's strong intra-TCGA-BRCA subtype signal (0.9646 TITAN → 0.9675 UNI2-h) on the
**same 640 patients** with an **independent foundation encoder**, establishing encoder-robustness
of the histology signal — NOT a second, independent cohort, and NOT any cross-institution claim.

---

## Options considered

1. **Keep Track-B MIL excluded from the manuscript (status quo, un-logged).**
   Pro: zero framing risk. Con: withholds a genuine, leakage-controlled methods-rigour result
   that strengthens arm-3's encoder-robustness story. **Rejected**: all five unlock conditions
   are met; continued exclusion under-reports verified, on-disk evidence.

2. **(CHOSEN) Ratify Track-B MIL as a reportable confirmation arm, reported alongside arm-3,
   bounded by the firewall below.**
   Pro: accurately reports the encoder-robustness of the Track-B histology signal (TITAN and
   UNI2-h agree on the same cohort). Con: requires careful framing to prevent drift into
   "independent corroboration" (same patients) or any Duke juxtaposition. Fully mitigable via the
   explicit firewall in this ADR.

3. **Report Track-B MIL as independently corroborating arm-3, or as evidence about the Duke null.**
   **REJECTED outright.** MIL and arm-3 are two encoders on the **same 640 TCGA-BRCA patients** —
   not independent evidence. The Duke null is DCE-MRI on a different institution's cohort; MIL is
   inert with respect to it. Any such framing is a LOCK-1 violation and a factual misrepresentation.

---

## Decision

**Ratify the Track-B MIL arm (TCGA-BRCA H&E, frozen UNI2-h + ABMIL, LumA vs Basal) as a
reportable methods-rigour confirmation arm for OPSI 2026, reported alongside — and clearly
distinguished from — arm-3.**

The contribution is: **standalone intra-TCGA-BRCA subtype characterisation at diagnosis from a
frozen UNI2-h encoder with attention-based MIL aggregation, confirming arm-3's TITAN result on the
same 640-patient cohort with an independent foundation encoder (encoder-robustness).**

It is reported alongside — and clearly distinguished from — the honest Duke imaging-fusion null
(clinical AUROC 0.708; imaging-fusion AUROC 0.599 [0.495, 0.610], CI crosses 0.50; characterised
information ceiling). The two results are from different cohorts, different modalities, and
different research questions. They are never framed as answering the same question.

**"Confirmation arm" vs LOCK-6.** LOCK-6 reserves "the headline" for Track A (MRI+clinical) and
governs the fusion claim; that assignment is unchanged. Track-B MIL is a Track-B *standalone*
confirmation of arm-3, not a new fusion headline and not a replacement for the Track-A headline.

---

## This ADR does NOT license (mandatory LOCK-1 firewall — read before ratifying)

Ratifying a 0.9675 histology number is exactly the kind of result that drifts naturally toward
forbidden claims. The following are **hard exclusions** that this ADR does NOT license, regardless
of how natural the framing feels:

- **"Imaging works" for the Duke DCE-MRI thesis.** Track-B MIL is TCGA-BRCA H&E histology. The
  Duke thesis is DCE-MRI on a different institution's cohort. Different modalities, different
  patient populations, different research questions. The MIL AUROC licenses no statement about
  DCE-MRI performance, and no statement about MRI-based imaging generally.

- **Any cross-cohort ceiling comparison or juxtaposition.** The sentence "Track-B MIL achieves
  0.9675 vs Duke clinical 0.708" is FORBIDDEN in any submission, slide, or summary — it implies a
  ceiling comparison that the different-cohort + different-modality design cannot support.
  TCGA and Duke results must be reported in separate paragraphs with explicit cohort labels; they
  may not be placed side-by-side in a table as if they answer the same question.

- **Any claim that Track-B MIL rescues, reopens, or overturns any Duke result.** The Duke
  imaging-fusion null (ADR-0008) stands unchanged. MIL provides no evidence about Duke MRI.

- **Cross-institution generalisation.** Track-B MIL is intra-TCGA-BRCA. It is not validated on
  Duke, and not on any external cohort. LOCK-1's cross-institution FORBIDDEN clause is unchanged
  and applies to Track-B MIL without exception.

- **Co-citing Track-B MIL and arm-3 as two independent corroborating results.**
  Track-B MIL (UNI2-h) and arm-3 (TITAN) are run on the **same 640 TCGA-BRCA patients**. They are
  **not independent evidence sources** — they are two frozen encoders on one cohort. Any
  submission, slide, or abstract that presents MIL and arm-3 as corroborating each other in a way
  that implies **cohort independence** is misrepresenting the evidence structure. The only claim
  licensed is **encoder-robustness on the same cohort** (TITAN and UNI2-h agree). MIL and arm-3
  results MUST each carry the cohort description "TCGA-BRCA, n=640" and MUST NOT be listed as
  independently-motivated or independent-cohort results.

- **Framing Track-B MIL's modality (histology) as evidence about MRI, DCE, or any imaging
  modality used in the Duke encoder.** Histology and DCE-MRI are distinct modalities with distinct
  signal sources. A high histology-based AUROC provides no evidence about the information content
  of MRI sequences.

**Permitted framing (verbatim rules every Track-B MIL mention must follow):**

- SAY: "standalone intra-TCGA-BRCA subtype characterisation at diagnosis", "frozen UNI2-h
  foundation-model embeddings", "attention-based MIL (ABMIL)", "H&E histology", "patient-level
  split, n=640", "confirms arm-3 with an independent encoder on the same cohort (encoder-robust)",
  "reported alongside the honest Duke imaging-fusion null (separate cohort, separate modality)".
- NEVER SAY: "imaging works", "rescue of the Duke null", "corroborates Duke findings",
  "cross-cohort", "generalises", "the model detects", "early detection", "growth rate",
  "kinetics", "MIL and arm-3 independently confirm", "independent corroboration", or any
  comparison table placing TCGA-BRCA and Duke results in the same row.

---

## Consequences

**Easier.** The OPSI 2026 submission reports the Track-B histology signal with encoder-robustness
evidence (TITAN 0.9646 and UNI2-h 0.9675 agree on the same 640-patient cohort), strengthening the
methods-rigour of the standalone histology result. The honest Duke null is retained in full as a
characterised information ceiling.

**Harder.** Every writeup that mentions Track-B MIL now carries a firewall obligation: cohort
label, CI, encoder name, the "same cohort as arm-3 → encoder-robustness not independence" note,
and the "separate from Duke" statement. Abstracts, slides, and the OPSI submission must not
silently compress these into a single "we achieved 0.9675" sentence without provenance.

**Commits downstream.**
- Every Track-B MIL mention in any submission artifact must carry: AUROC + DeLong CI + cohort
  (TCGA-BRCA, n=640) + modality (H&E histology, frozen UNI2-h + ABMIL) + split (patient-level,
  3 seeds) + shuffle sentinel (0.4309) + the "same-cohort-as-arm-3, encoder-robustness" notice.
- MIL and arm-3 co-occurrence in any document requires the shared-cohort notice (same 640
  patients) and must not be styled as independent-cohort corroboration.
- The existing juxtaposition/keyword guards remain a crude-substring backstop, not the firewall;
  the prose firewall above + human review at writeup time are primary (mirrors ADR-0015 §
  Consequences scope note).

**Does not commit.**
- This ADR does not fund or authorise any GPU retraining pass. The arm runs on frozen UNI2-h
  embeddings; option (b) end-to-end encoder fine-tune remains permanently excluded without a new
  ADR + new budget line.
- This ADR does not attach Track-B MIL to the Duke fusion trunk. That remains a separate-ADR +
  fresh-red-team requirement (per LOCK-1 and the companion-organ governance rule).
- This ADR does not move any LOCK, any gate target, any Duke headline number, or any ADR-0008
  architecture-forward framing.

---

## What stays in force (unchanged by this ADR)

- LOCK-1 (claim discipline, incl. cross-institution FORBIDDEN, kinetics FORBIDDEN,
  early-detection FORBIDDEN) — verbatim unchanged.
- LOCK-2 (leakage / evaluation integrity, patient-level splits, DeLong CI + shuffle on every
  number) — unchanged.
- LOCK-5 ($150 total / $25 incremental budget) — unchanged; Track-B encode spend ≈ $5, under cap.
- LOCK-6 (Track-A holds "the headline"; fusion claim) — unchanged.
- ADR-0008 G3 fusion architecture-forward framing and its honest-null body — unchanged.
- ADR-0015 arm-3 governance (same cohort, same-cohort co-citation rule) — unchanged; the
  MIL/arm-3 same-cohort co-citation rule in this ADR is the direct analogue.
- The Duke imaging-fusion null numbers (clinical 0.708; hierarchical #4 0.599 [0.495, 0.610];
  CI crosses 0.50) — unchanged, retained in the body of every Track-B-adjacent section.

---

## Ratification conditions

Both conditions are now met (Status: **ACCEPTED / RATIFIED**):

1. **All five unlock conditions satisfied on disk** — G5 PASS ✓; /red-team CONDITIONAL-PASS ✓
   (BOUND-F1 de-dup gate + BOUND-F3 multi-seed both satisfied); PI budget ≤$23 signed ✓;
   de-dup gate test-wire ✓ (`direct_overlap_n == 0`, re-confirmed 2026-08-12); GDC manifest ✓.
   **[MET as of 2026-08-12.]**
2. **PI ratification** — a dated `decisions.md` entry by Richard records acceptance.
   **[MET — PI sign-off recorded 2026-08-12 via checkpoint decision; see the `[ADR-0012-RATIFIED]`
   entry in `decisions.md`.]**

Both conditions being met and recorded, submission artifacts (Table 2, paper section) may now
report the Track-B MIL 0.9675 number with the full firewall citation.

---

## Status / sign-off

- **ACCEPTED / RATIFIED 2026-08-12.** Drafted during the model-integrity-remediation reconciliation
  session (plan item 13); ratified the same day on PI sign-off (plan item 14). All five unlock
  conditions verified met on disk; the PI sign-off line now exists (below).
- **/red-team: CONDITIONAL PASS (inherited from `[ADR-0012-PROPOSED]`, 2 bound fixes satisfied).**
  The two bound fixes (BOUND-F1 de-dup gate path → `dedup_report.json`/`direct_overlap_n`;
  BOUND-F3 multi-seed ≥3 spread → 0.9622 ± 0.0038) are both satisfied on disk. A fresh /red-team
  specifically on the MIL/arm-3 same-cohort co-citation risk was offered as optional belt-and-braces;
  the prose firewall above binds it.
- **PI sign-off: Richard ratified on 2026-08-12** (via the 2026-08-12 checkpoint decision). The
  `[ADR-0012-RATIFIED]` `decisions.md` entry is appended and the 0.9675 number is restored to
  Table2 (and, by the concurrent paper pass, to the manuscript) with the full firewall citation.
