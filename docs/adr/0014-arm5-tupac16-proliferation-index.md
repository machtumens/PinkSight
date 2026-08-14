# ADR-0014: Arm 5 — PAM50 proliferation-index characterisation from TUPAC16 WSI (mitotic-count task EXCLUDED)

Date: 2026-07-28   Status: **PROPOSED** (drafted in a CPU-only execute session; requires PI sign-off before any arm-5 training run, and a SEPARATE Gate-2 authorisation before any GPU pass)

> **This ADR records a NEW additive companion-organ arm; it does NOT propose changing any LOCKED invariant.**
> It fits the **existing** LOCK-1 ALLOWED framing ("Ki-67 stratification AT DIAGNOSIS", "characterisation")
> with no amendment — see §Claim-ledger compatibility. Per **LAW L-1** the operative go/no-go is a dated
> `decisions.md` entry, not this file. Arm 5 is a standalone companion organ: it is **NOT** fused into the
> Duke imaging encoder and **no arm-5 result moves any LOCK**.

## Context

Wave 3 of the novel-heads roadmap proposes a histology "proliferation-snapshot" organ. The natural public
dataset is **TUPAC16** (Tumor Proliferation Assessment Challenge 2016, ~500 training WSI, TCIA-linked,
public). TUPAC16 is attractive because it is one of very few public breast-histology datasets that ships a
**molecular proliferation label** alongside the slides.

The binding constraint is that TUPAC16 ships **two** tasks, and they are not equally admissible here:

| TUPAC16 task | What it is | Admissible under LOCK-1? |
|---|---|---|
| **Primary — mitotic-count / mitosis detection** | Count mitotic figures per high-power field; a per-unit-time cell-division readout | **NO — FORBIDDEN** |
| **Secondary — PAM50 proliferation-score regression** | Regress a continuous gene-expression index derived from the PAM50 proliferation gene module | **YES** |

The primary task is the one the challenge is famous for, so it is the one a future contributor (or a model
completing a prompt) will drift toward by default. That default is a claim-ledger violation, which is why
this decision is written down rather than left implicit in a config comment.

**Why mitotic count is forbidden and PAM50 proliferation score is not.** A mitotic count is a measurement of
how frequently cells are dividing — it is a rate-like, kinetics-adjacent quantity, and PinkSight's claim
ledger forbids kinetics framing outright (no growth rate, no doubling time, no tumour kinetics). The PAM50
proliferation score is a different object: it is a **snapshot index** computed from gene expression measured
at a single timepoint at diagnosis. It carries no temporal denominator and supports no statement about how
fast anything is changing. It is the same class of quantity as Ki-67, which the ledger explicitly ALLOWS
"at diagnosis".

The distinction is not cosmetic. Two models trained on the same slides differ in what they license a reader
to conclude: one invites "this tumour is growing at rate X" (forbidden, and unsupported by cross-sectional
data), the other supports "this tumour presents a high proliferation index at diagnosis" (allowed, and
exactly what the data can bear).

## Options considered

1. **Use TUPAC16's primary mitotic-count task** — pro: the largest, best-benchmarked label; strongest
   comparability to published work. con: **kinetics-adjacent, LOCK-1 FORBIDDEN**. Rejected outright; not a
   close call and not available via any reframing.
2. **Use TUPAC16's secondary PAM50 proliferation-score regression** — pro: molecular snapshot index at
   diagnosis, ledger-clean, directly analogous to the ALLOWED Ki-67 framing; continuous target suits a
   cheap Ridge floor gate. con: smaller/noisier label set than the mitosis annotations; weaker published
   comparison base. **Chosen.**
3. **Skip arm 5 entirely** — pro: zero ledger risk, zero spend. con: forfeits the one Wave-3 arm with a
   molecular proliferation target; the ledger risk in option 2 is fully mitigable by an explicit exclusion
   plus a programmatic wording guard. Rejected as over-cautious.
4. **Substitute TCGA-BRCA PAM50 proliferation scores for TUPAC16 slides** — pro: data already on disk
   (`data/pathology/`), no new download. con: not TUPAC16; changes the arm's identity. **Retained as the
   documented fallback** if TUPAC16 acquisition proves infeasible, and noted in the arm-5 config.

## Decision

Arm 5 targets the **PAM50 proliferation score** — a continuous gene-expression **proliferation index**
measured at diagnosis — regressed from TUPAC16 whole-slide images. Cheapest-first: a CPU texture-feature
Ridge floor gate produces Pearson r + bootstrap CI + shuffle sentinel across ≥3 seeds before any GPU
embedding pass is even proposed.

**TUPAC16's primary mitotic-count task is EXCLUDED by name.** See §Forbidden task below.

## Forbidden task

**The TUPAC16 mitotic-count / mitosis-detection task is FORBIDDEN for arm 5 and for every downstream
PinkSight artifact.** This is a hard exclusion, not a preference.

Named and rejected:

- **TUPAC16 auxiliary task 1 — mitosis detection** (locating mitotic figures in high-power fields).
- **TUPAC16 primary task — mitotic-count scoring** (the per-field mitotic-count-derived proliferation score).

Rejected because a mitotic count is a division-frequency readout: it is rate-like and therefore
kinetics-adjacent, and PinkSight's claim ledger forbids kinetics framing without exception. No arm-5
model may take a mitotic count as a target, an auxiliary target, an input feature, or a reported
comparison number. Reframing the mitotic count as "proliferation activity" does not make it admissible —
the objection is to the quantity, not to its name.

## Out of scope

Explicitly out of scope for arm 5, now and without a NEW ADR:

- **Any rate, kinetics, or temporal claim.** No growth rate, no doubling time, no tumour kinetics, no
  "how fast" statement of any kind. Arm 5 describes a single timepoint.
- **Mitotic count in any role** (target / auxiliary / feature / reported comparison) — see §Forbidden task.
- **Prognosis, survival, or treatment-response.** TUPAC16 supplies no outcome labels arm 5 may use, and
  arm 5 makes no outcome claim.
- **Early detection / screening / pre-detection framing.** Arm 5 operates on slides from tumours already
  diagnosed and resected.
- **Cross-institution generalisation.** Arm 5 is TUPAC16-internal. It is not compared to Duke, not
  validated on Duke, and licenses no transfer claim. Duke has no matched WSI.
- **Fusion into the Duke imaging encoder.** Arm 5 is a standalone companion organ. Attaching it to the
  Track-A trunk requires a NEW ADR and a fresh red-team pass.
- **Any GPU embedding pass** until Gate 2 is granted by the PI in the literal required form.

## Consequences

**Easier.** Arm 5 gains a ledger-clean molecular target with a cheap CPU floor gate, so the expensive
question ("do frozen foundation embeddings carry proliferation-index signal?") is only asked after a cheap
number justifies asking it. The exclusion is machine-checkable, so drift is caught by CI rather than by
review attention.

**Harder.** We forgo comparability with the large published TUPAC16 mitosis literature; arm-5 numbers will
not sit next to the challenge leaderboard. The PAM50 label set is smaller and noisier, so the floor gate
may return an honest null purely on power grounds — which is an acceptable outcome, reported as such.

**Commits / forbids downstream.**

- The wording guard `validate_arm_report_keywords(report_text, "arm5")` (in
  `scripts/novel_heads/wave1_eval_harness.py`) MUST be the final step before any arm-5 report is written
  to disk. It bans "mitotic rate", "growth rate", "doubling time", "kinetics", "proliferation rate" and
  requires "proliferation index" or "snapshot at diagnosis". It is not bypassable, including when the
  wording looks obviously fine.
- The arm-5 config records `target: pam50_proliferation_score` and carries the exclusion in its
  `ledger_guard`.
- No arm-5 GPU pass may run without this ADR ratified AND Gate 2 granted. Both, not either.

## Claim-ledger compatibility

LOCK-1 ALLOWED includes "Ki-67 stratification AT DIAGNOSIS" and "characterisation". The PAM50 proliferation
score is a molecular **proliferation index at diagnosis** — the same class of at-diagnosis snapshot
quantity as Ki-67, which is why it needs no ledger amendment. LOCK-1 FORBIDDEN includes kinetics framing,
which is why the mitotic-count task is excluded by name above rather than merely discouraged.

**No LOCK is moved by this ADR.** LOCK-1 through LOCK-6 are unchanged. Arm 5 is a standalone companion
organ; no arm-5 result — GREENLIGHT, KILL, or null — moves a gate target or changes a headline number.

## Required framing (verbatim, for any arm-5 artifact)

- SAY: "proliferation index", "PAM50 proliferation score", "snapshot at diagnosis", "Ki-67 surrogate index
  at diagnosis", "characterisation".
- NEVER SAY: "mitotic rate", "mitotic count", "mitosis rate", "growth rate", "doubling time", "tumour
  kinetics", "proliferation rate", "kinetics", "early detection", "pre-detection".

## Status / sign-off

- **PROPOSED 2026-07-28.** Drafted during a CPU-only Wave-3 execute session.
- **PI sign-off: PENDING.** No arm-5 training run may proceed until a dated `decisions.md` entry records
  acceptance (LAW L-1).
- **Gate 2 (GPU spend): NOT GRANTED.** Separate and additional to the sign-off above. Requires the PI to
  write the literal phrase "authorize Wave 3 GPU spend".
