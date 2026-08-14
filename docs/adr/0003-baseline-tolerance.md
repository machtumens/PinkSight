# ADR-0003: Baseline replication tolerance — two checks, not one
Date: 2026-06-20   Status: proposed (raised by P16 governance reconciliation — awaiting human ratification)

## Context
The corpus disagrees on the G1 baseline-replication tolerance, and the reconciliation pass
(P16) is forbidden from silently picking a value. Two numbers are in play:

- `decisions.md:54` (LOCK-4, G1 row): reproduced AUC in the **0.74–0.84 band (±0.005)**.
- `execution-roadmap.md:168` / `prd-charter.md:144`: radiomics AUC **within ±0.05 of published**.

These are almost certainly **two different checks**, not a contradiction:
- **±0.05 of published** = *literature-match* tolerance — "did we land near the number the paper
  reported?" A loose, cross-implementation sanity bar.
- **±0.005** = *self-reproducibility* tolerance — "does the teammate's re-run of OUR pipeline land
  on OUR number?" A tight, same-code determinism bar (it matches the O3 KR3.3 repro rule in
  `prd-charter.md:30`: "reproduced by teammate within ±0.005").

The band `0.74–0.84` itself is settled (LOCK-4) and was reconciled everywhere by P16; only the
**tolerance semantics** are open.

## Options considered
1. **Split into two named checks (recommended)** — at G1 acceptance: "match-published ≤ ±0.05 of
   the cited result"; separately, "re-run reproducibility ≤ ±0.005 vs our own logged number."
   Pro: both numbers survive, each gets a clear owner and gate. Con: two checks to track.
2. **Keep only ±0.005** — treat ±0.05 as informal. Pro: one number. Con: loses the
   literature-match sanity bar; ±0.005 across different published implementations is unrealistic.
3. **Keep only ±0.05** — Pro: simplest acceptance bar. Con: drops the reproducibility guarantee
   that O3/KR3.3 already locks; weakens the integrity story judges will probe.

## Decision
<PENDING HUMAN RULING — recommended: Option 1, two checks.>

## Consequences
- (If Option 1) G1 acceptance contract gains two explicit lines; `consistency_lint` can later
  assert both tolerances coexist without one overwriting the other.
- Resolves the apparent ±0.005-vs-±0.05 conflict without weakening either integrity guarantee.
