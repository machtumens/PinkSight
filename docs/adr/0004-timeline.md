# ADR-0004: Project timeline — inception vs build window
Date: 2026-06-20   Status: proposed (raised by P16 governance reconciliation — awaiting human ratification)

## Context
The corpus carries two mutually inconsistent start dates, and the reconciliation pass (P16) is
forbidden from silently picking one:

- `reading-list.md` (pre-pivot glance) + `execution-roadmap.md:16`: **7 Apr – 19 Aug 2026**.
- `prd-charter.md:14`: **17 Jun → 19 Aug 2026 (~9 weeks)**.

These cannot both describe the same span: 7 Apr → 19 Aug is ≈19 weeks, not 9; only
17 Jun → 19 Aug ≈ 9 weeks. So "7 April … ≈9 weeks" is internally impossible. The end date
(**19 Aug 2026**) and code-freeze (**16 Aug**) are settled (LOCK-7) and not in question.

The most likely reality: 7 April is the **proposal / inception** date; 17 June is when the
**build window** actually opened. The "~9 weeks" figure belongs to the build window.

## Options considered
1. **Two-phase framing (recommended)** — "inception 7 Apr 2026 (proposal) · build window
   17 Jun → 19 Aug 2026 (~9 weeks), freeze 16 Aug." Pro: keeps both real dates, fixes the
   arithmetic, matches the ~9-week figure everyone cites. Con: slightly more verbose.
2. **Build window only (17 Jun → 19 Aug)** — drop 7 Apr. Pro: simplest. Con: erases the proposal
   milestone, which may matter for the OPSI submission narrative.
3. **Full span (7 Apr → 19 Aug, ~19 weeks)** — drop "~9 weeks." Pro: one continuous range.
   Con: contradicts the universally-cited ~9-week runway and the prd-charter.

## Decision
<PENDING HUMAN RULING — recommended: Option 1, two-phase framing.>

## Consequences
- (If Option 1) `reading-list.md` and `execution-roadmap.md` adopt the two-phase wording (P16 has
  already applied it provisionally to the reading-list glance block, tagged "pending ratification").
- Removes the only internally-impossible date claim in the corpus.
