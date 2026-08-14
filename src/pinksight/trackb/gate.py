"""Track B hard-gate guard (LOCK-6 / decisions.md [5.1]).

Track B (TCGA WSI + genomics) is stretch/gated upside — it MUST NOT run on real TCGA data until
Track A (MRI+clinical) has cleared external validation at G5. This module is the single mechanical
enforcement point: `TRACKB_GATE_OPEN` is False, and the real-data entrypoint calls
`assert_gate_open()`, which raises. The harness therefore cannot produce a real number until a
human flips the flag AFTER Track A clears G5.

Synthetic fixtures bypass this gate on purpose — they are PoC only, produce no patient number, and
are labelled "PoC · synthetic · not patient-matched · gate-closed" everywhere they surface.
"""

from __future__ import annotations

# Constitutional flag — RATIFIED OPEN 2026-07-11 by Richard (LAW L-1), decisions.md [TRACKB-GATE-OPEN].
# Track B real-data path is unlocked so real TCGA-BRCA data can flow through the harness (intra-TCGA
# subtype characterisation only, [5.2]). The ~84-patient Duke∩TCGA overlap MUST still be de-duplicated
# before any "external" claim, and this is NOT a cross-institution generalisation claim.
# Revisit-if Track A does NOT clear G5 → revert to False and park Track B ([5.1] / [4.8] / LOCK-6).
TRACKB_GATE_OPEN = True

_LOCK6_MSG = (
    "Track B is HARD-GATED (LOCK-6 / decisions.md [5.1]): real TCGA data is FORBIDDEN until "
    "Track A clears external validation at G5. To run on real data, a human must set "
    "TRACKB_GATE_OPEN=True in src/pinksight/trackb/gate.py AND log a dated decisions.md entry. "
    "Synthetic fixtures do not require the gate."
)


class TrackBGateClosedError(RuntimeError):
    """Raised when a real-TCGA entrypoint is invoked while the Track B gate is closed."""


def assert_gate_open() -> None:
    """Refuse to proceed unless a human has flipped the gate. Call this from every real-data path."""
    if not TRACKB_GATE_OPEN:
        raise TrackBGateClosedError(_LOCK6_MSG)
