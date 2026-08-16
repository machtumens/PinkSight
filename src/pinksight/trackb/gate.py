
from __future__ import annotations

TRACKB_GATE_OPEN = True

_LOCK6_MSG = (
    "Track B is HARD-GATED (LOCK-6 / decisions.md [5.1]): real TCGA data is FORBIDDEN until "
    "Track A clears external validation at G5. To run on real data, a human must set "
    "TRACKB_GATE_OPEN=True in src/pinksight/trackb/gate.py AND log a dated decisions.md entry. "
    "Synthetic fixtures do not require the gate."
)


class TrackBGateClosedError(RuntimeError):
    pass


def assert_gate_open() -> None:
    if not TRACKB_GATE_OPEN:
        raise TrackBGateClosedError(_LOCK6_MSG)
