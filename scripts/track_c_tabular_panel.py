"""Track C — public-benchmark tabular-risk ENSEMBLE companion-panel router (ADR-0010, Scope C).

ZERO gradient, inference-time routing ONLY. This is a THIN wrapper that selects one of Track C's
standalone, own-cohort demo pipelines; it trains nothing, imports no deep-learning framework, holds
no parameters, and never fuses anything.

ADR-0010 FRAMING GUARD (path-scoped LOCK-1 amendment — Scope C; verbatim in spirit):
  Track C = the Coimbra / BCSC / METABRIC public-benchmark tabular suite, admitted as an ENSEMBLE
  companion panel (ADR-0006 standalone-organ family). LOCK-1 is PATH-SCOPE-amended for Track-C
  artifacts ONLY (explore/tabular_risk/ + reports/**/track_c*): within Track-C artifacts the
  otherwise-FORBIDDEN framings early/pre-detection (Coimbra), incidence/screening-population risk
  (BCSC) and prognosis/survival (METABRIC) are EXEMPTED and reported per-cohort as independent public
  benchmarks. This is an ENSEMBLE (the four cohorts share ZERO patients / zero shared patients), so it
  is physically NOT cross-attention fusion — "fusion" / "integrated into the architecture" are BANNED
  words for Track C; it is NEVER fused into the Duke imaging encoder. Inference-time routing only, no
  cross-cohort gradient.
  What stays FORBIDDEN on ALL tracks (incl. Track C): growth-rate / tumour-kinetics / doubling-time;
  clinical-trial-grade FP/FN reduction; cross-institution generalisation / transfer. Track A (Duke)
  keeps un-amended LOCK-1 — no detection/incidence framing may bleed onto the Duke headline.

The four Track-C cohorts are patient-disjoint public benchmarks; routing them through one wrapper
returns a demo path, never a comparison and never a cross-cohort number.

Run:  .venv/bin/python scripts/track_c_tabular_panel.py --selfcheck
      .venv/bin/python scripts/track_c_tabular_panel.py --panel coimbra          # routing message, exit 0
      .venv/bin/python scripts/track_c_tabular_panel.py --panel coimbra --run    # exec the demo (sandbox venv)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# The untracked ADR-0010 sandbox that owns the standalone panel pipelines (gitignored; absent on a
# clean checkout). This wrapper routes to it but does not depend on it existing.
_SANDBOX = Path("explore/tabular_risk")
_SANDBOX_VENV = _SANDBOX / ".venv" / "bin" / "python"

# Panel -> its standalone demo FILENAME inside the sandbox. selfcheck asserts these NAMES are correct
# (not that they exist on disk — the sandbox is untracked). Each panel is a separately-trained,
# own-cohort ENSEMBLE member; there is no shared parameter across panels.
_PANEL_DEMOS: dict[str, str] = {
    "coimbra": "demo_coimbra.py",    # M1 — Breast Cancer Coimbra (UCI): metabolic/inflammatory host-state
    "bcsc": "demo_bcsc.py",          # M2 — BCSC: calibrated incidence risk + fairness subgroups
    "metabric": "demo_metabric.py",  # M3 — METABRIC: calibrated 5-yr overall-survival prognosis
}


def selfcheck() -> int:
    """Assert the panel->demo routing map is well-formed (correct names), with NO data and NO sandbox
    dependency. Imports no training framework; runs no panel; returns 0.
    """
    assert set(_PANEL_DEMOS) == {"coimbra", "bcsc", "metabric"}, (
        "Track-C panel set drifted from the ADR-0010 companion panel (Coimbra/BCSC/METABRIC)"
    )
    for panel, demo in _PANEL_DEMOS.items():
        assert demo == f"demo_{panel}.py", f"panel {panel!r} routes to a mis-named demo {demo!r}"
        assert demo.endswith(".py") and "/" not in demo, f"panel {panel!r} demo name malformed: {demo!r}"

    print(
        f"[track_c_tabular_panel] selfcheck OK — {len(_PANEL_DEMOS)} ADR-0010 ENSEMBLE companion "
        f"panels ({', '.join(sorted(_PANEL_DEMOS))}) route to correctly-named demos under {_SANDBOX}/; "
        f"ensemble NOT fusion; zero shared patients; inference-time routing only (no cross-cohort "
        f"gradient). No torch, no training, no data required."
    )
    return 0


def route_panel(panel: str, run: bool) -> int:
    """Resolve a panel to its sandbox demo.

    Sandbox ABSENT (clean checkout)  -> print a clear 'not present' message, exit 0 (never raises,
                                        never fabricates a result).
    Sandbox PRESENT, no --run         -> print a routing message naming the exact demo + reproduction
                                        command, exit 0.
    Sandbox PRESENT, --run            -> exec the demo through the sandbox's own venv, propagate its
                                        exit code.
    """
    demo_rel = _PANEL_DEMOS[panel]
    demo_path = _SANDBOX / demo_rel

    if not demo_path.exists():
        print(
            f"Track C sandbox ({_SANDBOX}/) not present — ADR-0010 companion panels live in an "
            f"untracked sandbox; see {_SANDBOX}/README.md to reproduce panel {panel!r} (demo {demo_rel})."
        )
        return 0

    interp = _SANDBOX_VENV if _SANDBOX_VENV.exists() else Path(sys.executable)
    if not run:
        print(
            f"Track C panel {panel!r} routes to {demo_path} — ADR-0010 ENSEMBLE companion "
            f"(own-cohort, standalone, zero shared patients; ensemble NOT fusion). "
            f"Reproduce: {interp} {demo_path}  (needs the sandbox venv + gitignored data; see "
            f"{_SANDBOX}/README.md). Re-run through this wrapper with --run to exec it."
        )
        return 0

    print(f"[track_c_tabular_panel] exec panel {panel!r}: {interp} {demo_path}")
    return subprocess.call([str(interp), str(demo_path)])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Track C public-benchmark tabular-risk ENSEMBLE companion-panel router (ADR-0010, routing only)"
    )
    ap.add_argument("--selfcheck", action="store_true", help="assert the panel->demo map is well-formed (no data)")
    ap.add_argument("--panel", choices=sorted(_PANEL_DEMOS), help="route to a Track-C companion panel demo")
    ap.add_argument(
        "--run",
        action="store_true",
        help="with --panel: exec the demo via the sandbox venv (default: print a routing message only)",
    )
    args = ap.parse_args(argv)

    if args.selfcheck:
        return selfcheck()
    if args.panel:
        return route_panel(args.panel, args.run)
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
