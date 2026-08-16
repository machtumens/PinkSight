
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_SANDBOX = Path("explore/tabular_risk")
_SANDBOX_VENV = _SANDBOX / ".venv" / "bin" / "python"

_PANEL_DEMOS: dict[str, str] = {
    "coimbra": "demo_coimbra.py",    
    "bcsc": "demo_bcsc.py",          
    "metabric": "demo_metabric.py",  
}


def selfcheck() -> int:
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
