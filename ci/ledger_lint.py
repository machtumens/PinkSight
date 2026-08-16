
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

DETECTION_INCIDENCE = [
    r"pre[- ]?detection",
    r"early detection",
    r"incidence[- ]risk",
    r"has cancer",
]
ALWAYS_FORBIDDEN = [
    r"growth[- ]rate",
    r"tumou?r kinetics",
    r"doubling time",
    r"(false[- ]?positive|false[- ]?negative|FP/FN).{0,20}reduction",
    r"cross[- ]?institution.{0,25}generali[sz]",
]
FORBIDDEN = DETECTION_INCIDENCE + ALWAYS_FORBIDDEN
_PATTERN = re.compile("|".join(FORBIDDEN), re.IGNORECASE)
_PATTERN_TRACKC = re.compile("|".join(ALWAYS_FORBIDDEN), re.IGNORECASE)

JUXTAPOSITION_PHRASES = [
    r"imaging works",
    r"rescue.{0,20}duke",
    r"corroborates?\s+duke",
]
_JUXTA_PHRASE = re.compile("|".join(JUXTAPOSITION_PHRASES), re.IGNORECASE)
_TRACKB_NUM = re.compile(r"0\.9[67]\d{1,2}")  
_DUKE_ANCHOR = re.compile(r"0\.708")

_NEGATION_LINE = re.compile(r"❌|\b(?:not|never|no|forbid\w*)\b", re.IGNORECASE)
_FORBIDDEN_HEADING = re.compile(
    r"forbidden|does\s+not\s+license|do\s+not\s+license|never\s+say|not\s+license", re.IGNORECASE
)

LEDGER_FILES = {
    "decisions.md",
    "CHECKLISTS.md",
    "model_card.md",
    "MODEL_CARD_TEMPLATE.md",
    "P16_g0_governance_reconciliation.md",
    "CLAIM_LEDGER.md",
}
SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".ruff_cache",
    ".pytest_cache",
    ".mypy_cache",
    ".ipynb_checkpoints",
    "archive",
    "graphify-out",  
}
ALLOW_MARKER = "# allow-ledger"


def _skipped(path: Path) -> bool:
    return any(part in SKIP_DIRS for part in path.parts)


def _is_trackc_path(path: Path) -> bool:
    parts = [p.lower() for p in path.parts]
    for i in range(len(parts) - 1):
        if parts[i] == "explore" and parts[i + 1] == "tabular_risk":
            return True
    if any(part.startswith("track_c") for part in parts):
        return True
    return False


def _iter_lines(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix == ".ipynb":
        try:
            nb = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return []
        lines: list[str] = []
        for cell in nb.get("cells", []):
            src = cell.get("source", [])
            if isinstance(src, str):
                src = src.splitlines()
            for chunk in src:
                lines.extend(chunk.rstrip("\n").split("\n"))
        return lines
    return text.splitlines()


def _is_negation_line(line: str) -> bool:
    return bool(_NEGATION_LINE.search(line))


def _is_adr_path(path: Path) -> bool:
    parts = [p.lower() for p in path.parts]
    for i in range(len(parts) - 1):
        if parts[i] == "docs" and parts[i + 1] == "adr":
            return True
    return False


def scan_file(path: Path) -> list[tuple[int, str]]:
    if path.name in LEDGER_FILES or _is_adr_path(path):
        return []
    pattern = _PATTERN_TRACKC if _is_trackc_path(path) else _PATTERN
    hits: list[tuple[int, str]] = []
    in_forbidden_section = False
    for n, line in enumerate(_iter_lines(path), 1):
        stripped = line.lstrip()
        if stripped.startswith("#"):  
            in_forbidden_section = bool(_FORBIDDEN_HEADING.search(line))
        if ALLOW_MARKER in line:
            continue
        if in_forbidden_section:  
            continue
        if _is_negation_line(line):  
            continue
        if pattern.search(line):
            hits.append((n, line.strip()))
            continue
        if _JUXTA_PHRASE.search(line):
            hits.append((n, line.strip()))
            continue
        if _TRACKB_NUM.search(line) and _DUKE_ANCHOR.search(line):
            hits.append((n, line.strip()))
    return hits


def scan(paths: list[str]) -> list[str]:
    problems: list[str] = []
    for root in paths:
        p = Path(root)
        if p.is_file():
            files = [p]
        else:
            files = sorted(
                [*p.rglob("*.md"), *p.rglob("*.ipynb"), *p.rglob("*.tsx"), *p.rglob("*.html")]
            )
        for f in files:
            if _skipped(f):
                continue
            for n, line in scan_file(f):
                problems.append(f"{f}:{n}: {line}")
    return problems


def _selfcheck() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        bad = d / "bad.md"
        bad.write_text("We claim early detection in healthy tissue.\n")
        bad2 = d / "bad2.md"
        bad2.write_text("PinkSight delivers false-positive reduction for clinicians.\n")
        good = d / "good.md"
        good.write_text("We claim subtype characterisation at diagnosis.\n")
        marked = d / "marked.md"
        marked.write_text("Never say early detection.  # allow-ledger\n")
        assert scan([str(bad)]), "self-check FAIL: forbidden term not caught"
        assert scan([str(bad2)]), "self-check FAIL: new FP/FN-reduction pattern not caught"
        assert not scan([str(good)]), "self-check FAIL: clean line flagged"
        assert not scan([str(marked)]), "self-check FAIL: allow-ledger marker ignored"

        trackc_dir = d / "explore" / "tabular_risk" / "reports"
        trackc_dir.mkdir(parents=True)
        trackc_ok = trackc_dir / "note.md"
        trackc_ok.write_text(
            "Coimbra is a case-control detection model; BCSC is an incidence-risk screening model.\n"
        )
        assert not scan([str(trackc_ok)]), (
            "self-check FAIL: detection/incidence flagged inside the Track-C sandbox (should be exempt)"
        )
        trackc_report = d / "reports" / "TRACK_C_suite_summary.md"
        trackc_report.parent.mkdir(parents=True, exist_ok=True)
        trackc_report.write_text("The Track-C panel reports early detection on the Coimbra POC.\n")
        assert not scan([str(trackc_report)]), (
            "self-check FAIL: detection/incidence flagged in a Track-C report file (should be exempt)"
        )
        trackc_bad = trackc_dir / "bad_kinetics.md"
        trackc_bad.write_text("The tumour kinetics / growth rate improves with our doubling time model.\n")
        assert scan([str(trackc_bad)]), (
            "self-check FAIL: growth-rate/kinetics NOT caught inside Track-C (must stay banned everywhere)"
        )
        trackc_bad2 = trackc_dir / "bad_crossinst.md"
        trackc_bad2.write_text("This suite shows cross-institution generalisation across cohorts.\n")
        assert scan([str(trackc_bad2)]), (
            "self-check FAIL: cross-institution generalisation NOT caught inside Track-C (banned everywhere)"
        )
        track_a_bad = d / "docs" / "some_track_a_doc.md"
        track_a_bad.parent.mkdir(parents=True, exist_ok=True)
        track_a_bad.write_text("PinkSight performs early detection of breast cancer.\n")
        assert scan([str(track_a_bad)]), (
            "self-check FAIL: detection framing in a Track-A doc was NOT caught — path-scope leaked "
            "the exemption outside the Track-C sandbox"
        )

        juxta_bad = d / "juxta_bad.md"
        juxta_bad.write_text("Track-B MIL 0.9675 beats Duke clinical 0.708 — imaging works.\n")
        assert scan([str(juxta_bad)]), (
            "self-check FAIL: Track-B (0.9675) vs Duke (0.708) juxtaposition NOT caught"
        )
        juxta_bad2 = d / "juxta_bad2.md"
        juxta_bad2.write_text("arm-3 histology 0.9646 vs the Duke anchor 0.708 shows the ceiling.\n")
        assert scan([str(juxta_bad2)]), (
            "self-check FAIL: 0.9646-vs-0.708 numeric co-occurrence NOT caught"
        )
        firewalled = d / "firewalled.md"
        firewalled.write_text(
            "| Track B MIL | UNI2-h + ABMIL | 0.9675 [0.9479, 0.9871] | RATIFIED (ADR-0012) — "
            "same cohort, different encoder, NOT independent corroboration; never juxtaposed with Duke |\n"
        )
        assert not scan([str(firewalled)]), (
            "self-check FAIL: firewalled 0.9675 restore row flagged (negation-awareness broken)"
        )
        firewalled2 = d / "firewalled2.md"
        firewalled2.write_text(
            "Track-B MIL 0.9675 is never juxtaposed with the Duke clinical 0.708 anchor (coincidence).\n"
        )
        assert not scan([str(firewalled2)]), (
            "self-check FAIL: negated 0.9675/0.708 co-location flagged (negation-awareness broken)"
        )
        heading_ctx = d / "adr_like.md"
        heading_ctx.write_text(
            "## This ADR does NOT license\n"
            "- The sentence \"Track-B MIL 0.9675 vs Duke clinical 0.708\" is forbidden; imaging works too.\n"
        )
        assert not scan([str(heading_ctx)]), (
            "self-check FAIL: firewall bullet under a 'does NOT license' heading flagged (heading-context broken)"
        )
        nb_bad = d / "bad_notebook.ipynb"
        nb_bad.write_text(
            json.dumps(
                {
                    "cells": [
                        {"cell_type": "markdown", "source": ["PinkSight performs early detection.\n"]}
                    ],
                    "metadata": {},
                    "nbformat": 4,
                    "nbformat_minor": 5,
                }
            )
        )
        assert scan([str(nb_bad)]), "self-check FAIL: forbidden term in a .ipynb cell NOT caught"
        nb_good = d / "good_notebook.ipynb"
        nb_good.write_text(
            json.dumps(
                {
                    "cells": [
                        {"cell_type": "code", "source": ["doc['task'] = 'characterisation at diagnosis'\n"]}
                    ],
                    "metadata": {},
                    "nbformat": 4,
                    "nbformat_minor": 5,
                }
            )
        )
        assert not scan([str(nb_good)]), "self-check FAIL: clean .ipynb cell flagged"

        tsx_bad = d / "Bad.tsx"
        tsx_bad.write_text('<p>PinkSight enables early detection of tumours.</p>\n')
        assert scan([str(tsx_bad)]), "self-check FAIL: forbidden term in a .tsx text node NOT caught"
        tsx_ok = d / "Ok.tsx"
        tsx_ok.write_text('<p>Not early detection; characterisation at diagnosis only.</p>\n')
        assert not scan([str(tsx_ok)]), "self-check FAIL: negated .tsx disclaimer flagged"
        html_bad = d / "bad.html"
        html_bad.write_text("<li>Growth-rate / kinetics modelling</li>\n")
        assert scan([str(html_bad)]), "self-check FAIL: forbidden term in .html body NOT caught"
        html_marked = d / "marked.html"
        html_marked.write_text("<li>Early / pre-detection</li>  <!-- # allow-ledger -->\n")
        assert not scan([str(html_marked)]), "self-check FAIL: .html allow-ledger marker ignored"

        adr_defining = d / "docs" / "adr" / "0099-example.md"
        adr_defining.parent.mkdir(parents=True, exist_ok=True)
        adr_defining.write_text(
            "This ADR bans early detection, growth-rate and cross-institution generalisation framing.\n"
        )
        assert not scan([str(adr_defining)]), (
            "self-check FAIL: docs/adr/ firewall line flagged (directory exemption broken)"
        )
        non_adr = d / "docs" / "notes" / "adr_summary.md"
        non_adr.parent.mkdir(parents=True, exist_ok=True)
        non_adr.write_text("PinkSight performs early detection of breast cancer.\n")
        assert scan([str(non_adr)]), (
            "self-check FAIL: forbidden assertion outside docs/adr/ was NOT caught — the ADR "
            "directory exemption leaked beyond docs/adr/"
        )
    print("ledger_lint self-check: PASS")


def main(argv: list[str]) -> int:
    if argv and argv[0] == "--selfcheck":
        _selfcheck()
        return 0
    targets = argv or ["."]
    problems = scan(targets)
    if problems:
        print("LEDGER VIOLATION — forbidden framing found:")
        for line in problems:
            print("  " + line)
        print(
            "Say 'characterisation/localisation', not 'early detection'; "
            "'Ki-67/aggressiveness', not 'growth rate'. Never juxtapose a Track-B histology AUROC "
            "with the Duke 0.708 anchor. "
            "If a line legitimately quotes a banned term (cited title / digest), "
            "append '# allow-ledger' to THAT line."
        )
        return 1
    print(f"ledger_lint: clean ({', '.join(targets)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
