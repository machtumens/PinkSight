#!/usr/bin/env python3
"""ledger_lint — fail CI if forbidden CLAIM-LEDGER framings leak into the docs.

The claim ledger drifts back constantly because the forbidden wording is the *natural*
way to describe the project. This guard greps prose for it and exits non-zero.

Usage: python ci/ledger_lint.py [path ...]   (defaults to the whole repo)
Scans `*.md`, `*.ipynb` (notebook cell source), `*.tsx`, and `*.html` (desktop-app UI surface)
across the active tree. Skips build/vendor/history dirs and the `archive/` tree (frozen history,
not active claims). A small set of ledger-DEFINING files (which quote the banned terms precisely in
order to FORBID them) are exempt by name; the `docs/adr/` decision-record tree is exempt by
directory for the same reason (each ADR's own firewall / NOT-FOR section quotes the bans in order
to forbid them — never to assert them — the same DEFINING role as the by-name exempt files).

The `.tsx`/`.html` surface is scanned as plain-text lines (same crude line-based path as `.md`, per
Phase-4 ID-1 — no JSX/HTML AST extractor). HTML has no markdown `#` headings, so the heading-context
suppression does not fire there; a legitimately DEFINING forbidden-term line in the UI (e.g. the
desktop mockup's "constitution" FORBIDDEN legend) is exempted with a per-line `# allow-ledger` marker
(embed it as an HTML comment `<!-- # allow-ledger -->`), never by weakening the regex.

Three ways a legitimate quote of a banned term is EXEMPTED (so disclaimers/firewalls do not
fail the build):
  1. a per-line `# allow-ledger` marker (per-LINE on purpose — a real new assertion on an
     unmarked line still fails);
  2. negation-awareness — a line that DISCLAIMS or FORBIDS a term (contains not / never / no /
     forbid / ❌) is a disclaimer, not a claim, and is skipped;
  3. heading-context — any line under a markdown heading that matches FORBIDDEN / "does NOT
     license" / "NEVER SAY" (e.g. an ADR's own firewall section) is skipped.

Negation-awareness + heading-context are deliberately CRUDE: a real new assertion that happens
to contain "no"/"not" will also be skipped. The prose firewall + human review remain primary;
this guard is a backstop (mirrors ADR-0015's own admission about `validate_arm_report_keywords`).

C3 juxtaposition tier (ADR-0012 / ADR-0015 firewall): flags a Track-B histology AUROC placed
beside the Duke clinical anchor (0.708) — a cross-cohort ceiling comparison the different-cohort +
different-modality design cannot support. Suppressed on negation/firewall lines, so the sentences
that FORBID the juxtaposition ("never juxtaposed with Duke", "not compared") do not trip it.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# ponytail: flat regex list, not a config file — these change ~never.
#
# Two tiers (ADR-0010 Track-C amendment):
#   DETECTION_INCIDENCE — detection/incidence/case-control framing. BANNED repo-wide by default,
#     but PATH-SCOPED-EXEMPT inside the Track-C sandbox (explore/tabular_risk/ + Track-C reports),
#     where ADR-0010 walls detection/incidence/prognosis framing to public benchmarks. Still BANNED
#     everywhere in Track-A artifacts + top-level docs.
#   ALWAYS_FORBIDDEN — growth-rate/kinetics + clinical-grade FP/FN reduction + cross-institution
#     generalisation. BANNED EVERYWHERE including Track-C; ADR-0010 does NOT touch these.
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
# Backwards-compat: the full ban list (used by default / outside the Track-C sandbox).
FORBIDDEN = DETECTION_INCIDENCE + ALWAYS_FORBIDDEN
_PATTERN = re.compile("|".join(FORBIDDEN), re.IGNORECASE)
# Track-C sandbox pattern: ONLY the always-forbidden tier bites (detection/incidence is exempt here).
_PATTERN_TRACKC = re.compile("|".join(ALWAYS_FORBIDDEN), re.IGNORECASE)

# --- C3 juxtaposition tier (ADR-0012 / ADR-0015 firewall) ---------------------------------------
# A Track-B histology AUROC (the 0.96xx / 0.97xx band — 0.9646 TITAN, 0.9675 UNI2-h, 0.9679 LOSO)
# placed on the SAME line as the Duke clinical anchor (0.708) implies a cross-cohort ceiling
# comparison the different-cohort + different-modality design cannot support (ADR-0012 firewall).
# Two catches, both SUPPRESSED on a negation/firewall line (see _is_negation_line / heading-context):
#   (1) literal NEVER-SAY comparison phrases (imaging works / rescue Duke / corroborates Duke);
#   (2) numeric co-occurrence — a Track-B 0.9[67]xx AUROC AND 0.708 on the same line.
# Crude substring/co-occurrence backstop, NOT a semantic firewall; prose firewall + human review
# at writeup time remain primary. Bites everywhere (not Track-C-scoped — orthogonal to detection).
# NOTE: bare "cross-cohort" is deliberately NOT a phrase here — it appears in far too much
# legitimate descriptive/disclaimer prose ("no cross-cohort claim", "cross-cohort comparison is
# banned"); the dangerous CLAIM is already caught by the ALWAYS_FORBIDDEN cross-institution tier
# and by the numeric co-occurrence below.
JUXTAPOSITION_PHRASES = [
    r"imaging works",
    r"rescue.{0,20}duke",
    r"corroborates?\s+duke",
]
_JUXTA_PHRASE = re.compile("|".join(JUXTAPOSITION_PHRASES), re.IGNORECASE)
_TRACKB_NUM = re.compile(r"0\.9[67]\d{1,2}")  # 0.96xx / 0.97xx histology-AUROC band
_DUKE_ANCHOR = re.compile(r"0\.708")

# --- negation-awareness / firewall-heading context ----------------------------------------------
# A disclaimer/firewall line quotes a banned term in order to FORBID or disclaim it — not to
# assert it. These clear the ~448 negation-checklist / ADR-firewall / "NOT early detection"
# false-positives (see ledger_lint_triage_12-08-26.md) WITHOUT a per-line `# allow-ledger` marker.
_NEGATION_LINE = re.compile(r"❌|\b(?:not|never|no|forbid\w*)\b", re.IGNORECASE)
# A markdown heading whose text forbids/disclaims — everything under it is firewall prose.
_FORBIDDEN_HEADING = re.compile(
    r"forbidden|does\s+not\s+license|do\s+not\s+license|never\s+say|not\s+license", re.IGNORECASE
)

# Files whose whole job is to QUOTE the banned terms in order to forbid them.
# model_card.md enumerates the bans in its NOT-FOR disclaimer; the P16 governance
# prompt is the reconciliation spec that names the whole DO-NOT list throughout.
LEDGER_FILES = {
    "decisions.md",
    "CHECKLISTS.md",
    "model_card.md",
    "MODEL_CARD_TEMPLATE.md",
    "P16_g0_governance_reconciliation.md",
    # CLAIM_LEDGER.md DEFINES the forbidden terms in its own `**FORBIDDEN:**` bullets (a ledger-
    # defining file) — quoting the bans to forbid them, never asserting them.
    "CLAIM_LEDGER.md",
}
# Build/vendor/history dirs and the frozen archive — never part of the active claim surface.
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
    "graphify-out",  # derived knowledge-graph index (L-4) — never an active claim surface
}
ALLOW_MARKER = "# allow-ledger"


def _skipped(path: Path) -> bool:
    return any(part in SKIP_DIRS for part in path.parts)


def _is_trackc_path(path: Path) -> bool:
    """ADR-0010 Track-C sandbox: detection/incidence framing is ALLOWED here (walled to public
    benchmarks), while the always-forbidden tier still bites. Two scoped locations:
      1. anything under `explore/tabular_risk/`
      2. a Track-C report file — a path component OR filename starting `track_c` / `TRACK_C`
         (case-insensitive), typically under reports/.
    Everything else (Track-A artifacts, top-level docs) uses the full ban list.
    """
    parts = [p.lower() for p in path.parts]
    # 1. explore/tabular_risk/** — the exploration sandbox.
    for i in range(len(parts) - 1):
        if parts[i] == "explore" and parts[i + 1] == "tabular_risk":
            return True
    # 2. Track-C report path: any dir component or the filename starts with "track_c".
    if any(part.startswith("track_c") for part in parts):
        return True
    return False


def _iter_lines(path: Path) -> list[str]:
    """Return the text lines to scan. For `.md`, the file lines verbatim (line numbers match the
    file). For `.ipynb`, the concatenated source of every cell (outputs/metadata are NOT scanned,
    so scraped/exec output cannot false-positive); the returned index is a logical line counter."""
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
    """A line that DISCLAIMS or FORBIDS a term (not / never / no / forbid / ❌) is a disclaimer,
    not a claim-ledger leak. Deliberately crude — see module docstring."""
    return bool(_NEGATION_LINE.search(line))


def _is_adr_path(path: Path) -> bool:
    """`docs/adr/*` are decision-RECORD files. Each ADR's own firewall / NOT-FOR section quotes the
    banned terms precisely in order to FORBID or disclaim them — the same DEFINING role as
    CLAIM_LEDGER.md (whole-file exempt in LEDGER_FILES), never an ASSERTION of the
    forbidden framing. Exempt by directory (not per-line) so a future ADR's firewall section does
    not need a `# allow-ledger` marker on every quoted ban. The prose firewall + human review
    remain primary (see module docstring): an ADR that made a genuine forbidden CLAIM would be
    caught there, exactly as for the other LEDGER_FILES. Scoped to `docs/adr/` only — a forbidden
    assertion anywhere else (top-level docs, Track-A artifacts) still fails, proven by _selfcheck.
    """
    parts = [p.lower() for p in path.parts]
    for i in range(len(parts) - 1):
        if parts[i] == "docs" and parts[i + 1] == "adr":
            return True
    return False


def scan_file(path: Path) -> list[tuple[int, str]]:
    if path.name in LEDGER_FILES or _is_adr_path(path):
        return []
    # Track-C sandbox → only the always-forbidden tier applies (detection/incidence is exempt).
    pattern = _PATTERN_TRACKC if _is_trackc_path(path) else _PATTERN
    hits: list[tuple[int, str]] = []
    in_forbidden_section = False
    for n, line in enumerate(_iter_lines(path), 1):
        stripped = line.lstrip()
        if stripped.startswith("#"):  # markdown heading → (re)set firewall-section context
            in_forbidden_section = bool(_FORBIDDEN_HEADING.search(line))
        if ALLOW_MARKER in line:
            continue
        if in_forbidden_section:  # heading-context: under a FORBIDDEN / "does NOT license" heading
            continue
        if _is_negation_line(line):  # negation-awareness: a disclaimer, not an assertion
            continue
        # tier 1+2 — detection/incidence + always-forbidden framings
        if pattern.search(line):
            hits.append((n, line.strip()))
            continue
        # C3 juxtaposition tier — literal NEVER-SAY phrases OR a Track-B AUROC beside Duke 0.708
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
            # Phase-4: `.tsx`/`.html` added — the desktop-app UI is a public claim surface too.
            # `.tsx` live only under app/src/, `.html` only under app/design/ + app/index.html;
            # node_modules/build dirs are excluded by SKIP_DIRS in the loop below.
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

        # --- ADR-0010 Track-C path-scoping ---------------------------------------------------------
        # (a) detection/incidence framing is EXEMPT inside explore/tabular_risk/**.
        trackc_dir = d / "explore" / "tabular_risk" / "reports"
        trackc_dir.mkdir(parents=True)
        trackc_ok = trackc_dir / "note.md"
        trackc_ok.write_text(
            "Coimbra is a case-control detection model; BCSC is an incidence-risk screening model.\n"
        )
        assert not scan([str(trackc_ok)]), (
            "self-check FAIL: detection/incidence flagged inside the Track-C sandbox (should be exempt)"
        )
        # (b) a Track-C report file (filename starts track_c) is EXEMPT too.
        trackc_report = d / "reports" / "TRACK_C_suite_summary.md"
        trackc_report.parent.mkdir(parents=True, exist_ok=True)
        trackc_report.write_text("The Track-C panel reports early detection on the Coimbra POC.\n")
        assert not scan([str(trackc_report)]), (
            "self-check FAIL: detection/incidence flagged in a Track-C report file (should be exempt)"
        )
        # (c) the ALWAYS-forbidden tier still BITES inside the Track-C sandbox (growth-rate,
        #     FP/FN reduction, cross-institution) — ADR-0010 does not touch those.
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
        # (d) THE KEY GUARD — detection framing in a TRACK-A artifact still FAILS (path-scope did not
        #     weaken the repo-wide ban outside the sandbox). A top-level docs file is Track-A scope.
        track_a_bad = d / "docs" / "some_track_a_doc.md"
        track_a_bad.parent.mkdir(parents=True, exist_ok=True)
        track_a_bad.write_text("PinkSight performs early detection of breast cancer.\n")
        assert scan([str(track_a_bad)]), (
            "self-check FAIL: detection framing in a Track-A doc was NOT caught — path-scope leaked "
            "the exemption outside the Track-C sandbox"
        )

        # --- C3 juxtaposition tier (ADR-0012 / ADR-0015 firewall) ----------------------------------
        # (e) a literal Track-B-histology-vs-Duke juxtaposition IS caught (numeric co-occurrence +
        #     the "imaging works" phrase) — this is the case the guard exists to catch.
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
        # (f) the FIREWALLED restore row is NOT flagged — a Track-B number that carries the
        #     "never juxtaposed" / "NOT independent" disclaimer must pass (negation-awareness).
        firewalled = d / "firewalled.md"
        firewalled.write_text(
            "| Track B MIL | UNI2-h + ABMIL | 0.9675 [0.9479, 0.9871] | RATIFIED (ADR-0012) — "
            "same cohort, different encoder, NOT independent corroboration; never juxtaposed with Duke |\n"
        )
        assert not scan([str(firewalled)]), (
            "self-check FAIL: firewalled 0.9675 restore row flagged (negation-awareness broken)"
        )
        # (g) a firewall line that co-locates BOTH numbers but negates the comparison passes.
        firewalled2 = d / "firewalled2.md"
        firewalled2.write_text(
            "Track-B MIL 0.9675 is never juxtaposed with the Duke clinical 0.708 anchor (coincidence).\n"
        )
        assert not scan([str(firewalled2)]), (
            "self-check FAIL: negated 0.9675/0.708 co-location flagged (negation-awareness broken)"
        )
        # (h) heading-context — an ADR firewall bullet under a "does NOT license" heading passes,
        #     even though it quotes the forbidden juxtaposition verbatim.
        heading_ctx = d / "adr_like.md"
        heading_ctx.write_text(
            "## This ADR does NOT license\n"
            "- The sentence \"Track-B MIL 0.9675 vs Duke clinical 0.708\" is forbidden; imaging works too.\n"
        )
        assert not scan([str(heading_ctx)]), (
            "self-check FAIL: firewall bullet under a 'does NOT license' heading flagged (heading-context broken)"
        )
        # (i) `.ipynb` notebook cell source IS scanned (E3).
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

        # --- Phase-4 `.tsx`/`.html` UI surface -----------------------------------------------------
        # (j) a forbidden term in a `.tsx` JSX text node IS scanned (plain-text line path).
        tsx_bad = d / "Bad.tsx"
        tsx_bad.write_text('<p>PinkSight enables early detection of tumours.</p>\n')
        assert scan([str(tsx_bad)]), "self-check FAIL: forbidden term in a .tsx text node NOT caught"
        # (k) a `.tsx` disclaimer with a same-line negation passes (negation-awareness, as for .md).
        tsx_ok = d / "Ok.tsx"
        tsx_ok.write_text('<p>Not early detection; characterisation at diagnosis only.</p>\n')
        assert not scan([str(tsx_ok)]), "self-check FAIL: negated .tsx disclaimer flagged"
        # (l) a forbidden term in an `.html` body IS scanned.
        html_bad = d / "bad.html"
        html_bad.write_text("<li>Growth-rate / kinetics modelling</li>\n")
        assert scan([str(html_bad)]), "self-check FAIL: forbidden term in .html body NOT caught"
        # (m) an `.html` DEFINING line (the UI 'constitution' FORBIDDEN legend) is exempted by an
        #     inline `# allow-ledger` marker embedded as an HTML comment — HTML has no `#` heading, so
        #     this is the intended exemption path (Phase-4 ID-1), never a regex weakening.
        html_marked = d / "marked.html"
        html_marked.write_text("<li>Early / pre-detection</li>  <!-- # allow-ledger -->\n")
        assert not scan([str(html_marked)]), "self-check FAIL: .html allow-ledger marker ignored"

        # --- Phase-5: docs/adr/ directory exemption (E3) -------------------------------------------
        # (n) an ADR firewall/NOT-FOR line that quotes banned terms in order to FORBID them is exempt
        #     by DIRECTORY — no per-line marker needed. ADRs are decision records that DEFINE the bans.
        adr_defining = d / "docs" / "adr" / "0099-example.md"
        adr_defining.parent.mkdir(parents=True, exist_ok=True)
        adr_defining.write_text(
            "This ADR bans early detection, growth-rate and cross-institution generalisation framing.\n"
        )
        assert not scan([str(adr_defining)]), (
            "self-check FAIL: docs/adr/ firewall line flagged (directory exemption broken)"
        )
        # (o) THE KEY GUARD — the ADR exemption must NOT leak outside docs/adr/. A forbidden
        #     ASSERTION in an ordinary docs file still FAILS (mirrors the Track-C key guard (d)).
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
