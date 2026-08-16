
from __future__ import annotations

import re
import sys
from pathlib import Path

SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".ruff_cache",
    ".pytest_cache",
    ".mypy_cache",
    "archive",
}
ABSENT_EXEMPT = {"decisions.md", "P16_g0_governance_reconciliation.md", "consistency_lint.py"}

CANONICAL_PRESENT = [
    ("leakage 6-set", "ER/PR/HER2/Ki-67/Mol-Subtype/Oncotype"),
    ("baseline band", "0.74–0.84"),  
    ("deadline", "19 Aug 2026"),
    ("pretest gate G6", "G6"),
    ("gate G4 retained", "G4"),
    ("LOCK-6 subtype tuple", "AUROC 0.75 / 0.80 / 0.85"),
]

_RX_OLD_BAND = re.compile(r"0\.70[–-]0\.84")
_RX_STALE_PTR = re.compile(r"~115 locked decisions")
_RX_4SET = re.compile(
    r"(exclude\s+ER/PR/HER2/Ki-67(?!/)|ER/PR/HER2/Ki-67(?!/)\s+(are|excluded))", re.IGNORECASE
)


def _pretest_at_g5(line: str) -> bool:
    low = line.lower()
    return "g5/g6" in low and ("pretest" in low or "usability" in low) and "external" not in low


CANONICAL_ABSENT = [
    ("old baseline band 0.70-0.84", lambda s: bool(_RX_OLD_BAND.search(s))),
    ("pretest gate written as G5/G6 (canonical: G6)", _pretest_at_g5),
    ("stale constitution version pointer", lambda s: bool(_RX_STALE_PTR.search(s))),
    ("leakage 4-set as the exclusion list", lambda s: bool(_RX_4SET.search(s))),
]


def _skipped(path: Path) -> bool:
    return any(part in SKIP_DIRS for part in path.parts)


def _md_files(root: str) -> list[Path]:
    p = Path(root)
    files = [p] if p.is_file() else sorted(p.rglob("*.md"))
    return [f for f in files if not _skipped(f)]


def check(root: str = ".") -> list[str]:
    files = _md_files(root)
    texts = {f: f.read_text(encoding="utf-8", errors="replace") for f in files}
    problems: list[str] = []

    for label, needle in CANONICAL_PRESENT:
        if not any(needle in t for t in texts.values()):
            problems.append(
                f"MISSING canonical [{label}]: expected '{needle}' somewhere in the active tree"
            )

    for f, t in texts.items():
        if f.name in ABSENT_EXEMPT:
            continue
        for n, line in enumerate(t.splitlines(), 1):
            for label, pred in CANONICAL_ABSENT:
                if pred(line):
                    problems.append(f"DIVERGENT [{label}] {f}:{n}: {line.strip()}")
    return problems


def _selfcheck() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        canon = Path(d) / "decisions.md"
        canon.write_text(
            "ER/PR/HER2/Ki-67/Mol-Subtype/Oncotype\n"
            "0.74–0.84 band\n19 Aug 2026\nG6 pretest · G4 stretch\n"
            "AUROC 0.75 / 0.80 / 0.85\n"
        )
        bad = Path(d) / "drift.md"
        bad.write_text(
            "radiomics AUC 0.70–0.84 band\nclinical-usability pretest at G5/G6\n"
            "~115 locked decisions, v1 17 Jun\nER/PR/HER2/Ki-67 are NOT in inputs\n"
        )
        ok = Path(d) / "ok.md"
        ok.write_text(
            "exclude ER/PR/HER2/Ki-67/Mol-Subtype/Oncotype from inputs\n"
            "Phase 11 — External Validation & Clinician Pretest (G5/G6)\n"
        )
        problems = check(str(d))
        assert not any(p.startswith("MISSING") for p in problems), (
            f"selfcheck: false MISSING: {problems}"
        )
        for label in ("old baseline band", "pretest gate", "stale constitution", "leakage 4-set"):
            assert any(label in p for p in problems), (
                f"selfcheck: failed to catch {label}: {problems}"
            )
        assert not any(f"{ok}:" in p for p in problems), (
            f"selfcheck: false positive on ok.md: {problems}"
        )
        assert not any(f"{canon}:" in p for p in problems)
    print("consistency_lint self-check: PASS")


def main(argv: list[str]) -> int:
    if argv and argv[0] == "--selfcheck":
        _selfcheck()
        return 0
    root = argv[0] if argv else "."
    problems = check(root)
    if problems:
        print("CONSISTENCY VIOLATION — facts disagree with decisions.md:")
        for line in problems:
            print("  " + line)
        return 1
    print(f"consistency_lint: clean ({root})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
