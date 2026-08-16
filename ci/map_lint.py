
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

MAP_DIR = Path("docs/map")
MASTER = "MASTER.md"

SRC_RE = re.compile(r"\(src:\s*([^)]*)\)")
PATH_RE = re.compile(r"^([\w./_-]+\.[\w]+|[\w./_-]+/)(?::(\d+)(?:[-–](\d+))?)?$")
LINK_RE = re.compile(r"\]\((?!https?:)([^)#]+?\.md)(?:#[^)]*)?\)")
COMMIT_RE = re.compile(r"Built from commit `([0-9a-f]{7,40})`")


def _clean(part: str) -> str:
    part = part.strip().strip("`").strip()
    for sep in (" —", " -", " §", " ("):
        if sep in part:
            part = part.split(sep, 1)[0]
    return part.strip().strip("`").strip()


def _line_count(path: Path) -> int:
    try:
        with path.open("rb") as fh:
            return sum(1 for _ in fh)
    except OSError:
        return 0


def check_citations(root: Path) -> list[str]:
    problems: list[str] = []
    for md in sorted((root / MAP_DIR).glob("*.md")):
        for lineno, line in enumerate(md.read_text(encoding="utf-8").splitlines(), 1):
            for group in SRC_RE.findall(line):
                for raw in re.split(r"[,;]", group):
                    part = _clean(raw)
                    m = PATH_RE.match(part)
                    if not m:
                        continue  
                    target, start, end = m.group(1), m.group(2), m.group(3)
                    tp = root / target
                    if not tp.exists():
                        problems.append(f"{md}:{lineno}: BROKEN cite — no such path: {target}")
                        continue
                    if start and tp.is_file():
                        want = int(end or start)
                        have = _line_count(tp)
                        if want > have:
                            problems.append(
                                f"{md}:{lineno}: BROKEN cite — {target}:{want} "
                                f"but file has {have} lines"
                            )
            for link in LINK_RE.findall(line):
                if not (md.parent / link).exists():
                    problems.append(f"{md}:{lineno}: BROKEN link — {link}")
    return problems


def cited_files(root: Path) -> set[str]:
    out: set[str] = set()
    for md in (root / MAP_DIR).glob("*.md"):
        for group in SRC_RE.findall(md.read_text(encoding="utf-8")):
            for raw in re.split(r"[,;]", group):
                m = PATH_RE.match(_clean(raw))
                if m and not m.group(1).endswith("/"):
                    out.add(m.group(1))
    return out


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, timeout=20, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None


def staleness_verdict(root: Path) -> tuple[str | None, list[str] | None]:
    master = root / MAP_DIR / MASTER
    if not master.exists():
        return None, None
    m = COMMIT_RE.search(master.read_text(encoding="utf-8"))
    if not m:
        return None, None
    commit = m.group(1)
    if (anc := _git(root, "merge-base", "--is-ancestor", commit, "HEAD")) is None or anc.returncode:
        return commit, None
    if (changed := _git(root, "diff", "--name-only", commit)) is None or changed.returncode:
        return commit, None
    return commit, sorted(cited_files(root) & set(changed.stdout.split()))


def _selfcheck() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / MAP_DIR).mkdir(parents=True)
        (root / "real.md").write_text("a\nb\nc\n")
        (root / "CLAIM_LEDGER.md").write_text("x\n")  
        (root / MAP_DIR / "e1-x.md").write_text("ok\n")
        (root / MAP_DIR / MASTER).write_text(
            "good (src: real.md:2) and range (src: `real.md:1-3` — note)\n"
            "prose-only (src: CLAIM_LEDGER.md §Targets) and ellipsis (src: …)\n"
            "link [x](e1-x.md)\n"
            "past-eof (src: real.md:99)\n"
            "no-file (src: ghost.md:1)\n"
            "dead [y](e9-nope.md)\n"
        )
        problems = check_citations(root)
        assert any("real.md:99" in p for p in problems), f"missed past-EOF: {problems}"
        assert any("ghost.md" in p for p in problems), f"missed missing file: {problems}"
        assert any("e9-nope.md" in p for p in problems), f"missed dead link: {problems}"
        assert not any(":1:" in p for p in problems), f"false positive on line 1: {problems}"
        assert not any("CLAIM_LEDGER.md" in p for p in problems), f"flagged a §section ref: {problems}"
        assert len(problems) == 3, f"expected exactly 3 problems, got: {problems}"
        (root / MAP_DIR / MASTER).write_text("**Built from commit `deadbee` · 2026-01-01.**\n")
        assert staleness_verdict(root) == ("deadbee", None), "unverifiable stamp must be None"
        (root / MAP_DIR / MASTER).write_text("no stamp here\n")
        assert staleness_verdict(root) == (None, None), "missing stamp must be None"
    print("map_lint self-check: PASS")


def main(argv: list[str]) -> int:
    if argv and argv[0] == "--selfcheck":
        _selfcheck()
        return 0
    strict = "--strict" in argv
    root = Path(next((a for a in argv if not a.startswith("--")), "."))
    if not (root / MAP_DIR).is_dir():
        print(f"map_lint: no {MAP_DIR}/ under {root} — nothing to check")
        return 0

    problems = check_citations(root)
    if problems:
        print("MAP BROKEN — docs/map/ cites something that does not exist:")
        for p in problems:
            print("  " + p)
        return 1

    commit, drifted = staleness_verdict(root)
    n = len(cited_files(root))
    if drifted is None:
        why = (
            f"stamped commit {commit} is not an ancestor of HEAD — orphaned by a rebase/amend, "
            "or this is not a git checkout"
            if commit
            else f"{MAP_DIR}/{MASTER} has no `Built from commit \\`<sha>\\`` header"
        )
        print(f"MAP UNVERIFIABLE — cannot check freshness: {why}.")
        print("  → drift detection is OFF until the stamp names a real ancestor of HEAD.")
        print(f"    Fix: re-verify the citations, then re-stamp. See {MAP_DIR}/README.md §Refresh.")
        return 1
    if drifted:
        print(f"MAP DRIFT-RISK — {len(drifted)} cited file(s) changed since {commit}:")
        for p in drifted:
            print("  " + p)
        print("  → cited lines may have moved, or the claim about the file may be stale.")
        print("    Re-verify, then re-stamp MASTER.md's commit.")
        return 1 if strict else 0
    print(f"map_lint: clean ({n} cited files resolve; map stamped at {commit or 'unstamped'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
