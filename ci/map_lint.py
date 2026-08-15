#!/usr/bin/env python3
"""Anti-drift gate for docs/map/ — the MoE router cluster.

The cluster's whole value is that an agent trusts it enough not to re-read the
sources. That trust is only safe if every `(src: file:LINE)` still resolves and
every expert link still points at a real file. This lint is that check.

Two failure classes, deliberately different severity:

  BROKEN  (exit 1) — a citation names a file that does not exist, or a line
          past EOF, or an expert link that dangles. The map is lying now.
  DRIFT   (exit 0, or 1 under --strict) — a cited file changed since the commit
          stamped in MASTER.md's header. Line numbers may have shifted, so the
          citation may now point at the wrong line. Needs a human re-verify.

ponytail: one tolerant regex pass over 9 files; no parser, no config, no deps.
Citation styles vary (backticks, en-dashes, `§Section`, bare paths) because the
experts were hand-written — so parse leniently and only assert on what is
unambiguously a path.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

MAP_DIR = Path("docs/map")
MASTER = "MASTER.md"

# `(src: ...)` groups. Non-greedy, single-line — citations never wrap in this cluster.
SRC_RE = re.compile(r"\(src:\s*([^)]*)\)")
# A path-looking token, optionally with :LINE or :LINE-LINE (ASCII or en-dash).
PATH_RE = re.compile(r"^([\w./_-]+\.[\w]+|[\w./_-]+/)(?::(\d+)(?:[-–](\d+))?)?$")
# Markdown links between expert files: [label](e2-gates-and-numbers.md)
LINK_RE = re.compile(r"\]\((?!https?:)([^)#]+?\.md)(?:#[^)]*)?\)")
COMMIT_RE = re.compile(r"Built from commit `([0-9a-f]{7,40})`")


def _clean(part: str) -> str:
    """Strip decoration a human added around a path: backticks, prose tails, §refs."""
    part = part.strip().strip("`").strip()
    # Cut prose tails: "decisions.md:43 — LOCK-1", "CLAIM_LEDGER.md §Targets"
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
    """BROKEN-class problems only: dangling files, lines past EOF, dead links."""
    problems: list[str] = []
    for md in sorted((root / MAP_DIR).glob("*.md")):
        for lineno, line in enumerate(md.read_text(encoding="utf-8").splitlines(), 1):
            for group in SRC_RE.findall(line):
                for raw in re.split(r"[,;]", group):
                    part = _clean(raw)
                    m = PATH_RE.match(part)
                    if not m:
                        continue  # prose, ellipsis, §section — not a checkable path
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
    """Run git. None means git itself was unusable — missing binary, timeout, not a repo."""
    try:
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, timeout=20, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None


def staleness_verdict(root: Path) -> tuple[str | None, list[str] | None]:
    """Which cited files moved since MASTER.md's stamped build commit.

    Returns (stamped_commit, drifted). An empty list means the map's line numbers were
    verified against the same tree that is checked out now.

    `drifted is None` means the stamp could not be checked AT ALL — no stamp, no git, or
    a stamp that is not an ancestor of HEAD. Callers must treat that as a failure, not as
    "clean". The orphan case is not theoretical: on 2026-07-29 a rebase elsewhere in this
    repo left MASTER.md stamped `0f9ee09`, a commit that still existed as an object but had
    been dropped from history. `git diff` against it SUCCEEDS and returns a plausible file
    list computed against an abandoned tree. Returning "no drift" there would make the gate
    green at exactly the moment it knows least.
    """
    master = root / MAP_DIR / MASTER
    if not master.exists():
        return None, None
    m = COMMIT_RE.search(master.read_text(encoding="utf-8"))
    if not m:
        return None, None
    commit = m.group(1)
    if (anc := _git(root, "merge-base", "--is-ancestor", commit, "HEAD")) is None or anc.returncode:
        return commit, None
    # Bare commit (not `commit..HEAD`) so this compares against the WORKING TREE. An agent
    # reads files on disk, not at HEAD — an uncommitted edit to a cited file shifts line
    # numbers just as surely as a committed one. In CI the tree is clean, so this
    # degenerates to commit..HEAD anyway.
    if (changed := _git(root, "diff", "--name-only", commit)) is None or changed.returncode:
        return commit, None
    return commit, sorted(cited_files(root) & set(changed.stdout.split()))


def _selfcheck() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / MAP_DIR).mkdir(parents=True)
        (root / "real.md").write_text("a\nb\nc\n")
        (root / "CLAIM_LEDGER.md").write_text("x\n")  # bare-path cite: exists, must not flag
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
        # The tempdir is not a git checkout, so the stamp cannot be verified. That must come
        # back as None (-> exit 1), never as an empty list (-> "clean"). This is the guard on
        # the failure mode that shipped: a gate going quiet when it loses the ability to check.
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
        # Loud on purpose. An unverifiable stamp means drift detection is silently OFF,
        # and a map that has stopped warning looks exactly like a map that is fresh.
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
        # Both citation shapes are worth flagging: a line-anchored cite can now point at the
        # wrong line, and a bare-path cite can describe behaviour the file no longer has.
        print("  → cited lines may have moved, or the claim about the file may be stale.")
        print("    Re-verify, then re-stamp MASTER.md's commit.")
        # Policy (settled 2026-07-29): warn locally, fail in CI via --strict.
        # A dev tree is dirty by definition, so a hard local fail would be switched off within
        # a week. CI runs on a clean checkout, so there DRIFT-RISK means a real un-re-verified
        # change. Auto-stamping was rejected outright: it removes the maintenance chore by
        # destroying the signal, letting the map claim a freshness nobody checked.
        return 1 if strict else 0
    print(f"map_lint: clean ({n} cited files resolve; map stamped at {commit or 'unstamped'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
