#!/usr/bin/env python3
"""fetch_weights — download + SHA-256-verify the 15 G5 imaging-encoder weight files.

These 15 files are distributed as **GitHub Release assets** (they are not committed to the git
tree). This script downloads each asset from the release, verifies it against the SHA-256 checksum
manifest (``scripts/g5_weights.sha256``, built from ``results/TRAINED_ARTIFACTS.md``), and fails
loudly on any hash mismatch — it never leaves an unverified weight file in place.

The weight files are licensed CC-BY-NC-4.0 (see ``LICENSE-WEIGHTS.md``), separately from the
Apache-2.0 code license; they are derived from Duke-Breast-Cancer-MRI (TCIA) and inherit its
non-commercial term. Attribution is required — cite Saha et al. (2021), DOI 10.7937/TCIA.e3sv-re93 —
and the files must not be used to re-identify subjects. They are provided for reproducibility and
provenance of the research pipeline; see the release results and
``docs/adr/0008-g3-fusion-architecture-reframe.md`` for what they do and do not demonstrate.

THE RELEASE TAG IS A PLACEHOLDER until the GitHub Release is published. Set it at fetch time:

    PINKSIGHT_RELEASE_TAG=vX.Y.Z-weights  python3 scripts/fetch_weights.py
    python3 scripts/fetch_weights.py --tag vX.Y.Z-weights

Other modes:

    python3 scripts/fetch_weights.py --check          # verify already-present files, no network
    python3 scripts/fetch_weights.py --base-url URL   # fetch <URL>/<filename> from a mirror

Stdlib only — no third-party dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # scripts/ -> repo root

DEFAULT_REPO = "machtumens/PinkSight"
# Placeholder — the GitHub Release does not exist yet. Override via --tag or PINKSIGHT_RELEASE_TAG
# at release time (e.g. "v1.0.0-weights").
PLACEHOLDER_TAG = "REPLACE_WITH_RELEASE_TAG"
DEFAULT_CHECKSUM_FILE = ROOT / "scripts" / "g5_weights.sha256"
CHUNK = 1 << 20  # 1 MiB
TIMEOUT = 120
USER_AGENT = "pinksight-fetch-weights/1.0"


def parse_checksums(path: Path) -> list[tuple[str, str]]:
    """Return ``[(repo_relative_path, sha256_hex), ...]`` from a sha256sum-format file."""
    if not path.exists():
        sys.exit(f"ERROR: checksum file not found: {path}")
    rows: list[tuple[str, str]] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2 or len(parts[0]) != 64:
            sys.exit(f"ERROR: malformed checksum line {path}:{lineno}: {raw!r}")
        rows.append((parts[1], parts[0].lower()))
    if not rows:
        sys.exit(f"ERROR: no checksum rows parsed from {path}")
    return rows


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest_tmp: Path) -> str:
    """Stream ``url`` -> ``dest_tmp``, returning the sha256 hex of the bytes written."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    dest_tmp.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp, dest_tmp.open("wb") as out:
        for chunk in iter(lambda: resp.read(CHUNK), b""):
            out.write(chunk)
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description="Download + SHA-256-verify the 15 G5 imaging-encoder weight files "
        "(distributed as GitHub Release assets; CC-BY-NC-4.0, see LICENSE-WEIGHTS.md)."
    )
    ap.add_argument(
        "--tag",
        default=os.environ.get("PINKSIGHT_RELEASE_TAG", PLACEHOLDER_TAG),
        help="GitHub Release tag the .pt assets are attached to (or set PINKSIGHT_RELEASE_TAG).",
    )
    ap.add_argument(
        "--repo",
        default=os.environ.get("PINKSIGHT_RELEASE_REPO", DEFAULT_REPO),
        help=f"owner/name of the GitHub repo hosting the release (default {DEFAULT_REPO}).",
    )
    ap.add_argument(
        "--base-url",
        default=os.environ.get("PINKSIGHT_WEIGHTS_BASE_URL"),
        help="Override the asset base URL entirely (a mirror). Assets fetched from <base-url>/<file>.",
    )
    ap.add_argument(
        "--checksum-file",
        type=Path,
        default=DEFAULT_CHECKSUM_FILE,
        help="SHA-256 manifest to verify against (sha256sum format).",
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="Verify already-present files against the manifest; download nothing.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if a correct file is already present.",
    )
    args = ap.parse_args(argv)

    rows = parse_checksums(args.checksum_file)

    # Resolve the asset base URL (skipped in --check mode; nothing is downloaded there).
    base_url = ""
    if not args.check:
        if args.base_url:
            base_url = args.base_url.rstrip("/")
        elif args.tag == PLACEHOLDER_TAG:
            sys.exit(
                "ERROR: no release tag set. The GitHub Release is not published yet, so the tag is\n"
                "a placeholder. Set it once the release exists, e.g.:\n"
                "  PINKSIGHT_RELEASE_TAG=v1.0.0-weights python3 scripts/fetch_weights.py\n"
                "  python3 scripts/fetch_weights.py --tag v1.0.0-weights\n"
                "Or point --base-url at wherever the 15 .pt assets are hosted.\n"
                "(Use --check to verify files you already have, without any network access.)"
            )
        else:
            base_url = f"https://github.com/{args.repo}/releases/download/{args.tag}"

    print(f"checksum file : {args.checksum_file}")
    if args.check:
        print("mode          : --check (verify only, no download)")
    else:
        print(f"asset base URL: {base_url}")
    print("-" * 62)

    verified = 0
    downloaded = 0
    cached = 0
    failed: list[str] = []

    for rel, expected in rows:
        # Destination honours the manifest's repo-relative path so `sha256sum -c` matches too.
        dest = ROOT / rel
        name = Path(rel).name

        if dest.exists() and not args.force:
            actual = sha256_file(dest)
            if actual == expected:
                print(f"OK (cached)    {rel}")
                verified += 1
                cached += 1
                continue
            if args.check:
                print(f"FAIL (hash)    {rel}\n  expected {expected}\n  actual   {actual}")
                failed.append(rel)
                continue
            print(f"stale on disk  {rel} (hash mismatch) — re-fetching")

        if args.check:
            print(f"MISSING        {rel}")
            failed.append(rel)
            continue

        url = f"{base_url}/{name}"
        tmp = dest.parent / (dest.name + ".part")
        try:
            actual = download(url, tmp)
        except urllib.error.HTTPError as e:
            tmp.unlink(missing_ok=True)
            hint = " (release/tag/asset may not exist yet)" if e.code == 404 else ""
            print(f"FAIL (http {e.code}){hint}  {url}")
            failed.append(rel)
            continue
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            tmp.unlink(missing_ok=True)
            print(f"FAIL (network) {url}\n  {e}")
            failed.append(rel)
            continue

        if actual != expected:
            tmp.unlink(missing_ok=True)
            print(
                f"FAIL (hash)    {rel}\n  expected {expected}\n  actual   {actual}\n"
                "  -> deleted the download; refusing to keep an unverified weight file."
            )
            failed.append(rel)
            continue

        os.replace(tmp, dest)
        print(f"OK (verified)  {rel}")
        verified += 1
        downloaded += 1

    print("-" * 62)
    print(f"{verified}/{len(rows)} verified  ({downloaded} downloaded, {cached} cached)")
    if failed:
        print(f"{len(failed)} FAILED:")
        for rel in failed:
            print(f"  - {rel}")
        print("Nothing partial was left in place. Fix the tag / network and re-run.")
        return 1
    print("All 15 G5 weight files present and SHA-256-verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
