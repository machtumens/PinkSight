#!/usr/bin/env bash
# fetch_weights.sh — download + SHA-256-verify the 15 G5 imaging-encoder weight files.
#
# These 15 files are distributed as GitHub Release assets (not in the git tree). They are licensed
# CC-BY-NC-4.0 (see LICENSE-WEIGHTS.md), separately from the Apache-2.0 code license; derived from
# Duke-Breast-Cancer-MRI (TCIA) and inherit its non-commercial term. Attribution required — cite
# Saha et al. (2021), DOI 10.7937/TCIA.e3sv-re93 — and the files must not be used to re-identify
# subjects. (Pure-shell alternative to scripts/fetch_weights.py; needs curl-or-wget + sha256sum-or-shasum.)
#
# THE RELEASE TAG IS A PLACEHOLDER until the GitHub Release is published. Set it at fetch time:
#   PINKSIGHT_RELEASE_TAG=vX.Y.Z-weights ./scripts/fetch_weights.sh
# Or override the whole base URL (a mirror):
#   PINKSIGHT_WEIGHTS_BASE_URL=https://host/path ./scripts/fetch_weights.sh
set -euo pipefail

REPO="${PINKSIGHT_RELEASE_REPO:-machtumens/PinkSight}"
TAG="${PINKSIGHT_RELEASE_TAG:-REPLACE_WITH_RELEASE_TAG}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKSUM_FILE="${ROOT}/scripts/g5_weights.sha256"
BASE_URL="${PINKSIGHT_WEIGHTS_BASE_URL:-https://github.com/${REPO}/releases/download/${TAG}}"

if [ "${TAG}" = "REPLACE_WITH_RELEASE_TAG" ] && [ -z "${PINKSIGHT_WEIGHTS_BASE_URL:-}" ]; then
  echo "ERROR: no release tag set — the GitHub Release is not published yet (tag is a placeholder)." >&2
  echo "  PINKSIGHT_RELEASE_TAG=v1.0.0-weights ./scripts/fetch_weights.sh" >&2
  echo "  (or set PINKSIGHT_WEIGHTS_BASE_URL to wherever the 15 .pt assets are hosted)" >&2
  exit 1
fi

# Pick a downloader.
if command -v curl >/dev/null 2>&1; then
  dl() { curl -fSL --retry 3 -o "$2" "$1"; }
elif command -v wget >/dev/null 2>&1; then
  dl() { wget -q -O "$2" "$1"; }
else
  echo "ERROR: need curl or wget on PATH." >&2
  exit 1
fi

# Pick a checker for `<hash>  <path>` files.
if command -v sha256sum >/dev/null 2>&1; then
  verify() { sha256sum -c "$1"; }
elif command -v shasum >/dev/null 2>&1; then
  verify() { shasum -a 256 -c "$1"; }
else
  echo "ERROR: need sha256sum or shasum on PATH." >&2
  exit 1
fi

echo "repo: ${REPO}   tag: ${TAG}"
echo "asset base URL: ${BASE_URL}"
echo "------------------------------------------------------------"

# Download every file named in the checksum manifest (skip blanks/comments).
while read -r _hash relpath; do
  case "${_hash}" in "" | \#*) continue ;; esac
  name="$(basename "${relpath}")"
  out="${ROOT}/${relpath}"
  if [ -f "${out}" ]; then
    echo "present  ${relpath} (will verify)"
  else
    echo "download ${BASE_URL}/${name}"
    mkdir -p "$(dirname "${out}")"
    dl "${BASE_URL}/${name}" "${out}.part"
    mv "${out}.part" "${out}"
  fi
done <"${CHECKSUM_FILE}"

echo "------------------------------------------------------------"
echo "verifying against ${CHECKSUM_FILE} ..."
# Paths in the manifest are repo-root-relative; verify from the repo root.
cd "${ROOT}"
verify "${CHECKSUM_FILE}"
echo "All 15 G5 weight files present and SHA-256-verified."
