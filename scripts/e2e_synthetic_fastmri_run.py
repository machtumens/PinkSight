#!/usr/bin/env python3
"""E2E synthetic plumbing harness — fastMRI-NYU standalone encoder organ (images-only; ADR-0016).

Forward-only, $0-local, CPU, no training-on-synthetic. Forwards a synthetic NYU-shaped sub-cohort of
2-channel (pre, post) DCE cubes through the REAL frozen (random-init) fastMRI 3D-ResNet encoder, then
runs the SAME control-sentinel spine every organ shares (coalition_oof real + permutation-null shuffle
-> ``control_verdict``) over the [mri_embedding] coalition, for both the negative and positive control,
and writes one non-reportable report JSON per control.

IMAGES-ONLY (ADR-0016 Fix #6): no biomarker/clinical column is ever an input — ``assert_images_only``
guards every batch and the fastMRI biomarker columns stay quarantined in FORBIDDEN_FEATURES. The 51
verified-normal / H6 anomaly head is NOT modelled (hard interlock): the synthetic labels are strictly
{benign, malignant}, so there is no "normal" analog in any batch. NYU-INTERNAL characterisation ONLY —
no synthetic number is ever pooled or juxtaposed with a Duke (or any real) number; the number proves
the encoder's tensor wiring, never biology, and no LOCK is moved.

Usage: uv run --extra ml python scripts/e2e_synthetic_fastmri_run.py --n-patients 10000
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))  # so `from fva.shuffle_sentinel import ...` resolves

from e2e_synthetic_common import permutation_null_oof  # noqa: E402
from fva.shuffle_sentinel import coalition_oof  # noqa: E402

from pinksight.data.fastmri_nyu import assert_images_only  # noqa: E402
from pinksight.data.synthetic_streams import (  # noqa: E402
    FASTMRI_CHANNELS,
    FASTMRI_IMAGE_FEATURES,
    build_stream_manifest,
    build_stream_report,
    generate_fastmri_stream,
)
from pinksight.eval.e2e_report_contract import (  # noqa: E402
    assert_synthetic_provenance,
    control_verdict,
)
from pinksight.models.mri_encoder import MriEncoder  # noqa: E402
from pinksight.seed import set_seed  # noqa: E402

ORGAN = "fastmri-nyu-standalone"
# Disclosed positive-control blob magnitude (a deliberate small effect, NOT a leak). Calibrated so the
# images-only random-init encoder embedding recovers real AUROC above 0.75 while the permutation-null
# shuffle collapses to chance — an images-only stream has no clinical crutch, so it needs a clear blob.
DEFAULT_EFFECT_SIZE = 2.0


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:  # noqa: BLE001 — provenance is best-effort; 'unknown' is honest if git is absent
        return "unknown"


def build_encoder() -> MriEncoder:
    """ONE frozen, random-init images-only fastMRI 3D-ResNet encoder for the whole run (2-channel DCE).
    Stateless/frozen -> forward every patient through the same weights; never gradient-trained (DD-1)."""
    enc = MriEncoder(in_channels=FASTMRI_CHANNELS, depth=18, medicalnet_weights=None).eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


def _embed_batch(encoder: MriEncoder, cubes: list[np.ndarray]) -> np.ndarray:
    x = torch.from_numpy(np.stack(cubes, axis=0)).float()  # (B, 2, cube, cube, cube)
    assert_images_only(x, expected_channels=FASTMRI_CHANNELS)  # firewall: images-only (ADR-0016 Fix #6)
    with torch.no_grad():
        emb = encoder.embed(x)
    return emb.cpu().numpy().astype(np.float32)


def _encode_all(encoder, gen, batch_size) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stream the generator through the frozen encoder in batches; no bulk cube array is held (one batch
    at a time, firewall #5). Returns (mri_emb (n, 512), y, pids)."""
    embs: list[np.ndarray] = []
    labels: list[int] = []
    pids: list[str] = []
    batch: list[np.ndarray] = []
    for pid, cube, label in gen:
        assert_images_only(cube, expected_channels=FASTMRI_CHANNELS)  # per-item images-only guard
        batch.append(cube)
        labels.append(label)
        pids.append(pid)
        if len(batch) == batch_size:
            embs.append(_embed_batch(encoder, batch))
            batch = []
    if batch:
        embs.append(_embed_batch(encoder, batch))
    return np.concatenate(embs, axis=0), np.asarray(labels, dtype=int), np.asarray(pids)


def run_control(
    encoder: MriEncoder,
    stream_name: str,
    n: int,
    seed: int,
    cube_size: int,
    batch_size: int,
    git_commit: str,
    effect_size: float = DEFAULT_EFFECT_SIZE,
) -> dict:
    """Generate one fastMRI control sub-cohort, forward it through the frozen encoder, run the real +
    permutation-null sentinel over the [mri_embedding] coalition, and assemble a gated non-reportable
    report. ``stream_name`` is 'negative_control' / 'positive_control'."""
    effect = 0.0 if stream_name == "negative_control" else effect_size
    gen = generate_fastmri_stream(n, seed=seed, cube_size=cube_size, effect_size=effect)
    mri_emb, y, pids = _encode_all(encoder, gen, batch_size)
    if not np.isfinite(mri_emb).all():
        raise ValueError("non-finite MRI embedding — Stream-F encoder plumbing bug (hard fail)")

    # SAME forward spine as every organ: LogReg over StratifiedGroupKFold, each patient its own group
    # (unique IDs) -> patient-disjoint folds. Images-only: the ONLY coalition matrix is the 512-dim MRI
    # embedding. Real OOF is a single pass; the shuffle sentinel is the permutation-null MEAN.
    real_oof = coalition_oof([mri_emb], [False], y, pids, seed=seed, shuffle=False)
    shuffle_oof = permutation_null_oof([mri_emb], [False], y, pids)
    verdict = control_verdict(stream_name, y=y, real_oof=real_oof, shuffle_oof=shuffle_oof)

    config = {
        "organ": ORGAN, "stream_name": stream_name, "n": n, "seed": seed, "effect_size": effect,
        "git_commit": git_commit, "cube_size": cube_size, "channels": "pre_post",
    }
    manifest = build_stream_manifest(config)
    report = build_stream_report(ORGAN, stream_name, manifest, FASTMRI_IMAGE_FEATURES, verdict)
    assert_synthetic_provenance(report, manifest["manifest_sha256"])  # firewall #2: gate before returning
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-patients", type=int, default=10000)
    ap.add_argument("--cube-size", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--effect-size", type=float, default=DEFAULT_EFFECT_SIZE)
    ap.add_argument("--out-dir", type=Path,
                    default=ROOT / "process/general-plans/active/synthetic-all-streams-e2e_08-08-26")
    args = ap.parse_args()

    git_commit = _git_commit()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)  # pin torch BEFORE the random-init encoder is built (LAW L-3 reproducibility)
    encoder = build_encoder()
    t0 = time.time()
    for stream_name in ("negative_control", "positive_control"):
        report = run_control(encoder, stream_name, args.n_patients, args.seed, args.cube_size,
                             args.batch_size, git_commit, args.effect_size)
        out = args.out_dir / f"e2e_synthetic_fastmri_{stream_name}.json"
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        v = report["controlVerdict"]
        print(f"[{ORGAN}] {stream_name}: verdict={v['verdict']} "  # noqa: T201
              f"auroc={v.get('auroc')} shuffle={v.get('shuffleAuroc')} -> {out.name}")
    print(f"[{ORGAN}] done, n={args.n_patients}, {time.time() - t0:.1f}s "  # noqa: T201
          "(SYNTHETIC — NOT A RESULT; images-only forward-only plumbing, no LOCK moved)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
