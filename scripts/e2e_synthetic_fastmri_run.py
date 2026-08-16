
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
sys.path.insert(0, str(ROOT / "scripts"))  

from e2e_synthetic_common import permutation_null_oof  
from fva.shuffle_sentinel import coalition_oof  

from pinksight.data.fastmri_nyu import assert_images_only  
from pinksight.data.synthetic_streams import (  
    FASTMRI_CHANNELS,
    FASTMRI_IMAGE_FEATURES,
    build_stream_manifest,
    build_stream_report,
    generate_fastmri_stream,
)
from pinksight.eval.e2e_report_contract import (  
    assert_synthetic_provenance,
    control_verdict,
)
from pinksight.models.mri_encoder import MriEncoder  
from pinksight.seed import set_seed  

ORGAN = "fastmri-nyu-standalone"
DEFAULT_EFFECT_SIZE = 2.0


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:  
        return "unknown"


def build_encoder() -> MriEncoder:
    enc = MriEncoder(in_channels=FASTMRI_CHANNELS, depth=18, medicalnet_weights=None).eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


def _embed_batch(encoder: MriEncoder, cubes: list[np.ndarray]) -> np.ndarray:
    x = torch.from_numpy(np.stack(cubes, axis=0)).float()  
    assert_images_only(x, expected_channels=FASTMRI_CHANNELS)  
    with torch.no_grad():
        emb = encoder.embed(x)
    return emb.cpu().numpy().astype(np.float32)


def _encode_all(encoder, gen, batch_size) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    embs: list[np.ndarray] = []
    labels: list[int] = []
    pids: list[str] = []
    batch: list[np.ndarray] = []
    for pid, cube, label in gen:
        assert_images_only(cube, expected_channels=FASTMRI_CHANNELS)  
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
    effect = 0.0 if stream_name == "negative_control" else effect_size
    gen = generate_fastmri_stream(n, seed=seed, cube_size=cube_size, effect_size=effect)
    mri_emb, y, pids = _encode_all(encoder, gen, batch_size)
    if not np.isfinite(mri_emb).all():
        raise ValueError("non-finite MRI embedding — Stream-F encoder plumbing bug (hard fail)")

    real_oof = coalition_oof([mri_emb], [False], y, pids, seed=seed, shuffle=False)
    shuffle_oof = permutation_null_oof([mri_emb], [False], y, pids)
    verdict = control_verdict(stream_name, y=y, real_oof=real_oof, shuffle_oof=shuffle_oof)

    config = {
        "organ": ORGAN, "stream_name": stream_name, "n": n, "seed": seed, "effect_size": effect,
        "git_commit": git_commit, "cube_size": cube_size, "channels": "pre_post",
    }
    manifest = build_stream_manifest(config)
    report = build_stream_report(ORGAN, stream_name, manifest, FASTMRI_IMAGE_FEATURES, verdict)
    assert_synthetic_provenance(report, manifest["manifest_sha256"])  
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
    set_seed(args.seed)  
    encoder = build_encoder()
    t0 = time.time()
    for stream_name in ("negative_control", "positive_control"):
        report = run_control(encoder, stream_name, args.n_patients, args.seed, args.cube_size,
                             args.batch_size, git_commit, args.effect_size)
        out = args.out_dir / f"e2e_synthetic_fastmri_{stream_name}.json"
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        v = report["controlVerdict"]
        print(f"[{ORGAN}] {stream_name}: verdict={v['verdict']} "  
              f"auroc={v.get('auroc')} shuffle={v.get('shuffleAuroc')} -> {out.name}")
    print(f"[{ORGAN}] done, n={args.n_patients}, {time.time() - t0:.1f}s "  
          "(SYNTHETIC — NOT A RESULT; images-only forward-only plumbing, no LOCK moved)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
