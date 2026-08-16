
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))  

from e2e_synthetic_common import permutation_null_oof  
from fva.shuffle_sentinel import coalition_oof  

from pinksight.data.synthetic_streams import (  
    _B_NTILES_RANGE,
    MODALITY_C_GENES,
    build_stream_manifest,
    build_stream_report,
    generate_wsi_genomics_stream,
)
from pinksight.eval.e2e_report_contract import (  
    assert_synthetic_provenance,
    control_verdict,
)

ORGAN = "trackb-wsi-genomics"
DEFAULT_EFFECT_SIZE = 0.5
BAG_FEATURE = "wsi_bag_meanpool_1536d"  


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:  
        return "unknown"


def _pool_stream(gen) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bag_embs: list[np.ndarray] = []
    gene_rows: list[np.ndarray] = []
    labels: list[int] = []
    pids: list[str] = []
    for pid, bag, genes, label in gen:
        bag_embs.append(bag.mean(axis=0).astype(np.float32))  
        gene_rows.append(genes)
        labels.append(label)
        pids.append(pid)
    return (
        np.stack(bag_embs, axis=0),
        np.stack(gene_rows, axis=0),
        np.asarray(labels, dtype=int),
        np.asarray(pids),
    )


def run_control(
    stream_name: str,
    n: int,
    seed: int,
    git_commit: str,
    effect_size: float = DEFAULT_EFFECT_SIZE,
    n_tiles_range: tuple[int, int] = _B_NTILES_RANGE,
) -> dict:
    effect = 0.0 if stream_name == "negative_control" else effect_size
    gen = generate_wsi_genomics_stream(n, seed=seed, effect_size=effect, n_tiles_range=n_tiles_range)
    bag_emb, genes, y, pids = _pool_stream(gen)
    if not (np.isfinite(bag_emb).all() and np.isfinite(genes).all()):
        raise ValueError("non-finite bag/gene matrix — Stream-B plumbing bug (hard fail, not degraded)")

    real_oof = coalition_oof([bag_emb, genes], [False, False], y, pids, seed=seed, shuffle=False)
    shuffle_oof = permutation_null_oof([bag_emb, genes], [False, False], y, pids)
    verdict = control_verdict(stream_name, y=y, real_oof=real_oof, shuffle_oof=shuffle_oof)

    features = list(MODALITY_C_GENES) + [BAG_FEATURE]
    config = {
        "organ": ORGAN, "stream_name": stream_name, "n": n, "seed": seed, "effect_size": effect,
        "git_commit": git_commit, "n_tiles_min": int(n_tiles_range[0]), "n_tiles_max": int(n_tiles_range[1]),
    }
    manifest = build_stream_manifest(config)
    report = build_stream_report(ORGAN, stream_name, manifest, features, verdict)
    assert_synthetic_provenance(report, manifest["manifest_sha256"])  
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-patients", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--effect-size", type=float, default=DEFAULT_EFFECT_SIZE)
    ap.add_argument("--out-dir", type=Path,
                    default=ROOT / "process/general-plans/active/synthetic-all-streams-e2e_08-08-26")
    args = ap.parse_args()

    git_commit = _git_commit()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for stream_name in ("negative_control", "positive_control"):
        report = run_control(stream_name, args.n_patients, args.seed, git_commit, args.effect_size)
        out = args.out_dir / f"e2e_synthetic_trackb_{stream_name}.json"
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        v = report["controlVerdict"]
        print(f"[{ORGAN}] {stream_name}: verdict={v['verdict']} "  
              f"auroc={v.get('auroc')} shuffle={v.get('shuffleAuroc')} -> {out.name}")
    print(f"[{ORGAN}] done, n={args.n_patients}, {time.time() - t0:.1f}s "  
          "(SYNTHETIC — NOT A RESULT; forward-only plumbing, no LOCK moved)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
