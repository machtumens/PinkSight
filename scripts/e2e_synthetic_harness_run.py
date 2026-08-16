
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))  

from fva.shuffle_sentinel import coalition_oof  

from pinksight import FORBIDDEN_FEATURES  
from pinksight.data.fastmri_heuristic_mask import enhancement_mask  
from pinksight.data.lesion_crop import RIM_MM_DEFAULT, lesion_crop  
from pinksight.data.synthetic_cohort import (  
    CAT_CARDINALITIES,
    build_manifest,
    generate_negative_control,
    generate_positive_control,
    generate_realistic_negative_control,
    generate_realistic_positive_control,
)
from pinksight.eval.calibration import calibration_report  
from pinksight.eval.e2e_report_contract import (  
    KI67_DESCRIPTOR_DEFAULT,
    build_report,
    control_verdict,
)
from pinksight.models.clinical_encoder import FEATURES_CAT, FEATURES_NUM  
from pinksight.models.fusion import (  
    FocalLoss,
    FusionModel,
    KendallUncertaintyWeighting,
    Ki67HuberLoss,
)
from pinksight.models.mri_encoder import MriEncoder  
from pinksight.seed import set_seed  
from pinksight.xai.faithfulness import iou, pointing_game, randomization_test, resample_mask_to  
from pinksight.xai.saliency import grad_cam_3d, randomize_weights  

MIN_N_FOR_SENTINEL = 20
TASK_FOLDER = ROOT / "process" / "general-plans" / "active" / "e2e-synthetic-harness_07-08-26"


def stage1_lesion_crop(mri_pair: np.ndarray, cube_size: int) -> tuple[np.ndarray, bool, np.ndarray]:
    pre, post = mri_pair[0], mri_pair[1]
    mask = enhancement_mask(pre, post)  
    used_fallback = not bool(mask.any())  
    crop = lesion_crop(mri_pair, mask, rim_mm=RIM_MM_DEFAULT, out_size=cube_size)
    mask_cube = resample_mask_to(mask, (cube_size, cube_size, cube_size)).astype(bool)
    return crop.astype(np.float32), used_fallback, mask_cube


def build_encoder(in_channels: int = 2, depth: int = 18) -> MriEncoder:
    enc = MriEncoder(in_channels=in_channels, depth=depth, medicalnet_weights=None).eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


def _encode_batch(encoder: MriEncoder, crops: list[np.ndarray]) -> np.ndarray:
    x = torch.from_numpy(np.stack(crops, axis=0)).float()  
    with torch.no_grad():
        emb = encoder.embed(x)
    return emb.cpu().numpy().astype(np.float32)


def _impute_standardize(x_num: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    med = np.nanmedian(x_num[train_idx], axis=0)
    xn = np.where(np.isnan(x_num), med, x_num)
    mu, sd = xn[train_idx].mean(0), xn[train_idx].std(0) + 1e-8
    return (xn - mu) / sd


def _one_hot(x_cat: np.ndarray, cards: list[int]) -> np.ndarray:
    cols = [np.eye(card, dtype=float)[np.clip(x_cat[:, j], 0, card - 1)] for j, card in enumerate(cards)]
    return np.hstack(cols) if cols else np.zeros((len(x_cat), 0))


def stage2b_clinical(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    from rtdl_revisiting_models import FTTransformer

    cards = [CAT_CARDINALITIES[f] for f in FEATURES_CAT]
    x_num = np.array([[row[f] for f in FEATURES_NUM] for row in rows], dtype=float)
    x_cat = np.array([[row[f] for f in FEATURES_CAT] for row in rows], dtype=int)
    xn = _impute_standardize(x_num, np.arange(len(rows)))
    clinical_matrix = np.hstack([xn, _one_hot(x_cat, cards)]).astype(float)

    ftt = FTTransformer(
        n_cont_features=xn.shape[1],
        cat_cardinalities=cards,
        d_out=1,
        **FTTransformer.get_default_kwargs(n_blocks=2),
    ).eval()
    for p in ftt.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        ftt_out = ftt(
            torch.tensor(xn, dtype=torch.float32), torch.tensor(x_cat, dtype=torch.long)
        ).cpu().numpy().astype(np.float32)
    return clinical_matrix, ftt_out, cards


def stage3_fuse(mri_emb: np.ndarray, ftt_out: np.ndarray) -> dict[str, Any]:
    fm = FusionModel(modality_dims={"mri": 512, "clinical": ftt_out.shape[1]}, fused_dim=128).eval()
    for p in fm.parameters():
        p.requires_grad_(False)
    feats = {
        "mri": torch.tensor(mri_emb, dtype=torch.float32),
        "clinical": torch.tensor(ftt_out, dtype=torch.float32),
    }
    with torch.no_grad():
        full = fm(feats, drop=None)
        clin_only = fm(feats, drop={"mri"})     
        mri_only = fm(feats, drop={"clinical"})  
    for name, out in (("full", full), ("clinical_only", clin_only), ("mri_only", mri_only)):
        if not bool(torch.isfinite(out["subtype_logit"]).all()):
            raise ValueError(f"non-finite subtype logit in the '{name}' fusion path — plumbing bug")
    return {
        "model": fm,
        "subtype_logit": full["subtype_logit"].cpu().numpy().reshape(-1),
        "ki67": full["ki67"].cpu().numpy().reshape(-1),
        "p_full": float(torch.sigmoid(full["subtype_logit"]).mean()),
        "p_no_mri": float(torch.sigmoid(clin_only["subtype_logit"]).mean()),
        "p_no_clinical": float(torch.sigmoid(mri_only["subtype_logit"]).mean()),
        "clinical_only_ran": True,
    }


def stage4_uncertainty(subtype_logit: np.ndarray, ki67: np.ndarray, y: np.ndarray) -> float:
    logit_t = torch.tensor(subtype_logit, dtype=torch.float32).reshape(-1, 1)
    ki67_t = torch.tensor(ki67, dtype=torch.float32).reshape(-1, 1)
    y_t = torch.tensor(y, dtype=torch.float32)
    with torch.no_grad():  
        l_subtype = FocalLoss()(logit_t, y_t)
        l_ki67 = Ki67HuberLoss()(ki67_t, torch.full((len(y),), float("nan")))
        total, _weights = KendallUncertaintyWeighting(n_tasks=2)([l_subtype, l_ki67])
    if not bool(torch.isfinite(total).all()):
        raise ValueError("non-finite Kendall-weighted total — uncertainty-weighting plumbing bug")
    return float(total)


def _last_conv3d(module: torch.nn.Module) -> torch.nn.Module:
    last = None
    for m in module.modules():
        if isinstance(m, torch.nn.Conv3d):
            last = m
    if last is None:
        raise RuntimeError("no Conv3d layer found for the Grad-CAM target")
    return last


def stage5_xai(
    encoder: MriEncoder,
    samples: list[tuple[np.ndarray, np.ndarray]],
    stream_name: str,
    maps_dir: Path,
) -> dict[str, Any]:
    target = _last_conv3d(encoder.backbone)
    rnd = randomize_weights(encoder)
    rnd_target = _last_conv3d(rnd.backbone)
    maps_dir.mkdir(parents=True, exist_ok=True)
    ious, hits, passes = [], [], []
    map_ref = ""
    for i, (crop, mask_cube) in enumerate(samples):
        vol = torch.tensor(crop, dtype=torch.float32).unsqueeze(0)  
        cam = grad_cam_3d(encoder, vol, target)
        cam_rnd = grad_cam_3d(rnd, vol, rnd_target)
        if not np.isfinite(cam).all():
            raise ValueError("non-finite Grad-CAM map — XAI plumbing bug")
        ious.append(iou(cam, mask_cube))
        hits.append(pointing_game(cam, mask_cube))
        passes.append(randomization_test(cam, cam_rnd)["passed"])
        if i == 0:
            map_ref = f"{maps_dir.name}/{stream_name}_000.npy"
            np.save(maps_dir / f"{stream_name}_000.npy", cam)
    return {
        "mapRef": map_ref,
        "iou": round(float(np.mean(ious)), 4) if ious else None,
        "pointingGame": bool(np.mean(hits) > 0.5) if hits else None,
        "randomizationPassed": bool(np.mean(passes) > 0.5) if passes else None,
        "note": "reported-not-gated (TICKET-001 discipline) — encoder is random-init, forward-only",
    }


def stage6_control(
    mri_emb: np.ndarray, clinical_matrix: np.ndarray, y: np.ndarray, stream_name: str, seed: int,
) -> dict[str, Any]:
    y = np.asarray(y, int)
    if len(y) < MIN_N_FOR_SENTINEL or len(np.unique(y)) < 2:
        return control_verdict(stream_name)  
    groups = np.arange(len(y))
    mats = [np.asarray(mri_emb, float), np.asarray(clinical_matrix, float)]
    real_oof = coalition_oof(mats, [False, False], y, groups, seed=seed, shuffle=False)
    shuffle_oof = coalition_oof(mats, [False, False], y, groups, seed=seed, shuffle=True)
    return control_verdict(stream_name, y, real_oof, shuffle_oof)


def _stratified_halves(y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    val, test = [], []
    for cls in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == cls))
        half = len(idx) // 2
        val.extend(idx[:half].tolist())
        test.extend(idx[half:].tolist())
    return np.array(val, dtype=int), np.array(test, dtype=int)


def stage6_calibration(subtype_logit: np.ndarray, y: np.ndarray, seed: int) -> dict[str, Any] | None:
    y = np.asarray(y, int)
    val, test = _stratified_halves(y, seed)
    if len(val) == 0 or len(test) == 0:
        return None
    rep = calibration_report(subtype_logit[val], y[val], subtype_logit[test], y[test], n_bins=10)
    ece = rep["ece_after"]
    band = "good" if ece <= 0.05 else ("acceptable" if ece <= 0.10 else "poor")
    return {"ece": ece, "smoothEce": rep["smooth_ece_after"], "band": band}


def _generator_for(
    stream_name: str, n: int, seed: int, cube_size: int, channels: str, effect: float,
    realistic: bool = False,
):
    if stream_name == "negative_control":
        gen = generate_realistic_negative_control if realistic else generate_negative_control
        return gen(n, seed=seed, cube_size=cube_size, channels=channels)
    if stream_name == "positive_control":
        gen = generate_realistic_positive_control if realistic else generate_positive_control
        return gen(n, seed=seed, cube_size=cube_size, channels=channels, effect_size=effect)
    raise ValueError(f"unknown stream_name {stream_name!r}")


def run_stream(
    stream_name: str,
    n: int,
    seed: int = 0,
    cube_size: int = 16,
    channels: str = "pre_post",
    effect_size: float = 1.2,
    xai_subsample: int = 30,
    batch_size: int = 64,
    out_dir: Path | None = None,
    git_commit: str = "unknown",
    realistic: bool = False,
) -> dict[str, Any]:
    t0 = time.time()
    out_dir = Path(out_dir) if out_dir is not None else TASK_FOLDER
    collect_xai = stream_name == "positive_control" and xai_subsample > 0

    set_seed(seed)
    encoder = build_encoder(in_channels=2, depth=18)
    embeddings: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    labels: list[int] = []
    pids: list[str] = []
    xai_samples: list[tuple[np.ndarray, np.ndarray]] = []
    n_fallback = 0
    batch: list[np.ndarray] = []

    for pid, mri_pair, row, label in _generator_for(
        stream_name, n, seed, cube_size, channels, effect_size, realistic
    ):
        crop, used_fallback, mask_cube = stage1_lesion_crop(mri_pair, cube_size)
        n_fallback += int(used_fallback)
        batch.append(crop)
        rows.append(row)
        labels.append(label)
        pids.append(pid)
        if collect_xai and len(xai_samples) < xai_subsample:
            xai_samples.append((crop, mask_cube))
        if len(batch) == batch_size:
            embeddings.append(_encode_batch(encoder, batch))
            batch = []
    if batch:
        embeddings.append(_encode_batch(encoder, batch))

    mri_emb = np.concatenate(embeddings, axis=0)
    if not np.isfinite(mri_emb).all():
        raise ValueError("non-finite MRI embedding — Stage 2a plumbing bug (hard fail, not degraded)")
    y = np.asarray(labels, int)

    all_keys: set[str] = set().union(*(r.keys() for r in rows)) if rows else set()
    leaked = all_keys & set(FORBIDDEN_FEATURES)
    if leaked:
        raise ValueError(f"LEAKAGE at scale: forbidden feature(s) {sorted(leaked)} in synthetic rows")

    clinical_matrix, ftt_out, _cards = stage2b_clinical(rows)
    fused = stage3_fuse(mri_emb, ftt_out)
    kendall_total = stage4_uncertainty(fused["subtype_logit"], fused["ki67"], y)
    xai_out = stage5_xai(encoder, xai_samples, stream_name, out_dir / "e2e_synthetic_xai_maps") \
        if xai_samples else None
    cv_block = stage6_control(mri_emb, clinical_matrix, y, stream_name, seed)
    calib_block = stage6_calibration(fused["subtype_logit"], y, seed)

    manifest_cfg = {
        "n": n, "seed": seed, "cube_size": cube_size, "channels": channels,
        "effect_size": (effect_size if stream_name == "positive_control" else 0.0),
        "stream_name": stream_name, "git_commit": git_commit,
    }
    if realistic:  
        manifest_cfg["variant"] = "realistic"
    manifest = build_manifest(manifest_cfg)
    report = _assemble_report(stream_name, manifest, pids, fused, calib_block, xai_out, cv_block)

    return {
        "report": report,
        "manifest": manifest,
        "control_verdict": cv_block,
        "calibration": calib_block,
        "xai": xai_out,
        "y": y,
        "patient_ids": pids,
        "per_patient_subtype_prob": 1.0 / (1.0 + np.exp(-fused["subtype_logit"])),
        "per_patient_ki67_raw": fused["ki67"],
        "mri_embeddings": mri_emb,
        "clinical_matrix": clinical_matrix,
        "n_fallback_masks": n_fallback,
        "dropout_clinical_only_ran": fused["clinical_only_ran"],
        "kendall_total": kendall_total,
        "wall_clock_s": round(time.time() - t0, 2),
    }


def _assemble_report(
    stream_name: str, manifest: dict[str, Any], pids: list[str], fused: dict[str, Any],
    calib_block: dict[str, Any] | None, xai_out: dict[str, Any] | None, cv_block: dict[str, Any],
) -> dict[str, Any]:
    prob0 = float(1.0 / (1.0 + np.exp(-fused["subtype_logit"][0])))
    subtype_out = {
        "label": "Triple-Negative" if prob0 > 0.5 else "Luminal A",
        "probability": round(prob0, 4),
        "uncertainty": [round(max(0.0, prob0 - 0.13), 4), round(min(1.0, prob0 + 0.12), 4)],
        "abstained": False,
    }
    ki67_out = {"descriptor": KI67_DESCRIPTOR_DEFAULT, "stratum": "not_assessed"}
    modalities = [
        {"modality": "clinical", "present": True,
         "contribution": round(fused["p_full"] - fused["p_no_clinical"], 4)},
        {"modality": "mri", "present": True,
         "contribution": round(fused["p_full"] - fused["p_no_mri"], 4)},
        {"modality": "path", "present": False, "contribution": 0.0},
        {"modality": "genomic", "present": False, "contribution": 0.0},
    ]
    return build_report(
        stream_name, manifest, subtype_out, ki67_out, calib_block, modalities, xai_out, cv_block,
        study_id=pids[0] if pids else None,
    )


def git_commit_short() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short=7", "HEAD"], cwd=ROOT, capture_output=True, text=True,
            timeout=5, check=True,
        )
        return out.stdout.strip() or "unknown"
    except Exception:  
        return "unknown"


def _consort_line(stream_name: str, res: dict[str, Any]) -> str:
    cv = res["control_verdict"]
    return (
        f"[{stream_name}] n={len(res['patient_ids'])} generated+forward-passed, "
        f"fallback_masks={res['n_fallback_masks']}, control={cv.get('verdict')} "
        f"(auroc={cv.get('auroc', 'n/a')}, shuffle={cv.get('shuffleAuroc', 'n/a')}), "
        f"wall_clock={res['wall_clock_s']}s — SYNTHETIC, NOT A RESULT."
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-patients", type=int, default=10000)
    ap.add_argument("--cube-size", type=int, default=16)
    ap.add_argument("--channels", type=str, default="pre_post")
    ap.add_argument("--xai-subsample", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--effect-size", type=float, default=1.2)
    ap.add_argument("--out-dir", type=Path, default=TASK_FOLDER)
    ap.add_argument("--realistic", action="store_true",
                    help="Stream A realism variant — Duke-like clinical marginals (SYN-A-* IDs, "
                         "distinct manifest hash); identical plumbing verdict, demo-grade cohort.")
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    git_commit = git_commit_short()
    prefix = "realistic_" if args.realistic else ""  
    manifests = {}
    for stream_name in ("negative_control", "positive_control"):
        res = run_stream(
            stream_name, n=args.n_patients, seed=args.seed, cube_size=args.cube_size,
            channels=args.channels, effect_size=args.effect_size, xai_subsample=args.xai_subsample,
            batch_size=args.batch_size, out_dir=args.out_dir, git_commit=git_commit,
            realistic=args.realistic,
        )
        out_path = args.out_dir / f"e2e_synthetic_{prefix}{stream_name}_plumbing_smoke.json"
        out_path.write_text(json.dumps(res["report"], indent=2))
        manifests[stream_name] = res["manifest"]
        print(_consort_line(stream_name, res))  
        print(f"  wrote {out_path}")  

    manifest_path = args.out_dir / f"e2e_synthetic_{prefix}cohort_manifest_plumbing_smoke.json"
    manifest_path.write_text(json.dumps(manifests, indent=2))
    print(f"  wrote {manifest_path}")  
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
