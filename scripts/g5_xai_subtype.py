
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

from pinksight.data.annotation_boxes import load_boxes  
from pinksight.data.dataset import NpyVolumeDataset  
from pinksight.models.heads import SubtypeClassifier  
from pinksight.models.mri_encoder import MriEncoder  
from pinksight.xai.faithfulness import (  
    box_to_cam_mask,
    iou,
    pointing_game,
    randomization_test,
)
from pinksight.xai.saliency import grad_cam_3d, randomize_weights  

CHANNELS = "first_post"
SPATIAL = (96, 96, 96)
DEPTH = 18
FREEZE_BN = False
N_SPLITS = 5
ENCODER_AUROC_FROM_TRAINING = 0.5008  
WEIGHTS_DIR = ROOT / "reports/G5_xai/weights"
OUT_DIR = ROOT / "reports/G5_xai"
_LABEL = {"luminal_like": 0, "tnbc": 1}


def build_items(manifest: Path, proc_dir: Path) -> list[tuple[str, int]]:
    import pandas as pd

    m = pd.read_csv(manifest)
    excl_p = proc_dir / "_phase_stack_exclusions.tsv"
    excl = set(pd.read_csv(excl_p, sep="\t")["patient_id"]) if excl_p.exists() else set()
    return [
        (r.patient_id, _LABEL[r.subtype])
        for r in m[m["split"] == "dev"].itertuples()
        if r.subtype in _LABEL
        and r.patient_id not in excl
        and (proc_dir / f"{r.patient_id}.npy").exists()
    ]


def _oof_folds(items: list[tuple[str, int]], seed: int):
    from sklearn.model_selection import StratifiedGroupKFold

    pids = [p for p, _ in items]
    y = np.array([lab for _, lab in items])
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    for f, (_tr, te) in enumerate(cv.split(np.zeros(len(items)), y, pids)):
        yield f, te


def _load_model(ckpt_path: Path, device: str) -> SubtypeClassifier:
    model = SubtypeClassifier(MriEncoder(in_channels=1, depth=DEPTH, freeze_bn=FREEZE_BN))
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)  
    return model.to(device).eval()


def _subtype_counterfactual_flip(
    model: SubtypeClassifier, volume: torch.Tensor, label: int, device: str,
    n_steps: int = 20, lr: float = 0.05, lambda_l1: float = 0.05,
) -> bool:
    base = volume.detach().clone().to(device)
    delta = torch.zeros_like(base, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    with torch.no_grad():
        logit0 = float(model(base)[0, 0])
    push_positive = label == 1  
    for p in model.parameters():
        p.requires_grad_(False)
    try:
        for step in range(n_steps):
            opt.zero_grad()
            logit = model(base + delta)[0, 0]
            margin = (-logit if push_positive else logit)  
            loss = torch.relu(margin + 1.0) + lambda_l1 * delta.abs().mean()
            loss.backward()
            opt.step()
            if step % 5 == 4:
                with torch.no_grad():
                    lcheck = float(model(base + delta)[0, 0])
                if bool((lcheck > 0) != (logit0 > 0)):
                    return True
    finally:
        for p in model.parameters():
            p.requires_grad_(True)
    with torch.no_grad():
        logit1 = float(model(base + delta)[0, 0])
    result = bool((logit1 > 0) != (logit0 > 0))  
    del opt, delta, base
    return result


def _randomization_verdict(
    model: SubtypeClassifier, volume: torch.Tensor, cam_trained: np.ndarray,
    n_seeds: int = 3, thresh: float = 0.50,
) -> dict:
    rels = []
    cpu_model = model.to("cpu")
    cpu_vol = volume.detach().cpu()
    for s in range(n_seeds):
        rnd = randomize_weights(cpu_model, seed=s)
        cam_rnd = grad_cam_3d(rnd, cpu_vol, rnd.encoder.backbone.layer4, mode="hirescam")
        rels.append(randomization_test(cam_trained, cam_rnd)["rel_drop"])
    model.to(volume.device)  
    mean_rel = float(np.mean(rels))
    return {
        "per_seed_rel_drop": [round(x, 4) for x in rels],
        "mean_randomization_rel_drop": round(mean_rel, 4),
        "randomization_verdict": "PASS" if mean_rel > thresh else "FAIL",
    }


def _partial_path(seed: int) -> Path:
    return OUT_DIR / f"partial_s{seed}.jsonl"


def _load_partial(seed: int) -> tuple[dict, dict | None]:
    path = _partial_path(seed)
    done_pids: dict = {}
    randomization: dict | None = None
    if not path.exists():
        return done_pids, randomization
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("_type") == "randomization":
            randomization = rec["data"]
        else:
            done_pids[rec["pid"]] = {"iou": rec["iou"], "hit": rec["hit"], "flip": rec["flip"]}
    return done_pids, randomization


def run_seed(
    seed: int, items: list[tuple[str, int]], boxes: dict, proc_dir: Path, device: str,
    save_cams: bool, save_panels: bool, max_patients: int = 0,
) -> dict:
    cam_dir = OUT_DIR / "cams"
    if save_cams:
        cam_dir.mkdir(parents=True, exist_ok=True)

    done_pids, randomization = _load_partial(seed)
    n_resumed = len(done_pids)
    if n_resumed:
        print(f"  [resume] seed {seed}: loaded {n_resumed} already-scored patients from partial JSONL")

    ious_loaded = [v["iou"] for v in done_pids.values()]
    hits_loaded = [v["hit"] for v in done_pids.values()]
    flips_loaded = [v["flip"] for v in done_pids.values()]

    ious_new, hits_new, flips_new = [], [], []
    n_newly_scored = 0
    panel_saved = set()
    partial_fh = _partial_path(seed).open("a")

    try:
        for fold, te in _oof_folds(items, seed):
            ckpt = WEIGHTS_DIR / f"model_s{seed}f{fold}.pt"
            if not ckpt.exists():
                raise FileNotFoundError(f"missing weight file {ckpt} — cannot score fold {fold}")
            model = _load_model(ckpt, device)
            ds = NpyVolumeDataset([items[i] for i in te], proc_dir, CHANNELS, SPATIAL, augment=False,
                                  crop_mode="none")  
            for k in range(len(ds)):
                x, _y, pid = ds[k]                       
                if pid not in boxes:                      
                    continue
                label = dict(items)[pid]

                if pid in done_pids:
                    continue

                volume = x.unsqueeze(0).to(device)       

                cam_file = cam_dir / f"cam_s{seed}_{pid}.npy"
                if cam_file.exists():
                    cam = np.load(cam_file)
                else:
                    cam = grad_cam_3d(model, volume, model.encoder.backbone.layer4, target=None,
                                     mode="hirescam")
                    if save_cams:
                        np.save(cam_file, cam.astype(np.float32))

                native_shape = np.load(proc_dir / f"{pid}.npy").shape[1:]   
                box_mask = box_to_cam_mask(boxes[pid], native_shape, cam_shape=SPATIAL)
                pt_iou = iou(cam, box_mask)
                pt_hit = bool(pointing_game(cam, box_mask))
                pt_flip = _subtype_counterfactual_flip(model, volume, label, device)

                if randomization is None:
                    randomization = _randomization_verdict(model, volume, cam)
                    partial_fh.write(json.dumps({"_type": "randomization", "data": randomization}) + "\n")
                    partial_fh.flush()

                partial_fh.write(json.dumps({"pid": pid, "iou": pt_iou,
                                             "hit": pt_hit, "flip": pt_flip}) + "\n")
                partial_fh.flush()

                ious_new.append(pt_iou)
                hits_new.append(pt_hit)
                flips_new.append(pt_flip)
                n_newly_scored += 1

                del volume
                if n_newly_scored % 10 == 0:
                    torch.cuda.empty_cache()

                if max_patients > 0 and (n_resumed + n_newly_scored) >= max_patients:
                    break

                if save_panels and fold not in panel_saved:  
                    panel_box_mask = box_to_cam_mask(
                        boxes[pid], np.load(proc_dir / f"{pid}.npy").shape[1:], cam_shape=SPATIAL
                    )
                    _save_panel(seed, fold, pid, x.numpy()[0], cam, panel_box_mask)
                    panel_saved.add(fold)

            del model
            torch.cuda.empty_cache()
            if max_patients > 0 and (n_resumed + n_newly_scored) >= max_patients:
                break
    finally:
        partial_fh.close()

    ious = ious_loaded + ious_new
    hits = hits_loaded + hits_new
    flips = flips_loaded + flips_new
    n_scored = len(ious)

    if n_scored == 0:
        raise RuntimeError(f"seed {seed}: scored 0 patients — no OOF patient had a box annotation?")
    if randomization is None:  
        randomization = {"per_seed_rel_drop": [], "mean_randomization_rel_drop": 0.0,
                         "randomization_verdict": "FAIL"}
    print(f"  [seed {seed}] resumed={n_resumed} newly-scored={n_newly_scored} total={n_scored}")
    return {
        "iou": round(float(np.mean(ious)), 4),
        "pointing_game_score": round(float(np.mean(hits)), 4),
        "cf_flip_rate": round(float(np.mean(flips)), 4),
        "randomization_verdict": randomization["randomization_verdict"],
        "mean_randomization_rel_drop": randomization["mean_randomization_rel_drop"],
        "randomization_per_seed_rel_drop": randomization["per_seed_rel_drop"],
        "n_scored": n_scored,
        "iou_std": round(float(np.std(ious)), 4),
    }


def _save_panel(seed, fold, pid, vol_ch, cam, box_mask) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return  
    z = vol_ch.shape[2] // 2
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.4))
    ax[0].imshow(vol_ch[:, :, z], cmap="gray"); ax[0].set_title(f"{pid} first_post z={z}")
    ax[1].imshow(cam[:, :, z], cmap="jet", vmin=0, vmax=1); ax[1].set_title("HiResCAM")
    ax[2].imshow(vol_ch[:, :, z], cmap="gray")
    ax[2].imshow(np.ma.masked_where(~box_mask[:, :, z], box_mask[:, :, z]),
                 cmap="autumn", alpha=0.4); ax[2].set_title("Duke box")
    for a in ax:
        a.axis("off")
    fig.suptitle(f"G5 Leg-3 XAI (null encoder, AUROC {ENCODER_AUROC_FROM_TRAINING}) — s{seed}f{fold}",
                 fontsize=9)
    fig.tight_layout()
    panel_dir = OUT_DIR / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(panel_dir / f"panel_s{seed}f{fold}_{pid}.png", dpi=90)
    plt.close(fig)


def main() -> None:
    import time

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0],
                    help="seeds to score (default: 0 — the gate-closing OOF run; add 1 2 for bonus)")
    ap.add_argument("--manifest", type=Path, default=ROOT / "data/manifest_v1.csv")
    ap.add_argument("--proc-dir", type=Path, default=ROOT / "data/processed")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--no-cams", dest="save_cams", action="store_false",
                    help="skip per-patient CAM .npy dumps (default: save)")
    ap.add_argument("--no-panels", dest="save_panels", action="store_false",
                    help="skip saliency PNG panels (default: save)")
    ap.add_argument("--max-patients", type=int, default=0,
                    help="[DEBUG ONLY] stop after N total patients per seed (0=full run, default)")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "metrics.json")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    torch.use_deterministic_algorithms(True, warn_only=True)

    items = build_items(args.manifest, args.proc_dir)
    boxes = load_boxes()
    print(f"[G5 leg3 XAI] device={device} n_dev={len(items)} seeds={args.seeds} "
          f"boxes={len(boxes)} encoder_auroc={ENCODER_AUROC_FROM_TRAINING} (leak-free null)")

    t0 = time.time()
    per_seed = {}
    for seed in args.seeds:
        r = run_seed(seed, items, boxes, args.proc_dir, device, args.save_cams, args.save_panels,
                     max_patients=args.max_patients)
        per_seed[seed] = r
        print(f"  seed {seed}: iou={r['iou']} pointing={r['pointing_game_score']} "
              f"cf_flip={r['cf_flip_rate']} randomization={r['randomization_verdict']} "
              f"(mean_rel_drop {r['mean_randomization_rel_drop']}) n={r['n_scored']}")
    wall_s = round(time.time() - t0, 1)

    gate = per_seed[args.seeds[0]]  
    doc = {
        "iou": gate["iou"],
        "pointing_game_score": gate["pointing_game_score"],
        "cf_flip_rate": gate["cf_flip_rate"],
        "randomization_verdict": gate["randomization_verdict"],
        "device": device,
        "n_scored": gate["n_scored"],
        "seed_used": args.seeds[0],
        "encoder_auroc_from_training": ENCODER_AUROC_FROM_TRAINING,
        "mean_randomization_rel_drop": gate["mean_randomization_rel_drop"],
        "randomization_per_seed_rel_drop": gate["randomization_per_seed_rel_drop"],
        "iou_std": gate["iou_std"],
        "wall_clock_s": wall_s,
        "seeds_scored": list(args.seeds),
        "per_seed": {str(s): per_seed[s] for s in args.seeds},
        "config": {
            "channels": CHANNELS, "spatial": list(SPATIAL), "depth": DEPTH,
            "freeze_bn": FREEZE_BN, "crop_mode": "none",
            "cam_mode": "hirescam", "target_layer": "encoder.backbone.layer4",
        },
        "gate_targets": {"iou": 0.30, "pointing": 0.70},
        "claim_ledger": (
            "Counterfactual input-sensitivity + Grad-CAM localisation for SUBTYPE characterisation "
            "on a leak-free NULL encoder (AUROC 0.5008 ≈ shuffle). A low/diffuse IoU is the EXPECTED "
            "corroboration of the Duke imaging→subtype information ceiling — reported as-measured, "
            "NOT tuned toward 0.30. NEVER 'imaging works' / kinetics / early-detection / "
            "cross-institution generalisation (LOCK-1). LOCK-1/LOCK-2 unchanged."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"[G5 leg3 XAI] wrote {args.out}  ({wall_s}s wall)")
    print(f"  HEADLINE (seed {args.seeds[0]}): IoU {doc['iou']} | pointing {doc['pointing_game_score']} "
          f"| cf_flip {doc['cf_flip_rate']} | randomization {doc['randomization_verdict']}")


if __name__ == "__main__":
    main()
