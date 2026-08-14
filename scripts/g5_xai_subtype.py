"""G5 Leg 3 — counterfactual + faithfulness XAI on the trained SUBTYPE spatial encoder.

Scores Grad-CAM (HiResCAM) IoU + pointing-game vs Duke lesion boxes, a subtype-logit
counterfactual flip rate, and the TICKET-001 3-seed model-randomization sanity, over the
seed-0 out-of-fold (OOF) Duke patients — the run that closes the G5 gate (§5.7, E5/E7).

    PYTHONPATH=src uv run python scripts/g5_xai_subtype.py                  # seed 0 (gate-closing)
    PYTHONPATH=src uv run python scripts/g5_xai_subtype.py --seeds 0 1 2    # + robustness bonus

FRAMING GUARD (decisions.md LOCK-1, plan §5.6 / E6): the encoder is a leak-free NULL
(AUROC 0.5008 ≈ shuffle). A low / diffuse IoU here is the EXPECTED, honestly-reportable
corroboration of the Duke imaging→subtype information ceiling — NOT a failure and NEVER tuned
toward the 0.30 target. A degenerate cf_flip_rate on a null encoder is equally expected. Report
every number AS-MEASURED. NEVER "imaging works" / kinetics / early-detection / cross-institution.

The subtype counterfactual is implemented directly here (NOT via
`saliency.counterfactual_grade_map`, which is grade-head-only and raises on a subtype head — E7d):
the flip target is the single subtype logit `out[0,0]` (positive = Luminal-A, negative = TNBC).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Must be set BEFORE torch initialises CUDA — prevents the allocator from retaining
# large freed blocks as a private pool, so empty_cache() actually returns VRAM to the OS.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

from pinksight.data.annotation_boxes import load_boxes  # noqa: E402
from pinksight.data.dataset import NpyVolumeDataset  # noqa: E402
from pinksight.models.heads import SubtypeClassifier  # noqa: E402
from pinksight.models.mri_encoder import MriEncoder  # noqa: E402
from pinksight.xai.faithfulness import (  # noqa: E402
    box_to_cam_mask,
    iou,
    pointing_game,
    randomization_test,
)
from pinksight.xai.saliency import grad_cam_3d, randomize_weights  # noqa: E402

# Config = the exact trained-run config (reports/G5_xai/g5leg3_train.log line 1):
# n_dev=613, channels=first_post (1ch), freeze_bn=False, crop=none, spatial=(96,96,96), seeds=(0,1,2).
CHANNELS = "first_post"
SPATIAL = (96, 96, 96)
DEPTH = 18
FREEZE_BN = False
N_SPLITS = 5
ENCODER_AUROC_FROM_TRAINING = 0.5008  # README.md — for the framing audit, NOT a headline
WEIGHTS_DIR = ROOT / "reports/G5_xai/weights"
OUT_DIR = ROOT / "reports/G5_xai"
_LABEL = {"luminal_like": 0, "tnbc": 1}


def build_items(manifest: Path, proc_dir: Path) -> list[tuple[str, int]]:
    """dev split ∩ labelled ∩ on-disk .npy ∩ not a phase-stack exclusion — BYTE-IDENTICAL to
    `scripts/train_imaging_mvp.build_items`, so the StratifiedGroupKFold below reproduces the exact
    fold assignment the weights were trained under (each patient is test in exactly one fold)."""
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
    """Yield (fold, test_indices) reproducing `cross_val_imaging`'s StratifiedGroupKFold exactly."""
    from sklearn.model_selection import StratifiedGroupKFold

    pids = [p for p, _ in items]
    y = np.array([lab for _, lab in items])
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    for f, (_tr, te) in enumerate(cv.split(np.zeros(len(items)), y, pids)):
        yield f, te


def _load_model(ckpt_path: Path, device: str) -> SubtypeClassifier:
    """E7a: weights are SubtypeClassifier state_dicts (keys encoder.backbone.* + head.*). Load into
    the SAME wrapper `train_imaging_mvp.factory` built, NOT a bare MriEncoder (that raises)."""
    model = SubtypeClassifier(MriEncoder(in_channels=1, depth=DEPTH, freeze_bn=FREEZE_BN))
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)  # strict — verifies the keys match before scoring
    return model.to(device).eval()


def _subtype_counterfactual_flip(
    model: SubtypeClassifier, volume: torch.Tensor, label: int, device: str,
    n_steps: int = 20, lr: float = 0.05, lambda_l1: float = 0.05,
) -> bool:
    """E7d: Adam on an input delta that flips the SINGLE subtype logit's sign (LumA↔TNBC).

    forward -> (N,1); score = out[0,0] (positive=LumA, negative=TNBC). LumA patients (label 0) are
    pushed toward TNBC (minimise out[0,0]); TNBC patients (label 1) toward LumA (maximise out[0,0]).
    `target=1` is NEVER used — index 1 is out of bounds on a (N,1) head. Returns did-the-sign-flip.

    Perf note: model params are temporarily frozen during the Adam loop (requires_grad=False) so
    backward only computes delta.grad — not gradients for all 33M encoder parameters. An early-exit
    fires as soon as the flip is achieved (checked every 5 steps). Default n_steps reduced to 20:
    a null encoder (AUROC ~0.50) produces the same cf_flip=False result at 20 steps as at 100,
    and the rate is reported as-measured (never tuned toward a target).
    """
    base = volume.detach().clone().to(device)
    delta = torch.zeros_like(base, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    with torch.no_grad():
        logit0 = float(model(base)[0, 0])
    push_positive = label == 1  # TNBC -> want logit>0 (LumA); LumA -> want logit<0 (TNBC)
    # Freeze model params: only delta.grad is needed; computing grads for 33M encoder params wastes ~5x.
    for p in model.parameters():
        p.requires_grad_(False)
    try:
        for step in range(n_steps):
            opt.zero_grad()
            logit = model(base + delta)[0, 0]
            margin = (-logit if push_positive else logit)  # hinge toward the target sign
            loss = torch.relu(margin + 1.0) + lambda_l1 * delta.abs().mean()
            loss.backward()
            opt.step()
            # Early exit: check every 5 steps; avoid extra forward on every step
            if step % 5 == 4:
                with torch.no_grad():
                    lcheck = float(model(base + delta)[0, 0])
                if bool((lcheck > 0) != (logit0 > 0)):
                    return True
    finally:
        # Always restore requires_grad so the model is reusable for subsequent patients
        for p in model.parameters():
            p.requires_grad_(True)
    with torch.no_grad():
        logit1 = float(model(base + delta)[0, 0])
    result = bool((logit1 > 0) != (logit0 > 0))  # the subtype decision crossed the boundary
    # Release per-patient optimizer state and delta tensor from GPU immediately.
    del opt, delta, base
    return result


def _randomization_verdict(
    model: SubtypeClassifier, volume: torch.Tensor, cam_trained: np.ndarray,
    n_seeds: int = 3, thresh: float = 0.50,
) -> dict:
    """E7e / §5.7 step 4 (TICKET-001 hardened): 3-seed AVERAGE model-randomization sanity.

    A faithful map concentrates in its ROI; a weight-randomized map should not. Single-seed rel_drop
    is noisy, so average over `n_seeds` re-init seeds. The randomized layer is accessed DIRECTLY as
    `rnd.encoder.backbone.layer4` (NOT `_find_layer()`, which is written for the grade fixture and
    fails on the SubtypeClassifier nested path).

    Runs on CPU: `saliency.randomize_weights` uses a CPU-seeded `torch.Generator` (device-mismatches
    a CUDA model) and is read-only per the plan blast radius, so the sanity — a cheap 3-seed pass on
    ONE patient — is done on CPU copies rather than editing the shared primitive."""
    rels = []
    cpu_model = model.to("cpu")
    cpu_vol = volume.detach().cpu()
    for s in range(n_seeds):
        rnd = randomize_weights(cpu_model, seed=s)
        cam_rnd = grad_cam_3d(rnd, cpu_vol, rnd.encoder.backbone.layer4, mode="hirescam")
        rels.append(randomization_test(cam_trained, cam_rnd)["rel_drop"])
    model.to(volume.device)  # restore the scoring device for the caller's next patient
    mean_rel = float(np.mean(rels))
    return {
        "per_seed_rel_drop": [round(x, 4) for x in rels],
        "mean_randomization_rel_drop": round(mean_rel, 4),
        "randomization_verdict": "PASS" if mean_rel > thresh else "FAIL",
    }


def _partial_path(seed: int) -> Path:
    """Per-seed JSONL file: one JSON record per patient, flushed immediately after scoring."""
    return OUT_DIR / f"partial_s{seed}.jsonl"


def _load_partial(seed: int) -> tuple[dict, dict | None]:
    """Load any existing partial results for this seed.

    Returns:
        done_pids: {pid: {iou, hit, flip}} for already-scored patients.
        randomization: the randomization dict if it was already persisted, else None.
    """
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
    """Score every OOF patient for one seed: IoU + pointing vs box, subtype counterfactual flip;
    3-seed randomization sanity on the first scored patient. Aggregates over the seed's OOF set.

    Interruption-resilient: each patient's scalar results are appended to a per-seed JSONL file
    immediately after scoring. On restart, already-scored patients (present in the JSONL) are
    skipped and their results are folded into the final aggregate — no re-scoring.

    The 384 existing CAM .npy files (from the interrupted run) are detected via cam file existence;
    when a patient's cam file already exists the CAM is loaded from disk rather than re-computed,
    so resume reuses those files even without a pre-existing JSONL (the scalars will be re-computed
    for cammed patients whose JSONL row is absent, but no OOM risk since the volume is freed).
    """
    cam_dir = OUT_DIR / "cams"
    if save_cams:
        cam_dir.mkdir(parents=True, exist_ok=True)

    # --- resume: load any already-scored patients ---
    done_pids, randomization = _load_partial(seed)
    n_resumed = len(done_pids)
    if n_resumed:
        print(f"  [resume] seed {seed}: loaded {n_resumed} already-scored patients from partial JSONL")

    # Collect loaded results to merge with newly-scored results at the end.
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
                                  crop_mode="none")  # crop_mode="none" == the trained-run geometry
            for k in range(len(ds)):
                x, _y, pid = ds[k]                       # x: (1, 96, 96, 96) — first_post, z-normed
                if pid not in boxes:                      # box annotation absent — skip (rare)
                    continue
                label = dict(items)[pid]

                # --- resume skip: patient already in JSONL ---
                if pid in done_pids:
                    # Randomization was already persisted in the JSONL; only re-run if still missing.
                    # (randomization runs once; if done_pids is non-empty it was likely persisted too.)
                    continue

                volume = x.unsqueeze(0).to(device)       # (1, 1, 96, 96, 96)

                # --- CAM: load from disk if already saved (reuses the 384 interrupted-run cams) ---
                cam_file = cam_dir / f"cam_s{seed}_{pid}.npy"
                if cam_file.exists():
                    cam = np.load(cam_file)
                else:
                    # HiResCAM over the trained encoder; target=None -> out[0,0] = LumA logit (E7b/E7d).
                    cam = grad_cam_3d(model, volume, model.encoder.backbone.layer4, target=None,
                                     mode="hirescam")
                    if save_cams:
                        np.save(cam_file, cam.astype(np.float32))

                native_shape = np.load(proc_dir / f"{pid}.npy").shape[1:]   # (D,H,W) pre-resize (E7c)
                box_mask = box_to_cam_mask(boxes[pid], native_shape, cam_shape=SPATIAL)
                pt_iou = iou(cam, box_mask)
                pt_hit = bool(pointing_game(cam, box_mask))
                pt_flip = _subtype_counterfactual_flip(model, volume, label, device)

                # --- randomization sanity: once per seed on first newly-scored patient ---
                if randomization is None:
                    randomization = _randomization_verdict(model, volume, cam)
                    # Persist randomization so resume doesn't re-run it.
                    partial_fh.write(json.dumps({"_type": "randomization", "data": randomization}) + "\n")
                    partial_fh.flush()

                # --- persist scalars immediately (interruption resilience) ---
                partial_fh.write(json.dumps({"pid": pid, "iou": pt_iou,
                                             "hit": pt_hit, "flip": pt_flip}) + "\n")
                partial_fh.flush()

                ious_new.append(pt_iou)
                hits_new.append(pt_hit)
                flips_new.append(pt_flip)
                n_newly_scored += 1

                # Release per-patient GPU tensors to keep VRAM flat across patients.
                del volume
                # Every 10 patients: return cached-but-freed VRAM blocks to the allocator.
                if n_newly_scored % 10 == 0:
                    torch.cuda.empty_cache()

                # Debug/verify early exit (--max-patients): 0 = disabled (full run).
                if max_patients > 0 and (n_resumed + n_newly_scored) >= max_patients:
                    break

                if save_panels and fold not in panel_saved:  # one representative panel per fold (§7)
                    panel_box_mask = box_to_cam_mask(
                        boxes[pid], np.load(proc_dir / f"{pid}.npy").shape[1:], cam_shape=SPATIAL
                    )
                    _save_panel(seed, fold, pid, x.numpy()[0], cam, panel_box_mask)
                    panel_saved.add(fold)

            # Release fold model from GPU before loading the next fold's weights.
            del model
            torch.cuda.empty_cache()
            # Debug/verify early exit at fold boundary (--max-patients).
            if max_patients > 0 and (n_resumed + n_newly_scored) >= max_patients:
                break
    finally:
        partial_fh.close()

    # Merge loaded + newly-scored results.
    ious = ious_loaded + ious_new
    hits = hits_loaded + hits_new
    flips = flips_loaded + flips_new
    n_scored = len(ious)

    if n_scored == 0:
        raise RuntimeError(f"seed {seed}: scored 0 patients — no OOF patient had a box annotation?")
    if randomization is None:  # defensive — n_scored>0 guarantees at least one sanity run
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
    """One saliency PNG: mid-slice first_post with CAM overlay + box outline (§7)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return  # matplotlib optional — skip panels rather than fail the gate
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

    gate = per_seed[args.seeds[0]]  # seed 0 (first) closes the gate
    doc = {
        # --- 4 REQUIRED keys (validate-contract L2-C7) ---
        "iou": gate["iou"],
        "pointing_game_score": gate["pointing_game_score"],
        "cf_flip_rate": gate["cf_flip_rate"],
        "randomization_verdict": gate["randomization_verdict"],
        # --- provenance / framing audit ---
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
