"""fastMRI-NYU STANDALONE encoder training (ADR-0016, RATIFIED) — H-char + H5, NYU-INTERNAL only.

NYU-ONLY. No Duke data, no Duke gradient. Reports NYU-internal AUROC + DeLong CI + ECE + shuffle
sentinel + >=3 seeds (H-char) and MAE + bootstrap CI (H5 chronological age). Go/no-go binds the DeLong
LOWER BOUND (characterisation min met only if lower bound > 0.60; n=50 test / 18 positive => wide CI).
NEVER juxtaposes an NYU number with a Duke number.

Claim guards asserted at runtime (ADR-0016): normals-quarantine absent from every batch, patient-disjoint
shipped 240/60 split, images-only encoder inputs.

    PYTHONPATH=src .venv/bin/python scripts/train_fastmri_nyu.py --smoke       # overfit probe (BN-alive)
    PYTHONPATH=src .venv/bin/python scripts/train_fastmri_nyu.py               # full 3-seed H-char + H5
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from monai.utils import set_determinism
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

from pinksight.data.dataset import NpyVolumeDataset, n_channels
from pinksight.data.fastmri_nyu import (
    age_items,
    assert_images_only,
    assert_no_normals,
    assert_patient_disjoint,
    hchar_items,
)
from pinksight.eval.metrics import classification_metrics
from pinksight.metrics import delong_ci
from pinksight.models.heads import SubtypeClassifier
from pinksight.models.mri_encoder import MriEncoder
from pinksight.seed import set_seed
from pinksight.train.loop import TrainCfg, train_model

PROC = Path("data/fastmri_processed_nyu")
# [lesion-crop plan] --crop-mode/--mask-dir/--proc-dir reassign these module globals in main() (the
# existing PROC-as-global idiom, read by _loader + _cached). Defaults preserve R0-R3 byte-for-byte:
# CROP_MODE="none" makes crop_size/mask_dir inert inside NpyVolumeDataset (the resize branch runs).
MASK_DIR = Path("data/fastmri_processed_nyu_masks")
CROP_MODE = "none"
TASK = Path("process/general-plans/active/fastmri-encoder-deferred-fusion_04-08-26")
OUT = TASK / "fastmri_nyu_results.json"
CHAR_MIN_AUROC = 0.75           # characterisation min target
GO_NOGO_LB = 0.60               # go/no-go binds the DeLong LOWER BOUND > 0.60 (red-team Fix #5)


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)  # noqa: T201


def _device(choice: str) -> str:
    return ("cuda" if torch.cuda.is_available() else "cpu") if choice == "auto" else choice


def _cached(items: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Keep only items whose processed cube exists on disk (resumable / partial-preprocess safe)."""
    return [(p, y) for p, y in items if (PROC / f"{p}.npy").exists()]


def _loader(items, channels, spatial, batch, augment, shuffle) -> DataLoader:
    # crop_mode/crop_size/mask_dir are inert when CROP_MODE=="none" (dataset takes the resize branch),
    # so a no-flag run is byte-identical to R0-R3. In "lesion" mode the dataset derives the tight
    # heuristic-mask box and resamples to crop_size^3 (== --spatial), so the fixed grid is unchanged.
    ds = NpyVolumeDataset(items, proc_dir=PROC, channels=channels,
                          spatial_size=(spatial,) * 3, augment=augment,
                          crop_mode=CROP_MODE, crop_size=spatial, mask_dir=MASK_DIR)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=2, drop_last=False)


def _inner_split(train_items, seed, val_frac=0.2):
    labels = [int(y) for _, y in train_items]
    tr, va = train_test_split(train_items, test_size=val_frac, random_state=seed,
                              stratify=labels if len(set(labels)) > 1 else None)
    assert_patient_disjoint([p for p, _ in tr], [p for p, _ in va])
    return tr, va


# --- H-char (malig vs benign) --------------------------------------------------------------------
def _run_hchar_seed(seed, train_items, test_items, channels, spatial, batch, epochs, device,
                    shuffle_labels=False, medicalnet=None, backbone_lr=None, head_lr=None,
                    scheduler="none") -> dict[str, float]:
    set_seed(seed)
    set_determinism(seed)
    tr, va = _inner_split(train_items, seed)
    if shuffle_labels:  # leakage sentinel: permute train+val labels; test AUROC must collapse to ~0.5
        rng = np.random.default_rng(seed)
        allp = tr + va
        ys = rng.permutation([y for _, y in allp]).tolist()
        allp = [(p, int(y)) for (p, _), y in zip(allp, ys)]
        tr, va = allp[: len(tr)], allp[len(tr):]
    nch = n_channels(channels)
    npos = sum(y for _, y in tr) or 1
    pos_weight = (len(tr) - npos) / npos
    cfg = TrainCfg(epochs=epochs, batch_size=batch, device=device, loss="focal",
                   amp=False, grad_clip=1.0, patience=10,  # fp32 + clip: dodge the fp16-NaN / freeze-bn burns
                   # [medicalnet-backbone] LR-split + cosine, flag-gated; None/None/"none" == byte-identical.
                   backbone_lr=backbone_lr, head_lr=head_lr, scheduler=scheduler)
    # [medicalnet-backbone] pretrained-init, flag-gated; medicalnet=None == scratch (byte-identical).
    model = SubtypeClassifier(MriEncoder(in_channels=nch, depth=18, freeze_bn=False,
                                         medicalnet_weights=medicalnet))
    tr_loader = _loader(tr, channels, spatial, batch, augment=True, shuffle=True)
    va_loader = _loader(va, channels, spatial, batch, augment=False, shuffle=False)
    te_loader = _loader(test_items, channels, spatial, batch, augment=False, shuffle=False)
    # images-only guard on the first real batch (ADR-0016 Fix #6)
    xb, _, _ = next(iter(tr_loader))
    assert_images_only(xb, nch)
    _, oof_probs, oof_pids = train_model(model, tr_loader, va_loader, cfg, pos_weight,
                                          log_tag=f"hchar_s{seed}", oof_loader=te_loader)
    return {pid: float(pr) for pid, pr in zip(oof_pids, oof_probs)}


def run_hchar(train_items, test_items, channels, spatial, batch, epochs, device, seeds,
              medicalnet=None, backbone_lr=None, head_lr=None, scheduler="none") -> dict:
    assert_no_normals([p for p, _ in train_items + test_items])
    assert_patient_disjoint([p for p, _ in train_items], [p for p, _ in test_items])
    per_seed, prob_by_seed = {}, {}
    for s in seeds:
        log(f"H-char seed {s}: train={len(train_items)} test={len(test_items)}")
        probs = _run_hchar_seed(s, train_items, test_items, channels, spatial, batch, epochs, device,
                                medicalnet=medicalnet, backbone_lr=backbone_lr, head_lr=head_lr,
                                scheduler=scheduler)
        pids = sorted(probs)  # fixed patient order
        tmap = {p: int(l) for p, l in test_items}
        yv = np.array([tmap[p] for p in pids])
        pv = np.array([probs[p] for p in pids])
        m = classification_metrics(yv, pv)
        per_seed[str(s)] = {"auroc": m["auroc"], "ece": m["ece"], "n": m["n"], "prevalence": m["prevalence"]}
        prob_by_seed[s] = {p: probs[p] for p in pids}
        log(f"  seed {s}: AUROC {m['auroc']['value']} CI {m['auroc']['ci95']} ECE {m['ece']['value']}")

    # ensemble mean-probability across seeds on the sealed 50 -> headline DeLong CI (go/no-go)
    pids = sorted(prob_by_seed[seeds[0]])
    tmap = {p: int(l) for p, l in test_items}
    yv = np.array([tmap[p] for p in pids])
    ens = np.array([np.mean([prob_by_seed[s][p] for s in seeds]) for p in pids])
    em = classification_metrics(yv, ens)
    auc, lo, hi = delong_ci(yv, ens)
    aurocs = [per_seed[str(s)]["auroc"]["value"] for s in seeds]

    # leakage sentinel (1 seed, shuffled labels)
    log("H-char shuffle sentinel (1 seed, permuted labels)")
    sh_probs = _run_hchar_seed(seeds[0], train_items, test_items, channels, spatial, batch,
                               max(5, epochs // 4), device, shuffle_labels=True,
                               medicalnet=medicalnet, backbone_lr=backbone_lr, head_lr=head_lr,
                               scheduler=scheduler)  # sentinel uses the SAME init/recipe as its cell
    shp = np.array([sh_probs[p] for p in pids])
    sh_auc = float(delong_ci(yv, shp)[0])

    go = bool(lo > GO_NOGO_LB)
    return {
        "task": "fastMRI-NYU malignant-vs-benign characterisation (NYU-INTERNAL)",
        "cohort": {"train": len(train_items), "test": len(test_items),
                   "test_positive": int(yv.sum()), "test_prevalence": round(float(yv.mean()), 4)},
        "seeds": list(seeds),
        "per_seed": per_seed,
        "auroc_across_seeds": {"mean": round(float(np.mean(aurocs)), 4),
                               "std": round(float(np.std(aurocs)), 4), "values": aurocs},
        "ensemble_headline": {
            "auroc": round(auc, 4), "delong_ci95": [round(lo, 4), round(hi, 4)],
            "ece": em["ece"]["value"], "method": "mean probability across seeds on the sealed 50",
        },
        "leakage_shuffle_sentinel": {"auroc": round(sh_auc, 4), "passes": bool(sh_auc < 0.60)},
        "go_no_go": {
            "rule": f"characterisation min ({CHAR_MIN_AUROC}) met ONLY if DeLong lower bound > {GO_NOGO_LB}",
            "delong_lower_bound": round(lo, 4), "verdict": "GO" if go else "NO-GO",
            "note": "NYU-internal characterisation; a NULL is honestly reportable and moves NO LOCK.",
        },
    }


# --- H5 (chronological age regression, Huber) ----------------------------------------------------
class _AgeNet(nn.Module):
    def __init__(self, nch: int, medicalnet=None) -> None:
        super().__init__()
        # [medicalnet-backbone] pretrained-init threaded for consistency (Section 2 decision 1);
        # medicalnet=None == scratch (byte-identical). H5 keeps its own flat-LR loop (no LR-split flags).
        self.encoder = MriEncoder(in_channels=nch, depth=18, freeze_bn=False,
                                  medicalnet_weights=medicalnet)
        self.head = nn.Linear(self.encoder.embed_dim, 1)

    def forward(self, x):
        return self.head(self.encoder(x))


def _mae_ci(y_true, y_pred, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    m = len(y_true)
    vals = [float(np.mean(np.abs(y_true[i] - y_pred[i])))
            for i in (rng.integers(0, m, m) for _ in range(n))]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(np.mean(np.abs(y_true - y_pred))), float(lo), float(hi)


def run_h5(train_items, test_items, channels, spatial, batch, epochs, device, medicalnet=None,
           seed=0) -> dict:
    assert_no_normals([p for p, _ in train_items + test_items])  # H5 also excludes normals (interlock)
    assert_patient_disjoint([p for p, _ in train_items], [p for p, _ in test_items])
    set_seed(seed); set_determinism(seed)
    nch = n_channels(channels)
    ages = np.array([a for _, a in train_items], float)
    mu, sd = float(ages.mean()), float(ages.std() + 1e-6)  # LOCK-2: standardize on TRAIN only
    tr = [(p, (a - mu) / sd) for p, a in train_items]
    tr_in, va_in = train_test_split(tr, test_size=0.2, random_state=seed)
    model = _AgeNet(nch, medicalnet=medicalnet).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    huber = nn.SmoothL1Loss()
    tl = _loader(tr_in, channels, spatial, batch, augment=True, shuffle=True)
    vl = _loader(va_in, channels, spatial, batch, augment=False, shuffle=False)
    te = _loader(test_items, channels, spatial, batch, augment=False, shuffle=False)
    xb, _, _ = next(iter(tl)); assert_images_only(xb, nch)

    def eval_mae(loader, destd):
        model.eval(); preds, ys, pids = [], [], []
        with torch.no_grad():
            for x, y, pid in loader:
                p = model(x.to(device)).float().cpu().numpy().ravel()
                preds.append(p); ys.append(y.numpy().ravel()); pids.extend(pid)
        pr = np.concatenate(preds); yv = np.concatenate(ys)
        if destd:
            pr = pr * sd + mu
        return pr, yv, pids

    best, best_state, stale = np.inf, None, 0
    for ep in range(epochs):
        model.train()
        for x, y, _ in tl:
            opt.zero_grad()
            loss = huber(model(x.to(device)).squeeze(1), y.to(device).float())
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        vp, vy, _ = eval_mae(vl, destd=False)
        mae = float(np.mean(np.abs(vp - vy)))
        if mae < best:
            best, stale, best_state = mae, 0, copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= 10:
                break
    if best_state:
        model.load_state_dict(best_state)
    tp, _, tpid = eval_mae(te, destd=True)
    tmap = {p: a for p, a in test_items}
    ty = np.array([tmap[p] for p in tpid])
    mae, lo, hi = _mae_ci(ty, tp)
    log(f"H5 age: MAE {mae:.2f} yrs CI [{lo:.2f},{hi:.2f}] (n={len(ty)})")
    return {
        "task": "fastMRI-NYU chronological-age regression (Huber; forward-register; NYU-INTERNAL)",
        "n_test": int(len(ty)), "mae_years": round(mae, 3), "mae_ci95": [round(lo, 3), round(hi, 3)],
        "note": "CHRONOLOGICAL age only — never a biological-age / physiological-progression proxy.",
    }


# --- smoke (overfit probe: prove the loop CAN memorize -> BN not dead, no freeze_bn artifact) ------
def run_smoke(channels, spatial, device) -> int:
    cached_tr = _cached(hchar_items()["train"])
    # BALANCE a tiny set so overfitting is MEANINGFUL. A single-class subset (e.g. the first cached
    # patients 001-008 are all benign on this cohort) makes train-AUROC undefined (nan) AND a constant
    # prediction optimal (spread ~0) -> a FALSE "BN-dead" smoke FAIL that would abort a healthy run.
    # Take up to 6 malignant + 6 benign; require >=3 of each so BOTH AUROC and spread are informative.
    malig = [(p, y) for p, y in cached_tr if y == 1]
    benign = [(p, y) for p, y in cached_tr if y == 0]
    items = malig[:6] + benign[:6]
    if len(malig) < 3 or len(benign) < 3:
        log(f"SMOKE BLOCKED: need >=3 malignant AND >=3 benign cached H-char cubes in {PROC} "
            f"(have malig={len(malig)}, benign={len(benign)}) — preprocess more patients first.")
        return 2
    set_seed(0); set_determinism(0)
    nch = n_channels(channels)
    model = SubtypeClassifier(MriEncoder(in_channels=nch, depth=18, freeze_bn=False))
    loader = _loader(items, channels, spatial, batch=4, augment=False, shuffle=True)
    xb, _, _ = next(iter(loader)); assert_images_only(xb, nch)
    cfg = TrainCfg(epochs=40, batch_size=4, device=device, loss="bce", amp=False, patience=40)
    npos = sum(y for _, y in items) or 1
    # train + evaluate ON THE SAME tiny set (memorization test, NOT a generalization claim)
    _, probs, pids = train_model(model, loader, loader, cfg, (len(items) - npos) / npos,
                                 log_tag="smoke", oof_loader=loader)
    tmap = {p: int(y) for p, y in items}
    y = np.array([tmap[p] for p in pids]); pv = np.array(probs)
    auc = float(delong_ci(y, pv)[0]) if len(set(y.tolist())) > 1 else float("nan")
    spread = float(pv.max() - pv.min())
    log(f"SMOKE: train-set AUROC {auc:.3f} (must be high), prob spread {spread:.3f} (must be >0.05 -> BN alive)")
    ok = (auc > 0.85 or np.isnan(auc)) and spread > 0.05
    log("SMOKE PASS — loop overfits, BN/GroupNorm alive (no freeze_bn dead-output)." if ok
        else "SMOKE FAIL — loop cannot memorize; BN may be dead (freeze_bn artifact). DO NOT TRUST NULLS.")
    return 0 if ok else 1


def main() -> None:
    global PROC, MASK_DIR, CROP_MODE  # _loader/_cached read these; reassigned from args below
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--channels", default="pre_post", choices=["first_post", "pre_post", "fixed4", "subtraction", "kinetic"])
    ap.add_argument("--spatial", type=int, default=64, help="cube edge at train time (VRAM: 64 fits 8GB)")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--hchar-only", action="store_true")
    # [lesion-crop plan] additive, backward-compatible: all three default to the R0-R3 behavior.
    ap.add_argument("--crop-mode", default="none", choices=["none", "lesion"],
                    help="none=resize whole cube (R0-R3 default, byte-identical); lesion=heuristic-mask "
                         "lesion crop via NpyVolumeDataset (Rung 0 reads --mask-dir on the 96^3 grid)")
    ap.add_argument("--mask-dir", default=str(MASK_DIR),
                    help="heuristic/predicted mask dir consumed only when --crop-mode lesion")
    ap.add_argument("--proc-dir", default=str(PROC),
                    help="processed-cube cache dir (Rung 1 points this at a pre-cropped lesion cache)")
    # [medicalnet-backbone] additive, backward-compatible (mirrors scripts/train_imaging_mvp.py flag
    # names/types/choices). Omitting all four is byte-identical to the R0-R3/Rung0/Rung1 behavior
    # (scratch init + single-group AdamW at lr=1e-3, no schedule).
    ap.add_argument("--medicalnet", type=Path, default=None,
                    help="MedicalNet pretrained .pth for transfer init; omit for random (scratch) init")
    ap.add_argument("--backbone-lr", type=float, default=None,
                    help="fine-tune LR for the pretrained backbone (params NOT under .head.); when set, "
                         "splits AdamW into backbone/head param groups. Omit for the single-group default "
                         "at the base lr")
    ap.add_argument("--head-lr", type=float, default=None,
                    help="LR for the fresh head (.head. params); only used with --backbone-lr; "
                         "defaults to the base lr when omitted")
    ap.add_argument("--scheduler", choices=["none", "cosine"], default="none",
                    help="LR schedule: none (default, unchanged) or cosine (CosineAnnealingLR "
                         "T_max=epochs, stepped per epoch — no warmup)")
    args = ap.parse_args()
    # [medicalnet-backbone] fail-fast: a missing/typo'd --medicalnet path must NOT silently fall back to
    # scratch init and waste a full ~14 min GPU run under a mislabeled cell. Fire BEFORE any data/GPU work.
    if args.medicalnet is not None and not Path(args.medicalnet).exists():
        raise SystemExit(
            f"BLOCKED: --medicalnet path {args.medicalnet} does not exist — refusing a silent "
            "scratch-init fallback. Fix the path or omit --medicalnet.")
    # reassign the module globals _loader/_cached read (defaults leave them unchanged -> byte-identical).
    PROC = Path(args.proc_dir)
    MASK_DIR = Path(args.mask_dir)
    CROP_MODE = args.crop_mode
    device = _device(args.device)
    log(f"fastMRI-NYU train: channels={args.channels} spatial={args.spatial} batch={args.batch} "
        f"epochs={args.epochs} seeds={args.seeds} device={device} crop_mode={CROP_MODE} proc_dir={PROC}")

    if args.smoke:
        raise SystemExit(run_smoke(args.channels, args.spatial, device))

    # [medicalnet-backbone] init-mode disclosure (Section 2 decision 4). The mechanical CONFIRMATION the
    # pretrained load actually applied is MriEncoder._load_medicalnet's own "N missing / M unexpected
    # keys" warning — captured only when the run's stderr is teed (every cell command uses 2>&1 | tee).
    log(f"init: {'MedicalNet-transfer (' + str(args.medicalnet) + ')' if args.medicalnet else 'scratch'}"
        + (f" | LR-split backbone={args.backbone_lr} head={args.head_lr} scheduler={args.scheduler}"
           if args.backbone_lr is not None else " | flat LR (single-group AdamW)"))

    hc = hchar_items()
    tr, te = _cached(hc["train"]), _cached(hc["test"])
    if len(tr) < 20 or len(te) < 10:
        log(f"BLOCKED: insufficient processed cubes (train={len(tr)}/199 test={len(te)}/50). "
            f"Run scripts/preprocess_fastmri_nyu.py first.")
        raise SystemExit(2)

    results = {
        "adr": "ADR-0016 (RATIFIED 2026-08-04) — standalone fastMRI-NYU encoder; NYU-INTERNAL only",
        "reporting_firewall": "No NYU number is juxtaposed with any Duke number. ADR-0008 Duke "
                              "imaging-fusion null (clinical 0.708; hierarchical #4 0.599 [0.495,0.610]) "
                              "co-locates ONLY as a firewall reminder, never as comparison.",
        "recipe": {"channels": args.channels, "spatial": args.spatial, "batch": args.batch,
                   "epochs": args.epochs, "norm": "BatchNorm (fp32, grad-clip 1.0)", "loss": "focal",
                   "crop_mode": CROP_MODE, "proc_dir": str(PROC),
                   "mask_dir": str(MASK_DIR) if CROP_MODE == "lesion" else None,
                   "preprocess": ("resize-cube + train-only Nyul (N4 dropped, no lesion crop — see REPORT)"
                                  if CROP_MODE == "none"
                                  else f"heuristic-mask lesion crop on {PROC.name} + train-only Nyul (see REPORT)")},
        "hchar": run_hchar(tr, te, args.channels, args.spatial, args.batch, args.epochs, device, args.seeds,
                           medicalnet=args.medicalnet, backbone_lr=args.backbone_lr,
                           head_lr=args.head_lr, scheduler=args.scheduler),
    }
    if not args.hchar_only:
        ag = age_items()
        results["h5_age"] = run_h5(_cached(ag["train"]), _cached(ag["test"]),
                                   args.channels, args.spatial, args.batch, args.epochs, device,
                                   medicalnet=args.medicalnet)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2) + "\n")
    hh = results["hchar"]
    log(f"WROTE {OUT}")
    log(f"H-char ensemble AUROC {hh['ensemble_headline']['auroc']} "
        f"CI {hh['ensemble_headline']['delong_ci95']} -> {hh['go_no_go']['verdict']} "
        f"(shuffle {hh['leakage_shuffle_sentinel']['auroc']})")


if __name__ == "__main__":
    main()
