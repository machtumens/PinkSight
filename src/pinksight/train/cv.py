"""P06: patient-disjoint CV for nn.Modules — the locked eval ritual, reused for the imaging arm.

Byte-for-byte the structure of `models/clinical_encoder.py::cross_val_auroc` (SEEDS=(0,1,2),
StratifiedGroupKFold(5), per-fold patient-disjoint assert, pooled-OOF -> per-seed DeLong CI + ECE)
so the imaging metrics.json slots into the exact same schema and comparison table. The only swap is
the inner fit: `_fit_eval_fold` (tabular FTT) -> `NpyVolumeDataset` + `train_model` (3D CNN).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader

from pinksight.data.dataset import NpyVolumeDataset
from pinksight.metrics import delong_ci, ece
from pinksight.seed import set_seed
from pinksight.train.loop import TrainCfg, train_model

SEEDS = (0, 1, 2)
N_SPLITS = 5


def _encoder_embed(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Return the penultimate per-sample embedding for a model or its wrapped encoder.

    The CV model is `models.heads.SubtypeClassifier` (`.encoder` -> `.head`); the frozen embedding
    the fusion arm wants is the encoder output BEFORE `.head`. Resolution order (first that exists):
    `model.encoder.embed(x)` (MriEncoder alias) -> `model.encoder(x)` -> `model.embed(x)`. This keeps
    the smoke test's 3-line `TinyEncoder` (no `.embed`) working via the plain-call fallback."""
    enc = getattr(model, "encoder", model)
    if hasattr(enc, "embed"):
        return enc.embed(x)
    return enc(x)


def _collect_oof_embeddings(
    model: nn.Module, best_state: dict | None, loader: DataLoader, device: str, amp: bool = True
) -> tuple[list[str], np.ndarray]:
    """Reload best-on-val weights, run ONE no-grad pass over the held-out (OOF) test loader, and
    return (pids, embeddings[N, d]). The model never early-stopped on this fold, so the embeddings
    are honest out-of-fold — same integrity guarantee as the OOF probs (no test-fold peeking).
    FIX-3: `amp=False` forces fp32 so an --no-amp run exports fp32 embeddings (no fp16 NaN risk)."""
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device).eval()
    embs: list[np.ndarray] = []
    pids: list[str] = []
    use_amp = amp and device.startswith("cuda")
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
        for x, _, pid in loader:
            e = _encoder_embed(model, x.to(device)).float().cpu().numpy()
            embs.append(e.reshape(e.shape[0], -1))
            pids.extend(pid)
    return pids, (np.concatenate(embs) if embs else np.zeros((0, 0), dtype=float))


def _seed_monai(seed: int) -> None:
    """Seed MONAI's rand transforms so train-time augmentation is reproducible per (seed,fold).
    ponytail: one call; import-local so cv stays importable if monai internals shift."""
    from monai.utils import set_determinism
    set_determinism(seed=seed)


def _inner_val_split(
    tr: np.ndarray, y: np.ndarray, pids: list[str], seed: int, fold: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Patient-disjoint ~80/20 inner train/val split of the training indices (for early stopping).
    Returns (inner_train_idx, inner_val_idx) into the ORIGINAL index space, or None when the fold is
    too small to spare a val split (tiny smoke) — caller then falls back to val=test. ponytail: one
    StratifiedGroupKFold split, no new dep."""
    groups = np.asarray(pids)[tr]
    # Bail to val=test when the fold can't feed an N_SPLITS inner split. StratifiedGroupKFold needs
    # >= N_SPLITS members in the *least-populated class* (not just N_SPLITS groups) — the old guard
    # only checked group count, so tiny cohorts (8-patient smoke) slipped through and raised.
    _min_class = int(np.unique(y[tr], return_counts=True)[1].min()) if len(tr) else 0
    if (len(set(groups.tolist())) < N_SPLITS
            or len(set(y[tr].tolist())) < 2
            or _min_class < N_SPLITS):
        return None
    inner = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed * 100 + fold)
    a, b = next(inner.split(np.zeros(len(tr)), y[tr], groups))
    return tr[a], tr[b]


def cross_val_imaging(
    items: list[tuple[str, int]],
    cfg: TrainCfg,
    model_factory: Callable[[], nn.Module],
    channels: str = "first_post",
    spatial_size: tuple[int, int, int] = (96, 96, 96),
    proc_dir: Path = Path("data/processed"),
    seeds: tuple[int, ...] = SEEDS,
    ckpt_dir: Path | None = None,
    embed_dir: Path | None = None,
    weights_dir: Path | None = None,
    crop_mode: str = "none",
    crop_size: int = 96,
    box_margin_mm: int = 20,
    use_sampler: bool = False,
) -> dict:
    """items=[(pid,label)]; groups=pids. Returns the frozen metrics.json schema dict.

    `seeds` defaults to the locked (0,1,2); the label-shuffle probe overrides it to one cheap seed.

    `ckpt_dir` (opt-in) makes the sweep FOLD-LEVEL RESUMABLE for ephemeral runners (Kaggle): each
    (seed,fold)'s out-of-fold preds are persisted the moment they're trained, and a fold already on
    disk is loaded instead of retrained. A config signature guards against silently pooling preds
    from two different configs. `ckpt_dir=None` (default) is byte-identical to the non-resumable path,
    so local runs and tests are unaffected. The patient-disjoint LOCK-2 assert ALWAYS runs, even on
    a resumed fold — resume never skips an integrity check.

    `embed_dir` (opt-in; FIX-1) makes CV ALSO export frozen OUT-OF-FOLD per-patient MRI embeddings
    for the fusion arm. When set, each fold's best-on-val model re-scores its held-out test fold and
    the penultimate encoder embedding (512-d for the 3D-ResNet default) is captured per patient;
    after all folds of a seed, `embed_dir/mri_embed_s{SEED}.npz` is written with keys `pids:[str]`,
    `emb:[N,512]` — the exact schema `notebooks/fusion_kaggle.ipynb` cell 3 expects. `embed_dir=None`
    (default) is byte-identical to the prior path: no extra passes, no files, existing runs/tests
    unaffected. The OOF integrity guarantee matches the probs — each patient is embedded by a model
    that never trained on it (test fold, not the inner-val early-stop fold).

    `weights_dir` (opt-in; G5-LEG3) persists each fold's BEST-ON-VAL `state_dict` to
    `weights_dir/model_s{SEED}f{FOLD}.pt`. Without it `train_model` keeps `best_state` in memory only,
    so a finished GPU run leaves nothing for Grad-CAM to attribute over. Weights are ~132 MB/fold, so
    this is deliberately NOT folded into `ckpt_dir` — `weights_dir=None` (default) is byte-identical
    to the prior path. Attribution requires the FULL set: when resuming a run whose folds are already
    on disk in `ckpt_dir`, a missing `.pt` raises rather than silently yielding a partial weight set
    (same refuse-don't-backfill rule as `embed_dir` above — weights cannot be recovered from probs).
    """
    pids = [p for p, _ in items]
    y = np.array([lab for _, lab in items])
    n = len(items)
    if embed_dir is not None:
        embed_dir.mkdir(parents=True, exist_ok=True)
    if weights_dir is not None:
        weights_dir.mkdir(parents=True, exist_ok=True)

    if ckpt_dir is not None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        sig = {"n": n, "seeds": list(seeds), "spatial": list(spatial_size),
               "channels": channels, "batch": cfg.batch_size, "epochs": cfg.epochs,
               "recipe": "znorm+aug-v1", "eval": "inner-val",  # 'eval' bump invalidates peeked (val=test) folds
               # FIX-3: precision + clip are part of the config identity — an fp16-trained fold and an
               # fp32-trained fold must NOT be silently pooled (same intent as the 'eval' guard).
               "amp": cfg.amp, "grad_clip": cfg.grad_clip,
               # [G2-LESION-CROP]: crop geometry is part of config identity — a tight-lesion-crop fold and
               # a loose-Duke-box fold produce different inputs and must NEVER be silently pooled.
               # [HEAD2-GRADE-PIVOT] box_margin_mm too: a box@20mm fold and a box@30mm fold see different
               # peritumoral context and must never be pooled (only bites crop_mode="box").
               "crop_mode": crop_mode, "crop_size": crop_size, "box_margin_mm": box_margin_mm,
               # [G2-SUBTRACTION-REOPEN]: recipe knobs are part of config identity — a sampled/param-group/
               # cosine fold trains on a different distribution/schedule than the plain fold; never pool them.
               "use_sampler": use_sampler, "backbone_lr": cfg.backbone_lr,
               "head_lr": cfg.head_lr, "scheduler": cfg.scheduler}
        sig_p = ckpt_dir / "config.json"
        if sig_p.exists() and json.loads(sig_p.read_text()) != sig:
            raise RuntimeError(
                f"checkpoint config mismatch in {ckpt_dir}\n  on disk: {json.loads(sig_p.read_text())}"
                f"\n  now:     {sig}\nRefusing to pool preds across configs — delete this dir to "
                "restart the arm cleanly.")
        sig_p.write_text(json.dumps(sig, indent=2))

    def _loader(idx: np.ndarray, shuffle: bool, augment: bool = False, fold: int = 0,
                sample_labels: np.ndarray | None = None) -> DataLoader:
        # crop_mode/crop_size apply to EVERY fold (input geometry, not a train-only augmentation), so
        # val/test see the same tight-crop@fixed-cube as train — the isolated geometry lever.
        ds = NpyVolumeDataset([items[i] for i in idx], proc_dir, channels, spatial_size,
                              augment=augment, crop_mode=crop_mode, crop_size=crop_size,
                              box_margin_mm=box_margin_mm)
        # TICKET-021: seed the shuffle RNG per (seed,fold) so a resumed fold reproduces the ORIGINAL
        # shuffle order. Without an explicit generator, DataLoader draws from global RNG state that
        # model init + prior folds have already advanced -> a resumed fold shuffles differently.
        gen = torch.Generator().manual_seed(seed * 1000 + fold) if (shuffle or sample_labels is not None) else None
        # [G2-SUBTRACTION-REOPEN] WeightedRandomSampler (flag-gated via use_sampler). When
        # `sample_labels` is passed (TRAIN folds only, when use_sampler=True), draw each sample with
        # inverse class-frequency probability so a batch at ~21% TNBC prevalence reliably contains
        # minority samples (no zero-minority-gradient batches). shuffle MUST be False with a sampler
        # (they are mutually exclusive in DataLoader). val/test never pass sample_labels -> byte-identical.
        if sample_labels is not None:
            lab = np.asarray(sample_labels)
            classes, counts = np.unique(lab, return_counts=True)
            freq = dict(zip(classes.tolist(), counts.tolist()))
            weights = np.array([1.0 / freq[int(v)] for v in lab], dtype=np.float64)
            sampler = torch.utils.data.WeightedRandomSampler(
                torch.as_tensor(weights, dtype=torch.double),
                num_samples=len(weights), replacement=True, generator=gen)
            return DataLoader(ds, batch_size=cfg.batch_size, sampler=sampler, shuffle=False)
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle, generator=gen)

    per_seed, pooled, ci, ece_seed = {}, {}, {}, {}
    for seed in seeds:
        cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        oof = np.full(n, np.nan)
        fold_aucs = []
        emb_by_pid: dict[str, np.ndarray] = {}  # FIX-1: OOF embeddings for this seed (empty unless embed_dir)
        for f, (tr, te) in enumerate(cv.split(np.zeros(n), y, pids)):
            if not set(np.asarray(pids)[tr]).isdisjoint(np.asarray(pids)[te]):
                raise AssertionError("patient leaked across a CV fold — LOCK-2 violation")
            fold_ck = (ckpt_dir / f"oof_s{seed}f{f}.npz") if ckpt_dir is not None else None
            fold_wt = (weights_dir / f"model_s{seed}f{f}.pt") if weights_dir is not None else None
            if fold_ck is not None and fold_ck.exists():
                d = np.load(fold_ck, allow_pickle=True)  # resume: reuse the already-trained fold
                val_pids, val_probs = list(d["pids"]), d["probs"]
                if fold_wt is not None and not fold_wt.exists():
                    raise RuntimeError(
                        f"weights_dir set but {fold_ck.name} was trained without weight export — "
                        f"{fold_wt.name} is missing. Weights cannot be back-filled from probs; XAI "
                        "would attribute over a partial fold set. Delete the oof_ckpt dir to re-run "
                        "this arm with weight export.")
                if embed_dir is not None:
                    if "emb" not in d.files:
                        raise RuntimeError(
                            f"embed_dir set but {fold_ck.name} is a probs-only (pre-FIX-1) checkpoint "
                            "with no OOF embeddings. Delete the oof_ckpt dir to re-run this arm with "
                            "embedding export (probs cannot be back-filled into embeddings).")
                    for p_, e_ in zip(list(d["emb_pids"]), d["emb"]):
                        emb_by_pid[str(p_)] = np.asarray(e_, dtype=float)
            else:
                set_seed(seed)
                _seed_monai(seed)  # reproducible train-time augmentation
                model = model_factory()
                # Carve a patient-disjoint inner val split from the training fold; early-stop on it
                # and predict `te` ONCE with the best-on-val model (no test-set peeking). Tiny folds
                # (smoke) can't spare a val split -> fall back to val=test (no real number there).
                inner = _inner_val_split(tr, y, pids, seed, f)
                if inner is None:
                    tri, val_loader = tr, _loader(te, False)
                else:
                    tri, vali = inner
                    if not set(np.asarray(pids)[vali]).isdisjoint(np.asarray(pids)[te]):
                        raise AssertionError("inner-val leaked into test fold — LOCK-2 violation")
                    val_loader = _loader(vali, False)
                pos = float((y[tri] == 1).sum())
                neg = float((y[tri] == 0).sum())
                # [G2-SUBTRACTION-REOPEN] TRAIN fold only: when use_sampler, draw with inverse
                # class-frequency weights (val/test loaders never sampled — LOCK-2 eval integrity).
                train_labels = y[tri] if use_sampler else None
                best_state, val_probs, val_pids = train_model(
                    model, _loader(tri, True, augment=True, fold=f, sample_labels=train_labels),
                    val_loader, cfg,
                    pos_weight=neg / max(pos, 1.0), log_tag=f"s{seed}f{f}",
                    oof_loader=_loader(te, False),
                )
                if fold_wt is not None:  # G5-LEG3: the only artifact Grad-CAM can attribute over
                    torch.save(best_state, fold_wt)
                emb_pids: list[str] = []
                emb_arr = np.zeros((0, 0), dtype=float)
                if embed_dir is not None:
                    # ponytail: known ceiling (TICKET-021 LOW) — this is a SECOND full no-grad pass over
                    # the test fold (~20% extra wall-time + a T4 activation spike) rather than reusing
                    # loop.py's OOF pass. Left as-is: the refactor to thread embeddings out of train_model
                    # isn't worth the coupling for a one-time export.
                    # Honest OOF embedding: best-on-val model re-scores its held-out test fold once
                    # (never early-stopped on it) — same no-peeking guarantee as the OOF probs above.
                    emb_pids, emb_arr = _collect_oof_embeddings(
                        model, best_state, _loader(te, False), cfg.device, amp=cfg.amp)
                    for p_, e_ in zip(emb_pids, emb_arr):
                        emb_by_pid[str(p_)] = e_
                if fold_ck is not None:
                    save_kw = dict(pids=np.array(val_pids, dtype=object),
                                   probs=np.asarray(val_probs, dtype=float))
                    if embed_dir is not None:  # persist so a resumed fold restores embeddings too
                        save_kw["emb_pids"] = np.array(emb_pids, dtype=object)
                        save_kw["emb"] = np.asarray(emb_arr, dtype=float)
                    np.savez(fold_ck, **save_kw)
            prob_by_pid = dict(zip(val_pids, val_probs))
            for j in te:
                oof[j] = prob_by_pid[pids[j]]
            yte = y[te]
            if len(set(yte.tolist())) > 1:
                fold_aucs.append(roc_auc_score(yte, [prob_by_pid[pids[j]] for j in te]))

        if np.isnan(oof).any():
            raise RuntimeError("OOF preds incomplete — a patient-row was never in a test fold")
        per_seed[seed] = float(np.mean(fold_aucs)) if fold_aucs else float("nan")
        auc, lo, hi = delong_ci(y, oof)
        pooled[seed], ci[seed] = auc, (lo, hi)
        ece_seed[seed] = ece(y, oof)

        if embed_dir is not None:  # FIX-1: write frozen OOF MRI embeddings for the fusion arm
            emb_pids_ordered = [p for p in pids if p in emb_by_pid]
            if len(emb_pids_ordered) != n:
                missing = n - len(emb_pids_ordered)
                raise RuntimeError(
                    f"OOF embeddings incomplete for seed {seed}: {missing}/{n} patients never "
                    "embedded (a test fold was skipped?). Refusing to write a partial embedding npz.")
            emb_mat = np.stack([emb_by_pid[p] for p in emb_pids_ordered]).astype(float)
            np.savez(embed_dir / f"mri_embed_s{seed}.npz",
                     pids=np.array(emb_pids_ordered, dtype=object), emb=emb_mat)

    means = np.array(list(per_seed.values()))
    pooled_v = np.array(list(pooled.values()))
    return {
        "auroc_mean": float(np.nanmean(means)),
        "auroc_std_across_seeds": float(np.nanstd(means)),
        "auroc_min": float(np.nanmin(means)),
        "auroc_max": float(np.nanmax(means)),
        "per_seed_mean_auroc": {str(k): round(v, 4) for k, v in per_seed.items()},
        "auroc_pooled_oof_mean": float(pooled_v.mean()),
        "auroc_pooled_oof_per_seed": {str(k): round(v, 4) for k, v in pooled.items()},
        "delong_ci95_per_seed": {str(k): [round(lo, 4), round(hi, 4)] for k, (lo, hi) in ci.items()},
        "delong_ci95_mean": [
            round(float(np.mean([lo for lo, _ in ci.values()])), 4),
            round(float(np.mean([hi for _, hi in ci.values()])), 4),
        ],
        "ece_mean": round(float(np.mean(list(ece_seed.values()))), 4),
        "ece_per_seed": {str(k): round(v, 4) for k, v in ece_seed.items()},
        "ece_n_bins": 10,
        "n_dev": n,
        "tnbc_prevalence": round(float(np.mean(y)), 4),
        "n_splits": N_SPLITS,
        "seeds": list(seeds),
    }
