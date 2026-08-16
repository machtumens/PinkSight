
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader

from pinksight.data.slice_dataset import SliceGradeDataset
from pinksight.metrics import delong_ci, ece
from pinksight.seed import set_seed
from pinksight.train.loop import TrainCfg, train_model

N_SPLITS = 5
SEEDS = (0, 1, 2)


def _seed_monai(seed: int) -> None:
    from monai.utils import set_determinism

    set_determinism(seed=seed)


def _inner_val_split(
    tr: np.ndarray, y: np.ndarray, pids: list[str], seed: int, fold: int
) -> tuple[np.ndarray, np.ndarray] | None:
    groups = np.asarray(pids)[tr]
    _min_class = int(np.unique(y[tr], return_counts=True)[1].min()) if len(tr) else 0
    if (len(set(groups.tolist())) < N_SPLITS
            or len(set(y[tr].tolist())) < 2
            or _min_class < N_SPLITS):
        return None
    inner = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed * 100 + fold)
    a, b = next(inner.split(np.zeros(len(tr)), y[tr], groups))
    return tr[a], tr[b]


def _assert_slice_disjoint(ds_a: SliceGradeDataset, ds_b: SliceGradeDataset, where: str) -> None:
    pa = {s[0] for s in ds_a.samples}
    pb = {s[0] for s in ds_b.samples}
    shared = pa & pb
    if shared:
        raise AssertionError(
            f"SLICE-LEVEL LEAK ({where}): {len(shared)} patient(s) have slices in both sets "
            f"(e.g. {sorted(shared)[:3]}) — patient-disjoint-at-slice-level violation (LOCK-2)."
        )


def cross_val_slices(
    items: list[tuple[str, int]],
    cfg: TrainCfg,
    model_factory: Callable[[], nn.Module],
    proc_dir: Path = Path("data/processed"),
    seeds: tuple[int, ...] = SEEDS,
) -> dict:
    pids = [p for p, _ in items]
    y = np.array([lab for _, lab in items])
    n = len(items)

    per_seed, pooled, ci, ece_seed = {}, {}, {}, {}
    total_train_slices, total_test_slices = [], []
    disjoint_ok = True

    def _loader(pairs: list[tuple[str, int]], split: str, augment: bool,
                shuffle: bool, seed: int, fold: int) -> tuple[DataLoader, SliceGradeDataset]:
        ds = SliceGradeDataset(pairs, proc_dir=proc_dir, split=split, augment=augment)
        gen = None
        if shuffle:
            import torch

            gen = torch.Generator().manual_seed(seed * 1000 + fold)
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle, generator=gen), ds

    for seed in seeds:
        cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        oof = np.full(n, np.nan)  
        fold_aucs = []
        for f, (tr, te) in enumerate(cv.split(np.zeros(n), y, pids)):
            if not set(np.asarray(pids)[tr]).isdisjoint(np.asarray(pids)[te]):
                raise AssertionError("patient leaked across a CV fold — LOCK-2 violation")

            set_seed(seed)
            _seed_monai(seed)
            model = model_factory()

            inner = _inner_val_split(tr, y, pids, seed, f)
            te_pairs = [items[i] for i in te]
            if inner is None:
                tri = tr
                val_loader, val_ds = _loader(te_pairs, "test", False, False, seed, f)
                train_loader, train_ds = _loader([items[i] for i in tri], "train", True, True, seed, f)
            else:
                tri, vali = inner
                if not set(np.asarray(pids)[vali]).isdisjoint(np.asarray(pids)[te]):
                    raise AssertionError("inner-val leaked into test fold — LOCK-2 violation")
                vali_pairs = [items[i] for i in vali]
                val_loader, val_ds = _loader(vali_pairs, "test", False, False, seed, f)
                train_loader, train_ds = _loader([items[i] for i in tri], "train", True, True, seed, f)

            oof_loader, oof_ds = _loader(te_pairs, "test", False, False, seed, f)

            _assert_slice_disjoint(train_ds, val_ds, f"s{seed}f{f} train vs val")
            _assert_slice_disjoint(train_ds, oof_ds, f"s{seed}f{f} train vs test")
            if inner is not None:
                _assert_slice_disjoint(val_ds, oof_ds, f"s{seed}f{f} val vs test")

            total_train_slices.append(len(train_ds))
            total_test_slices.append(len(oof_ds))

            pos = float((y[tri] == 1).sum())
            neg = float((y[tri] == 0).sum())
            _best, oof_probs, oof_pids = train_model(
                model, train_loader, val_loader, cfg,
                pos_weight=neg / max(pos, 1.0), log_tag=f"s{seed}f{f}",
                oof_loader=oof_loader,
            )
            prob_by_pid = dict(zip(oof_pids, oof_probs))
            if len(prob_by_pid) != len(te):
                raise RuntimeError(
                    f"expected 1 OOF prob per test patient ({len(te)}); got {len(prob_by_pid)} unique "
                    "pids — the test dataset emitted != 1 slice/patient (supra-central protocol broken)."
                )
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
        "oof_unit": "patient (single supra-central slice per patient; DeLong N = patient count)",
        "train_slices_total_per_fold_mean": round(float(np.mean(total_train_slices)), 1),
        "test_slices_total_per_fold_mean": round(float(np.mean(total_test_slices)), 1),
        "slice_level_patient_disjoint_asserted": disjoint_ok,
    }
