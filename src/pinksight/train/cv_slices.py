"""[HEAD2-GRADE-2D-SLICE] Patient-level CV for the 2D-per-slice grade recipe (DeepRadGrade).

Mirrors `train/cv.py::cross_val_imaging` (patient-level StratifiedGroupKFold(5), inner-val early stop,
pooled-OOF DeLong CI + ECE, same metrics.json schema) so the 2D result slots into the exact same
comparison table as the 3D null. The differences that matter for this recipe:

  * inputs are 2D SLICES (SliceGradeDataset) — a patient contributes up to 8 train slices but exactly
    ONE test slice (the supra-central slice, DeepRadGrade protocol).
  * ***NEW slice-level leakage guard*** — grouping is by PATIENT, so we assert no `pid` has slices in
    both the train and val slice sets of any fold. This is the leakage risk the slice representation
    introduces and the pre-reg's critical new guard.
  * OOF AUROC is computed at the PATIENT level: one prediction per patient (its single test slice), so
    N in the DeLong CI is the patient count, never an inflated slice count.

Same seed plumbing, same `train_model`, same DeLong/ECE metrics as the 3D harness — only the dataset
and the split-unit assertions change.
"""

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
    """Patient-disjoint ~80/20 inner split of the training PATIENTS (for early stopping). None when the
    fold is too small to spare a val split (tiny smoke) -> caller falls back to val=test."""
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
    """The NEW guard: no patient's SLICES may appear in both sample sets. Reads the (pid,...) tags the
    dataset stamps on every emitted slice, so it verifies the actual materialised samples, not the
    intended split. A shared pid here would mean a patient's slices leaked across the fold boundary."""
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
    """items=[(pid,label)]; grouping=pids. Returns the frozen metrics.json schema dict + slice/disjoint
    fields. One OOF prediction per patient (its supra-central test slice) -> patient-level DeLong CI."""
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
        oof = np.full(n, np.nan)  # one prob per PATIENT (index-aligned to items)
        fold_aucs = []
        for f, (tr, te) in enumerate(cv.split(np.zeros(n), y, pids)):
            # patient-level fold disjointness (same as the 3D harness)
            if not set(np.asarray(pids)[tr]).isdisjoint(np.asarray(pids)[te]):
                raise AssertionError("patient leaked across a CV fold — LOCK-2 violation")

            set_seed(seed)
            _seed_monai(seed)
            model = model_factory()

            inner = _inner_val_split(tr, y, pids, seed, f)
            te_pairs = [items[i] for i in te]
            if inner is None:
                tri = tr
                # tiny fold: val=test (no honest number there, mirrors the 3D smoke fallback)
                val_loader, val_ds = _loader(te_pairs, "test", False, False, seed, f)
                train_loader, train_ds = _loader([items[i] for i in tri], "train", True, True, seed, f)
            else:
                tri, vali = inner
                if not set(np.asarray(pids)[vali]).isdisjoint(np.asarray(pids)[te]):
                    raise AssertionError("inner-val leaked into test fold — LOCK-2 violation")
                vali_pairs = [items[i] for i in vali]
                val_loader, val_ds = _loader(vali_pairs, "test", False, False, seed, f)
                train_loader, train_ds = _loader([items[i] for i in tri], "train", True, True, seed, f)

            # OOF test loader: exactly one slice per test patient (supra-central)
            oof_loader, oof_ds = _loader(te_pairs, "test", False, False, seed, f)

            # ***NEW slice-level patient-disjoint assertions*** (the pre-reg's critical new guard)
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
            # exactly one prob per test patient (SliceGradeDataset split="test" emits one slice/patient)
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
        auc, lo, hi = delong_ci(y, oof)  # PATIENT-level DeLong (N = patient count, not slice count)
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
        "tnbc_prevalence": round(float(np.mean(y)), 4),  # here = NHG3 prevalence
        "n_splits": N_SPLITS,
        "seeds": list(seeds),
        # 2D-slice-specific provenance
        "oof_unit": "patient (single supra-central slice per patient; DeLong N = patient count)",
        "train_slices_total_per_fold_mean": round(float(np.mean(total_train_slices)), 1),
        "test_slices_total_per_fold_mean": round(float(np.mean(total_test_slices)), 1),
        "slice_level_patient_disjoint_asserted": disjoint_ok,
    }
