
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("monai")

from torch import nn
from torch.utils.data import DataLoader

from pinksight.data.dataset import NpyVolumeDataset
from pinksight.models.heads import SubtypeClassifier
from pinksight.train.cv import cross_val_imaging
from pinksight.train.loop import TrainCfg, _score, train_model

SCHEMA_KEYS = {
    "auroc_mean", "auroc_std_across_seeds", "auroc_min", "auroc_max", "per_seed_mean_auroc",
    "auroc_pooled_oof_mean", "auroc_pooled_oof_per_seed", "delong_ci95_per_seed", "delong_ci95_mean",
    "ece_mean", "ece_per_seed", "ece_n_bins", "n_dev", "tnbc_prevalence", "n_splits", "seeds",
}
_SIZES = [(18, 16, 20), (20, 18, 16), (16, 20, 18)]


class TinyEncoder(nn.Module):

    embed_dim = 8

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(in_channels, 8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _make_items(tmp_path, n=24) -> list[tuple[str, int]]:
    rng = np.random.default_rng(0)
    items = []
    for i in range(n):
        label = i % 2
        d, h, w = _SIZES[i % len(_SIZES)]
        vol = (rng.standard_normal((2, d, h, w)).astype(np.float32) + 3.0 * label)  
        pid = f"FAKE_{i:03d}"
        np.save(tmp_path / f"{pid}.npy", vol)
        items.append((pid, label))
    return items


def test_train_model_runs_and_probs_valid(tmp_path):
    items = _make_items(tmp_path, n=12)
    ds = NpyVolumeDataset(items, tmp_path, channels="first_post", spatial_size=(16, 16, 16))
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    model = SubtypeClassifier(TinyEncoder(in_channels=1))
    _, probs, pids = train_model(model, loader, loader, TrainCfg(epochs=3, device="cpu"), pos_weight=1.0)
    assert probs.shape == (12,)
    assert ((probs >= 0) & (probs <= 1)).all()
    assert set(pids) == {p for p, _ in items}


def test_oof_loader_preds_come_from_oof_fold_not_val(tmp_path):
    items = _make_items(tmp_path, n=24)
    val_items, oof_items = items[:12], items[12:]
    mk = lambda its: DataLoader(
        NpyVolumeDataset(its, tmp_path, channels="first_post", spatial_size=(16, 16, 16)),
        batch_size=4)
    model = SubtypeClassifier(TinyEncoder(in_channels=1))
    _, probs, pids = train_model(
        model, mk(val_items), mk(val_items), TrainCfg(epochs=2, device="cpu"),
        pos_weight=1.0, oof_loader=mk(oof_items))
    assert set(pids) == {p for p, _ in oof_items}          
    assert set(pids).isdisjoint({p for p, _ in val_items})  
    assert probs.shape == (12,) and ((probs >= 0) & (probs <= 1)).all()


def test_cross_val_emits_full_schema(tmp_path):
    items = _make_items(tmp_path, n=24)
    cfg = TrainCfg(epochs=3, batch_size=4, device="cpu")
    factory = lambda: SubtypeClassifier(TinyEncoder(in_channels=1))
    m = cross_val_imaging(items, cfg, factory, spatial_size=(16, 16, 16), proc_dir=tmp_path, seeds=(0,))
    assert SCHEMA_KEYS <= set(m), f"missing keys: {SCHEMA_KEYS - set(m)}"
    assert m["n_dev"] == 24
    assert m["n_splits"] == 5
    assert 0.0 <= m["auroc_pooled_oof_mean"] <= 1.0
    assert m["ece_mean"] >= 0.0
    lo, hi = m["delong_ci95_mean"]
    assert lo <= hi


def test_oof_embedding_export_writes_fusion_schema(tmp_path):
    items = _make_items(tmp_path, n=24)
    cfg = TrainCfg(epochs=3, batch_size=4, device="cpu")
    factory = lambda: SubtypeClassifier(TinyEncoder(in_channels=1))
    embed_dir = tmp_path / "embeddings"
    cross_val_imaging(items, cfg, factory, spatial_size=(16, 16, 16), proc_dir=tmp_path,
                      seeds=(0,), embed_dir=embed_dir)

    npz = embed_dir / "mri_embed_s0.npz"
    assert npz.exists(), "embed_dir set but mri_embed_s0.npz was not written"
    d = np.load(npz, allow_pickle=True)
    assert set(d.files) >= {"pids", "emb"}, f"schema drift: {d.files}"
    pids, emb = list(d["pids"]), d["emb"]
    assert set(pids) == {p for p, _ in items}      
    assert len(pids) == len(set(pids)) == 24        
    assert emb.shape == (24, TinyEncoder.embed_dim)  
    assert emb.dtype.kind == "f" and np.isfinite(emb).all()


def test_score_returns_neg_inf_on_nonfinite_probs():
    y = np.array([0, 1, 0, 1])
    assert _score(y, np.array([np.nan, 0.6, 0.4, 0.7])) == -np.inf
    assert _score(y, np.array([0.1, np.inf, 0.4, 0.7])) == -np.inf
    assert _score(y, np.array([0.1, -np.inf, 0.4, 0.7])) == -np.inf
    assert np.isfinite(_score(y, np.array([0.1, 0.6, 0.4, 0.7])))  


def test_amp_false_runs_fp32_and_finite(tmp_path):
    items = _make_items(tmp_path, n=12)
    ds = NpyVolumeDataset(items, tmp_path, channels="first_post", spatial_size=(16, 16, 16))
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    model = SubtypeClassifier(TinyEncoder(in_channels=1))
    _, probs, _ = train_model(
        model, loader, loader,
        TrainCfg(epochs=2, device="cpu", amp=False, grad_clip=1.0), pos_weight=1.0)
    assert probs.shape == (12,)
    assert np.isfinite(probs).all() and ((probs >= 0) & (probs <= 1)).all()


def test_channel_policy_sets_input_channels(tmp_path):
    items = _make_items(tmp_path, n=4)
    x, _, _ = NpyVolumeDataset(items, tmp_path, channels="pre_post", spatial_size=(8, 8, 8))[0]
    assert x.shape == (2, 8, 8, 8)  
    x1, _, _ = NpyVolumeDataset(items, tmp_path, channels="first_post", spatial_size=(8, 8, 8))[0]
    assert x1.shape == (1, 8, 8, 8)


def _split_head_backbone(model: nn.Module) -> tuple[list[str], list[str]]:
    head, backbone = [], []
    for name, _ in model.named_parameters():
        (head if ".head." in f".{name}." else backbone).append(name)
    return head, backbone


def test_param_group_filter_puts_head_params_in_head_group():
    model = SubtypeClassifier(TinyEncoder(in_channels=1))
    head, backbone = _split_head_backbone(model)
    assert set(head) == {"head.weight", "head.bias"}, head
    assert all(n.startswith("encoder.") for n in backbone), backbone
    assert not any(".head." in f".{n}." for n in backbone)  
    assert head and backbone  


def test_param_group_optimizer_has_two_groups_with_distinct_lrs():
    model = SubtypeClassifier(TinyEncoder(in_channels=1))
    head_names, backbone_names = _split_head_backbone(model)
    by_name = dict(model.named_parameters())
    param_groups = [
        {"params": [by_name[n] for n in backbone_names], "lr": 1e-4},
        {"params": [by_name[n] for n in head_names], "lr": 1e-3},
    ]
    opt = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    assert len(opt.param_groups) == 2
    assert opt.param_groups[0]["lr"] == 1e-4  
    assert opt.param_groups[1]["lr"] == 1e-3  


def test_train_model_backbone_lr_and_cosine_runs_and_probs_valid(tmp_path):
    items = _make_items(tmp_path, n=12)
    ds = NpyVolumeDataset(items, tmp_path, channels="first_post", spatial_size=(16, 16, 16))
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    model = SubtypeClassifier(TinyEncoder(in_channels=1))
    _, probs, _ = train_model(
        model, loader, loader,
        TrainCfg(epochs=3, device="cpu", backbone_lr=1e-4, head_lr=1e-3, scheduler="cosine"),
        pos_weight=1.0)
    assert probs.shape == (12,)
    assert np.isfinite(probs).all() and ((probs >= 0) & (probs <= 1)).all()


def test_bad_scheduler_value_raises(tmp_path):
    items = _make_items(tmp_path, n=8)
    ds = NpyVolumeDataset(items, tmp_path, channels="first_post", spatial_size=(8, 8, 8))
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    model = SubtypeClassifier(TinyEncoder(in_channels=1))
    with pytest.raises(ValueError):
        train_model(model, loader, loader,
                    TrainCfg(epochs=1, device="cpu", scheduler="linear"), pos_weight=1.0)


def _sampler_min_per_batch(labels: np.ndarray, batch_size: int, n_batches: int, seed: int = 0) -> int:
    classes, counts = np.unique(labels, return_counts=True)
    freq = dict(zip(classes.tolist(), counts.tolist()))
    weights = np.array([1.0 / freq[int(v)] for v in labels], dtype=np.float64)
    gen = torch.Generator().manual_seed(seed)
    sampler = torch.utils.data.WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights), replacement=True, generator=gen)
    minority_class = int(classes[np.argmin(counts)])
    idx = list(sampler)
    worst = batch_size
    for b in range(n_batches):
        batch = idx[b * batch_size:(b + 1) * batch_size]
        if len(batch) < batch_size:
            break
        worst = min(worst, sum(int(labels[i]) == minority_class for i in batch))
    return worst


def test_weighted_sampler_places_minority_in_every_batch():
    rng = np.random.default_rng(0)
    labels = np.array([1 if rng.random() < 0.21 else 0 for _ in range(100)])
    assert 0 < labels.sum() < len(labels)  
    worst = _sampler_min_per_batch(labels, batch_size=4, n_batches=5, seed=0)
    assert worst >= 1, f"a batch had {worst} minority samples — sampler failed its >=1-per-batch job"


def test_weighted_sampler_beats_plain_shuffle_on_minority_coverage():
    rng = np.random.default_rng(1)
    labels = np.array([1 if rng.random() < 0.15 else 0 for _ in range(80)])  
    assert 0 < labels.sum() < len(labels)
    weighted_worst = _sampler_min_per_batch(labels, batch_size=4, n_batches=5, seed=1)
    assert weighted_worst >= 1  


def test_cross_val_imaging_use_sampler_runs_and_emits_schema(tmp_path):
    items = _make_items(tmp_path, n=24)
    cfg = TrainCfg(epochs=2, batch_size=4, device="cpu")
    factory = lambda: SubtypeClassifier(TinyEncoder(in_channels=1))
    m = cross_val_imaging(items, cfg, factory, spatial_size=(16, 16, 16), proc_dir=tmp_path,
                          seeds=(0,), use_sampler=True)
    assert SCHEMA_KEYS <= set(m), f"missing keys: {SCHEMA_KEYS - set(m)}"
    assert m["n_dev"] == 24 and m["n_splits"] == 5
    assert 0.0 <= m["auroc_pooled_oof_mean"] <= 1.0
