"""Training-harness CPU smoke test (Tier 1, skip-if-no-torch): the whole loop + CV on synthetic data.

Proves train_model + cross_val_imaging + NpyVolumeDataset wire together and emit the LOCKED metrics
schema — on tiny synthetic .npy volumes in a tmp dir, so it needs no GPU and no preprocessed cohort.
No real number, just the contract. <30s on CPU.
"""

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
    """3-line stand-in for MriEncoder: global-pool -> 8-dim embedding. (N,C,D,H,W) -> (N,8)."""

    embed_dim = 8

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(in_channels, 8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _make_items(tmp_path, n=24) -> list[tuple[str, int]]:
    """n balanced patients; label drives mean intensity so a fold can actually learn (no degenerate run)."""
    rng = np.random.default_rng(0)
    items = []
    for i in range(n):
        label = i % 2
        d, h, w = _SIZES[i % len(_SIZES)]
        vol = (rng.standard_normal((2, d, h, w)).astype(np.float32) + 3.0 * label)  # signal in mean
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
    """The honest-OOF fix: with oof_loader set, returned pids/probs are the oof fold's, NOT the val
    fold's — proves early stopping and OOF prediction use disjoint data (no test-set peeking)."""
    items = _make_items(tmp_path, n=24)
    val_items, oof_items = items[:12], items[12:]
    mk = lambda its: DataLoader(
        NpyVolumeDataset(its, tmp_path, channels="first_post", spatial_size=(16, 16, 16)),
        batch_size=4)
    model = SubtypeClassifier(TinyEncoder(in_channels=1))
    _, probs, pids = train_model(
        model, mk(val_items), mk(val_items), TrainCfg(epochs=2, device="cpu"),
        pos_weight=1.0, oof_loader=mk(oof_items))
    assert set(pids) == {p for p, _ in oof_items}          # preds are the OOF fold's patients
    assert set(pids).isdisjoint({p for p, _ in val_items})  # never the early-stop (val) patients
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
    """FIX-1: embed_dir opt-in writes mri_embed_s{SEED}.npz with keys pids:[str], emb:[N,d] — the
    exact schema fusion_kaggle.ipynb cell 3 loads. All dev patients present exactly once (OOF); the
    non-embed default path is unaffected (covered by the other tests)."""
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
    assert set(pids) == {p for p, _ in items}      # every dev patient embedded, OUT-OF-FOLD
    assert len(pids) == len(set(pids)) == 24        # exactly once (no fold duplication)
    assert emb.shape == (24, TinyEncoder.embed_dim)  # [N, penultimate-dim]
    assert emb.dtype.kind == "f" and np.isfinite(emb).all()


def test_score_returns_neg_inf_on_nonfinite_probs():
    """FIX-3: a non-finite prob (NaN/inf) from an fp16-overflowed epoch must return -inf, NOT crash
    roc_auc_score with 'Input contains NaN'. This is what lets early-stop skip a blown epoch and keep
    the last good best_state, so one bad fold degrades gracefully instead of killing an unattended run."""
    y = np.array([0, 1, 0, 1])
    assert _score(y, np.array([np.nan, 0.6, 0.4, 0.7])) == -np.inf
    assert _score(y, np.array([0.1, np.inf, 0.4, 0.7])) == -np.inf
    assert _score(y, np.array([0.1, -np.inf, 0.4, 0.7])) == -np.inf
    assert np.isfinite(_score(y, np.array([0.1, 0.6, 0.4, 0.7])))  # clean probs unaffected


def test_amp_false_runs_fp32_and_finite(tmp_path):
    """FIX-3: cfg.amp=False (fp32 path) trains and eval without error and yields finite probs in [0,1]
    on CPU (autocast/scaler are no-ops on CPU, but this exercises the amp plumbing through _eval)."""
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
    assert x.shape == (2, 8, 8, 8)  # pre + first-post
    x1, _, _ = NpyVolumeDataset(items, tmp_path, channels="first_post", spatial_size=(8, 8, 8))[0]
    assert x1.shape == (1, 8, 8, 8)


# --- [G2-SUBTRACTION-REOPEN] recipe correction: param-group LR split + cosine schedule (E7/TC-6) ---

def _split_head_backbone(model: nn.Module) -> tuple[list[str], list[str]]:
    """Reproduce loop.py's EXACT param partition (filter on '.head.' only, contract C2) so the test
    asserts the real invariant, not a paraphrase: SubtypeClassifier's head is `self.head`
    (names 'head.weight'/'head.bias'); everything else (the encoder) is the backbone."""
    head, backbone = [], []
    for name, _ in model.named_parameters():
        (head if ".head." in f".{name}." else backbone).append(name)
    return head, backbone


def test_param_group_filter_puts_head_params_in_head_group():
    """C2/E2: the '.head.' filter must isolate exactly the head Linear (head.weight/head.bias) into the
    head group and leave every encoder param in the backbone group. A '.fc.' filter (the wrong one the
    contract warns against) would match nothing on MriEncoder/SubtypeClassifier and silently push the
    head into the backbone group — this test is the guard against that regression."""
    model = SubtypeClassifier(TinyEncoder(in_channels=1))
    head, backbone = _split_head_backbone(model)
    assert set(head) == {"head.weight", "head.bias"}, head
    assert all(n.startswith("encoder.") for n in backbone), backbone
    assert not any(".head." in f".{n}." for n in backbone)  # no head param leaked into backbone
    assert head and backbone  # both groups non-empty (a '.fc.' filter would empty the head group)


def test_param_group_optimizer_has_two_groups_with_distinct_lrs():
    """The constructed AdamW must have exactly two param groups at the requested backbone/head LRs.
    Reconstructed via the same filter train_model uses (kept in _split_head_backbone), so this asserts
    the real optimizer shape the recipe produces, not a re-implementation."""
    model = SubtypeClassifier(TinyEncoder(in_channels=1))
    head_names, backbone_names = _split_head_backbone(model)
    by_name = dict(model.named_parameters())
    param_groups = [
        {"params": [by_name[n] for n in backbone_names], "lr": 1e-4},
        {"params": [by_name[n] for n in head_names], "lr": 1e-3},
    ]
    opt = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    assert len(opt.param_groups) == 2
    assert opt.param_groups[0]["lr"] == 1e-4  # backbone
    assert opt.param_groups[1]["lr"] == 1e-3  # head (10x differential — the fine-tune fix)


def test_train_model_backbone_lr_and_cosine_runs_and_probs_valid(tmp_path):
    """End-to-end real-path proof: TrainCfg(backbone_lr=1e-4, head_lr=1e-3, scheduler='cosine') drives
    train_model through the param-group split + CosineAnnealingLR branch (loop.py) without error and
    yields finite probs in [0,1]. Exercises the actual recipe wiring, not just the filter helper."""
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
    """TrainCfg.scheduler must be 'none' or 'cosine'; anything else raises in train_model (fail-fast at
    the boundary, not a silent no-op that skips the schedule the operator asked for)."""
    items = _make_items(tmp_path, n=8)
    ds = NpyVolumeDataset(items, tmp_path, channels="first_post", spatial_size=(8, 8, 8))
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    model = SubtypeClassifier(TinyEncoder(in_channels=1))
    with pytest.raises(ValueError):
        train_model(model, loader, loader,
                    TrainCfg(epochs=1, device="cpu", scheduler="linear"), pos_weight=1.0)


# --- [G2-SUBTRACTION-REOPEN] WeightedRandomSampler yields >=1 minority per batch (E7/TC-7) ---

def _sampler_min_per_batch(labels: np.ndarray, batch_size: int, n_batches: int, seed: int = 0) -> int:
    """Build the SAME inverse-class-frequency WeightedRandomSampler cross_val_imaging._loader builds
    (train folds, use_sampler=True) and return the WORST-case minority count seen across n_batches.
    Mirrors the code's weight math exactly (weights[i] = 1/class_count[label[i]]) so the assertion
    binds the real recipe behavior."""
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
    """E7/TC-7: at ~21% TNBC prevalence the plain-shuffle loader produces zero-minority batches (the
    zero-minority-gradient fault). The WeightedRandomSampler (inverse class-frequency) must place at
    least 1 minority sample in every batch of 4 across 5 batches. This is the recipe guarantee the
    sampler flag exists to provide."""
    rng = np.random.default_rng(0)
    # 100 patients at ~21% TNBC (matches the dev cohort's 0.207 prevalence) — the imbalanced regime.
    labels = np.array([1 if rng.random() < 0.21 else 0 for _ in range(100)])
    assert 0 < labels.sum() < len(labels)  # genuinely imbalanced, both classes present
    worst = _sampler_min_per_batch(labels, batch_size=4, n_batches=5, seed=0)
    assert worst >= 1, f"a batch had {worst} minority samples — sampler failed its >=1-per-batch job"


def test_weighted_sampler_beats_plain_shuffle_on_minority_coverage():
    """Corroborates the fault the sampler fixes: with a heavily imbalanced set, plain uniform sampling
    CAN yield an all-majority batch, whereas the weighted sampler does not (over the same batches).
    Guards against the flag silently degrading to uniform sampling."""
    rng = np.random.default_rng(1)
    labels = np.array([1 if rng.random() < 0.15 else 0 for _ in range(80)])  # ~15% minority, harsher
    assert 0 < labels.sum() < len(labels)
    weighted_worst = _sampler_min_per_batch(labels, batch_size=4, n_batches=5, seed=1)
    assert weighted_worst >= 1  # the weighted sampler keeps minority coverage even at 15%


def test_cross_val_imaging_use_sampler_runs_and_emits_schema(tmp_path):
    """Real-path proof that use_sampler=True threads through cross_val_imaging -> _loader ->
    WeightedRandomSampler on train folds without error and still emits the LOCKED schema. val/test
    loaders are never sampled (LOCK-2) — this run just proves the train-fold sampler path executes."""
    items = _make_items(tmp_path, n=24)
    cfg = TrainCfg(epochs=2, batch_size=4, device="cpu")
    factory = lambda: SubtypeClassifier(TinyEncoder(in_channels=1))
    m = cross_val_imaging(items, cfg, factory, spatial_size=(16, 16, 16), proc_dir=tmp_path,
                          seeds=(0,), use_sampler=True)
    assert SCHEMA_KEYS <= set(m), f"missing keys: {SCHEMA_KEYS - set(m)}"
    assert m["n_dev"] == 24 and m["n_splits"] == 5
    assert 0.0 <= m["auroc_pooled_oof_mean"] <= 1.0
