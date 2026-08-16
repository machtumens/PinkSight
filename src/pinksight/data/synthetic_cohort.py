
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Any

import numpy as np

from pinksight import FORBIDDEN_FEATURES
from pinksight.models.clinical_encoder import AGE_FEATURE, FEATURES_CAT, FEATURES_NUM

_CAT_CARDS: tuple[int, ...] = (2, 5, 2, 2, 2)
if len(_CAT_CARDS) != len(FEATURES_CAT):  
    raise RuntimeError(
        f"_CAT_CARDS ({len(_CAT_CARDS)}) must match FEATURES_CAT ({len(FEATURES_CAT)}) — update the "
        "synthetic categorical cardinality schema if the clinical feature set changed."
    )
CAT_CARDINALITIES: dict[str, int] = dict(zip(FEATURES_CAT, _CAT_CARDS))

NEG_PREFIX = "SYN-NEG"
POS_PREFIX = "SYN-POS"
DEFAULT_EFFECT_SIZE = 1.2  
_SUPPORTED_CHANNELS = ("pre_post",)  


def _assert_no_forbidden_keys(row: dict[str, Any]) -> None:
    leaked = set(row) & set(FORBIDDEN_FEATURES)
    if leaked:
        raise ValueError(
            f"LEAKAGE: synthetic clinical row carries forbidden feature(s) {sorted(leaked)} — "
            "ER/PR/HER2/Ki-67/etc. must never enter classifier inputs (LOCK-2)."
        )


def _validate_channels(channels: str) -> None:
    if channels not in _SUPPORTED_CHANNELS:
        raise ValueError(f"channels must be one of {_SUPPORTED_CHANNELS}, got {channels!r}")


def _balanced_labels(n: int, rng: np.random.Generator) -> np.ndarray:
    y = np.zeros(n, dtype=int)
    y[: n // 2] = 1
    return rng.permutation(y)


def _clinical_row(rng: np.random.Generator, shift: float) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for name in FEATURES_NUM:  
        row[name] = float(rng.normal(0.0, 1.0) + shift)
    for name in FEATURES_CAT:
        row[name] = int(rng.integers(0, CAT_CARDINALITIES[name]))
    _assert_no_forbidden_keys(row)
    return row


def _mri_pair(rng: np.random.Generator, cube_size: int, blob: float) -> np.ndarray:
    edge = cube_size * 2
    pre = rng.normal(0.5, 0.1, (edge, edge, edge)).astype(np.float32)
    post = pre + rng.normal(0.0, 0.1, (edge, edge, edge)).astype(np.float32)
    if blob > 0.0:
        q = edge // 4
        post[q : 3 * q, q : 3 * q, q : 3 * q] += np.float32(blob)  
    return np.stack([pre, post], axis=0).astype(np.float32)


def _generate(
    prefix: str,
    n: int,
    seed: int,
    cube_size: int,
    channels: str,
    effect_size: float,
) -> Iterator[tuple[str, np.ndarray, dict[str, Any], int]]:
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if cube_size < 1:
        raise ValueError(f"cube_size must be >= 1, got {cube_size}")
    _validate_channels(channels)
    rng = np.random.default_rng(seed)
    labels = _balanced_labels(n, rng)
    for i in range(n):
        label = int(labels[i])
        shift = effect_size if (effect_size > 0.0 and label == 1) else 0.0
        blob = (0.5 * effect_size) if (effect_size > 0.0 and label == 1) else 0.0
        patient_id = f"{prefix}-{i:05d}"
        yield patient_id, _mri_pair(rng, cube_size, blob), _clinical_row(rng, shift), label


def generate_negative_control(
    n: int,
    seed: int = 0,
    cube_size: int = 16,
    channels: str = "pre_post",
) -> Iterator[tuple[str, np.ndarray, dict[str, Any], int]]:
    yield from _generate(NEG_PREFIX, n, seed, cube_size, channels, effect_size=0.0)


def generate_positive_control(
    n: int,
    seed: int = 0,
    cube_size: int = 16,
    channels: str = "pre_post",
    effect_size: float = DEFAULT_EFFECT_SIZE,
) -> Iterator[tuple[str, np.ndarray, dict[str, Any], int]]:
    if effect_size <= 0.0:
        raise ValueError(f"positive control needs effect_size > 0, got {effect_size}")
    yield from _generate(POS_PREFIX, n, seed, cube_size, channels, effect_size)


def compute_synthetic_manifest_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_manifest(config: dict[str, Any]) -> dict[str, Any]:
    required = {"n", "seed", "cube_size", "channels", "effect_size", "stream_name", "git_commit"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"manifest config missing required fields: {sorted(missing)}")
    stream = config["stream_name"]
    summary = (
        f"SYNTHETIC plumbing cohort '{stream}': n={config['n']} patients, seed={config['seed']}, "
        f"cube_size={config['cube_size']}, channels={config['channels']}, "
        f"effect_size={config['effect_size']} — non-reportable, forward-only characterisation wiring."
    )
    return {
        "manifest_sha256": compute_synthetic_manifest_hash(config),
        "config": dict(config),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
    }


A_NEG_PREFIX = "SYN-A-NEG"
A_POS_PREFIX = "SYN-A-POS"
TNBC_PREVALENCE_DEFAULT = 0.21  

_A_AGE_MEAN, _A_AGE_STD = 56.0, 13.0
_A_AGE_CLIP = (25.0, 90.0)
_A_TSTAGE_P = (0.05, 0.40, 0.38, 0.12, 0.05)        
_A_NSTAGE_VALS = (-1, 0, 1, 2, 3)                     
_A_NSTAGE_P = (0.05, 0.55, 0.25, 0.10, 0.05)
_A_GRADE_VALS = (1, 2, 3)                             
_A_GRADE_P = (0.20, 0.50, 0.30)
_A_NUM_SCALE = {
    FEATURES_NUM[0]: 1.0,       
    FEATURES_NUM[1]: 1.0,       
    FEATURES_NUM[2]: 0.7,       
    AGE_FEATURE: _A_AGE_STD,    
}
_A_CAT_P: dict[str, tuple[float, ...]] = {
    FEATURES_CAT[0]: (0.45, 0.55),                    
    FEATURES_CAT[1]: (0.55, 0.20, 0.12, 0.08, 0.05),  
    FEATURES_CAT[2]: (0.75, 0.25),                    
    FEATURES_CAT[3]: (0.95, 0.05),                    
    FEATURES_CAT[4]: (0.70, 0.30),                    
}
for _f, _p in _A_CAT_P.items():
    if len(_p) != CAT_CARDINALITIES[_f]:
        raise RuntimeError(
            f"realistic categorical prob vector for {_f!r} has {len(_p)} levels != cardinality "
            f"{CAT_CARDINALITIES[_f]} — update _A_CAT_P if the clinical feature schema changed."
        )


def _prevalence_labels(n: int, prevalence: float, rng: np.random.Generator) -> np.ndarray:
    if not 0.0 < prevalence < 1.0:
        raise ValueError(f"prevalence must be in (0,1), got {prevalence}")
    n_pos = min(max(int(round(n * prevalence)), 1), n - 1)
    y = np.zeros(n, dtype=int)
    y[:n_pos] = 1
    return rng.permutation(y)


def _clinical_row_realistic(rng: np.random.Generator, shift_std: float) -> dict[str, Any]:
    raw = {
        FEATURES_NUM[0]: float(rng.choice(len(_A_TSTAGE_P), p=_A_TSTAGE_P)),
        FEATURES_NUM[1]: float(rng.choice(_A_NSTAGE_VALS, p=_A_NSTAGE_P)),
        FEATURES_NUM[2]: float(rng.choice(_A_GRADE_VALS, p=_A_GRADE_P)),
        AGE_FEATURE: float(np.clip(rng.normal(_A_AGE_MEAN, _A_AGE_STD), *_A_AGE_CLIP)),
    }
    row: dict[str, Any] = {name: raw[name] + shift_std * _A_NUM_SCALE[name] for name in FEATURES_NUM}
    if shift_std > 0.0:  
        row[AGE_FEATURE] = float(np.clip(row[AGE_FEATURE], *_A_AGE_CLIP))
    for name in FEATURES_CAT:
        row[name] = int(rng.choice(CAT_CARDINALITIES[name], p=_A_CAT_P[name]))
    _assert_no_forbidden_keys(row)
    return row


def _generate_realistic(
    prefix: str,
    n: int,
    seed: int,
    cube_size: int,
    channels: str,
    effect_size: float,
    prevalence: float,
) -> Iterator[tuple[str, np.ndarray, dict[str, Any], int]]:
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if cube_size < 1:
        raise ValueError(f"cube_size must be >= 1, got {cube_size}")
    _validate_channels(channels)
    rng = np.random.default_rng(seed)
    labels = _prevalence_labels(n, prevalence, rng)
    for i in range(n):
        label = int(labels[i])
        shift = effect_size if (effect_size > 0.0 and label == 1) else 0.0
        blob = (0.5 * effect_size) if (effect_size > 0.0 and label == 1) else 0.0
        yield f"{prefix}-{i:05d}", _mri_pair(rng, cube_size, blob), _clinical_row_realistic(rng, shift), label


def generate_realistic_negative_control(
    n: int,
    seed: int = 0,
    cube_size: int = 16,
    channels: str = "pre_post",
    prevalence: float = TNBC_PREVALENCE_DEFAULT,
) -> Iterator[tuple[str, np.ndarray, dict[str, Any], int]]:
    yield from _generate_realistic(A_NEG_PREFIX, n, seed, cube_size, channels, 0.0, prevalence)


def generate_realistic_positive_control(
    n: int,
    seed: int = 0,
    cube_size: int = 16,
    channels: str = "pre_post",
    effect_size: float = DEFAULT_EFFECT_SIZE,
    prevalence: float = TNBC_PREVALENCE_DEFAULT,
) -> Iterator[tuple[str, np.ndarray, dict[str, Any], int]]:
    if effect_size <= 0.0:
        raise ValueError(f"positive control needs effect_size > 0, got {effect_size}")
    yield from _generate_realistic(A_POS_PREFIX, n, seed, cube_size, channels, effect_size, prevalence)


def selfcheck() -> int:
    cube = 8
    neg = list(generate_negative_control(6, seed=0, cube_size=cube))
    pos = list(generate_positive_control(6, seed=0, cube_size=cube))
    assert len(neg) == len(pos) == 6
    neg_ids = {pid for pid, *_ in neg}
    pos_ids = {pid for pid, *_ in pos}
    assert neg_ids.isdisjoint(pos_ids), "SYN-NEG / SYN-POS namespaces overlap (DD-2 violated)"
    assert all(pid.startswith(NEG_PREFIX) for pid in neg_ids)
    assert all(pid.startswith(POS_PREFIX) for pid in pos_ids)

    _, mri, row, _ = neg[0]
    assert mri.shape == (2, cube * 2, cube * 2, cube * 2), mri.shape
    assert set(row) == set(FEATURES_NUM) | set(FEATURES_CAT), "row keys != feature schema"
    assert set(row).isdisjoint(FORBIDDEN_FEATURES), "forbidden key leaked into a synthetic row"
    assert AGE_FEATURE in row, "age feature missing from the synthetic row"

    def _num_mean_by_label(stream: list[Any]) -> float:
        c1 = [r[FEATURES_NUM[0]] for _, _, r, y in stream if y == 1]
        c0 = [r[FEATURES_NUM[0]] for _, _, r, y in stream if y == 0]
        return float(np.mean(c1) - np.mean(c0))

    assert abs(_num_mean_by_label(pos)) > abs(_num_mean_by_label(neg)), "positive shift not injected"

    cfg = {"n": 6, "seed": 0, "cube_size": cube, "channels": "pre_post", "effect_size": 1.2,
           "stream_name": "negative_control", "git_commit": "unknown"}
    assert compute_synthetic_manifest_hash(cfg) == compute_synthetic_manifest_hash(dict(cfg)), \
        "manifest hash is not deterministic"
    assert build_manifest(cfg)["manifest_sha256"] == compute_synthetic_manifest_hash(cfg)

    r_neg = list(generate_realistic_negative_control(40, seed=0, cube_size=cube))
    r_pos = list(generate_realistic_positive_control(40, seed=0, cube_size=cube))
    r_neg_ids = {pid for pid, *_ in r_neg}
    r_pos_ids = {pid for pid, *_ in r_pos}
    assert r_neg_ids.isdisjoint(r_pos_ids), "SYN-A-NEG / SYN-A-POS overlap"
    assert r_neg_ids.isdisjoint(neg_ids | pos_ids), "SYN-A-* overlaps the plain SYN-NEG/POS namespace"
    assert all(pid.startswith(A_NEG_PREFIX) for pid in r_neg_ids)
    assert all(pid.startswith(A_POS_PREFIX) for pid in r_pos_ids)
    _, _rmri, r_row, _ = r_neg[0]
    assert set(r_row) == set(FEATURES_NUM) | set(FEATURES_CAT), "realistic row keys != feature schema"
    assert set(r_row).isdisjoint(FORBIDDEN_FEATURES), "forbidden key leaked into a realistic row"
    ages = [row[AGE_FEATURE] for _, _, row, _ in r_neg]  
    assert _A_AGE_CLIP[0] <= min(ages) and max(ages) <= _A_AGE_CLIP[1], (min(ages), max(ages))
    prev = float(np.mean([y for *_, y in r_neg]))  
    assert 0.05 <= prev <= 0.45, f"realistic TNBC prevalence {prev} implausible"
    assert abs(_num_mean_by_label(r_pos)) > abs(_num_mean_by_label(r_neg)), "realistic positive shift not injected"

    print(  
        "synthetic_cohort selfcheck OK — SYN-NEG/SYN-POS/SYN-A-* ID-disjoint, rows match the feature "
        "schema and are forbidden-free, plain + realistic positive controls inject a class shift, "
        "realistic marginals are Duke-like (age range + TNBC prevalence), manifest hash deterministic."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(selfcheck())
