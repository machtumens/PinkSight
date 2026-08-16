
from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

import numpy as np

from pinksight import FORBIDDEN_FEATURES
from pinksight.data.synthetic_cohort import compute_synthetic_manifest_hash
from pinksight.eval.e2e_report_contract import REPORT_VERSION, SYNTHETIC_TAG

DEFAULT_EFFECT_SIZE = 0.5

COIMBRA_FEATURES: tuple[str, ...] = (
    "Age", "BMI", "Glucose", "Insulin", "HOMA", "Leptin", "Adiponectin", "Resistin", "MCP.1",
)

_C_PREFIX = "SYN-C"


def _balanced_labels(n: int, rng: np.random.Generator) -> np.ndarray:
    y = np.zeros(n, dtype=int)
    y[: n // 2] = 1
    return rng.permutation(y)


def _assert_forbidden_free(feature_names: Sequence[str]) -> None:
    leaked = set(feature_names) & set(FORBIDDEN_FEATURES)
    if leaked:
        raise ValueError(
            f"LEAKAGE: synthetic tabular panel carries forbidden feature(s) {sorted(leaked)} — "
            "ER/PR/HER2/Ki-67/etc. must never enter classifier inputs (LOCK-2)."
        )


def generate_tabular_stream(
    n: int,
    seed: int,
    feature_names: Sequence[str] = COIMBRA_FEATURES,
    effect_size: float = 0.0,
    prefix: str = _C_PREFIX,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if effect_size < 0.0:
        raise ValueError(f"effect_size must be >= 0, got {effect_size}")
    _assert_forbidden_free(feature_names)
    rng = np.random.default_rng(seed)
    y = _balanced_labels(n, rng)
    x = rng.normal(0.0, 1.0, (n, len(feature_names))).astype(np.float64)
    if effect_size > 0.0:
        x[y == 1] += effect_size  
    pids = np.array([f"{prefix}-{i:05d}" for i in range(n)])
    return pids, x, y


def _prevalence_labels(n: int, prevalence: float, rng: np.random.Generator) -> np.ndarray:
    if not 0.0 < prevalence < 1.0:
        raise ValueError(f"prevalence must be in (0,1), got {prevalence}")
    n_pos = min(max(int(round(n * prevalence)), 1), n - 1)
    y = np.zeros(n, dtype=int)
    y[:n_pos] = 1
    return rng.permutation(y)


_F_PREFIX = "SYN-F"
FASTMRI_CHANNELS = 2  
FASTMRI_MALIG_PREVALENCE = 0.36  
FASTMRI_IMAGE_FEATURES: tuple[str, ...] = ("dce_pre_channel", "dce_post_channel")


def _assert_fastmri_images_only(feature_names: Sequence[str]) -> None:
    from pinksight.data.fastmri_nyu import FORBIDDEN_NYU_BIOMARKERS  

    if not set(FORBIDDEN_NYU_BIOMARKERS) <= set(FORBIDDEN_FEATURES):
        raise ValueError(
            "fastMRI biomarker columns must stay quarantined in FORBIDDEN_FEATURES (ADR-0016 Fix #6)."
        )
    leaked = set(feature_names) & set(FORBIDDEN_FEATURES)
    if leaked:
        raise ValueError(
            f"LEAKAGE: Stream-F images-only descriptor carries forbidden key(s) {sorted(leaked)} — "
            "the fastMRI encoder is images-only; a biomarker column must never become an input (LOCK-2)."
        )


def _fastmri_cube(rng: np.random.Generator, cube_size: int, blob: float) -> np.ndarray:
    pre = rng.normal(0.5, 0.1, (cube_size, cube_size, cube_size)).astype(np.float32)
    post = pre + rng.normal(0.0, 0.1, (cube_size, cube_size, cube_size)).astype(np.float32)
    if blob > 0.0:
        q = max(cube_size // 4, 1)
        post[q : 3 * q, q : 3 * q, q : 3 * q] += np.float32(blob)  
    return np.stack([pre, post], axis=0).astype(np.float32)


def generate_fastmri_stream(
    n: int,
    seed: int,
    cube_size: int = 16,
    effect_size: float = 0.0,
    prevalence: float = FASTMRI_MALIG_PREVALENCE,
    prefix: str = _F_PREFIX,
) -> "Iterator[tuple[str, np.ndarray, int]]":
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if cube_size < 1:
        raise ValueError(f"cube_size must be >= 1, got {cube_size}")
    if effect_size < 0.0:
        raise ValueError(f"effect_size must be >= 0, got {effect_size}")
    _assert_fastmri_images_only(FASTMRI_IMAGE_FEATURES)
    rng = np.random.default_rng(seed)
    labels = _prevalence_labels(n, prevalence, rng)
    for i in range(n):
        label = int(labels[i])
        blob = effect_size if (effect_size > 0.0 and label == 1) else 0.0
        yield f"{prefix}-{i:05d}", _fastmri_cube(rng, cube_size, blob), label


_B_PREFIX = "SYN-B"
UNI_DIM = 1536  
MODALITY_C_GENES: tuple[str, ...] = ("TP53", "PIK3CA", "GATA3", "CDH1", "MAP3K1", "PTEN")
PAM50_BASAL_PREVALENCE = 0.26  
_B_NTILES_RANGE = (128, 512)
_B_GENE_SHIFT = 3.0    
_B_BAG_OFFSET = 0.008  


def _assert_genes_forbidden_free(genes: Sequence[str]) -> None:
    leaked = set(genes) & set(FORBIDDEN_FEATURES)
    if leaked:
        raise ValueError(
            f"LEAKAGE: Stream-B gene panel carries forbidden gene mirror(s) {sorted(leaked)} — "
            "ESR1/PGR/ERBB2/MKI67 must never enter genomic inputs (LOCK-2)."
        )


def generate_wsi_genomics_stream(
    n: int,
    seed: int,
    effect_size: float = 0.0,
    prevalence: float = PAM50_BASAL_PREVALENCE,
    n_tiles_range: tuple[int, int] = _B_NTILES_RANGE,
    prefix: str = _B_PREFIX,
) -> "Iterator[tuple[str, np.ndarray, np.ndarray, int]]":
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if effect_size < 0.0:
        raise ValueError(f"effect_size must be >= 0, got {effect_size}")
    lo, hi = n_tiles_range
    if not (1 <= lo <= hi):
        raise ValueError(f"n_tiles_range must satisfy 1 <= lo <= hi, got {n_tiles_range}")
    _assert_genes_forbidden_free(MODALITY_C_GENES)
    rng = np.random.default_rng(seed)
    labels = _prevalence_labels(n, prevalence, rng)
    for i in range(n):
        label = int(labels[i])
        n_tiles = int(rng.integers(lo, hi + 1))
        bag = rng.normal(0.0, 1.0, (n_tiles, UNI_DIM)).astype(np.float32)
        genes = rng.normal(8.0, 2.0, len(MODALITY_C_GENES)).astype(np.float64)  
        if effect_size > 0.0 and label == 1:
            bag += np.float32(effect_size * _B_BAG_OFFSET)  
            genes[0] += effect_size * _B_GENE_SHIFT          
        yield f"{prefix}-{i:05d}", bag, genes, label


_U4M_PREFIX = "SYN-U4M"
UNIFIED_FUSION_MODALITY_DIMS: dict[str, int] = {"mri": 512, "clinical": 128, "wsi": 1536, "genomics": 128}
UNIFIED_SHIFT_WIDTH_DEFAULT = 16
UNIFIED_EFFECT_SIZE_DEFAULT = 3.0


def _generate_unified_fusion_raw(
    n: int,
    seed: int,
    effect_size: float,
    shift_width: int,
    modality_dims: dict[str, int],
    prefix: str,
) -> "Iterator[tuple[str, dict[str, np.ndarray], int, int, int]]":
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if effect_size < 0.0:
        raise ValueError(f"effect_size must be >= 0, got {effect_size}")
    if shift_width < 1:
        raise ValueError(f"shift_width must be >= 1, got {shift_width}")
    min_dim = min(modality_dims.values())
    if shift_width > min_dim:
        raise ValueError(
            f"shift_width {shift_width} exceeds the smallest modality dim {min_dim} — the injected "
            "sub-block would run off the end of a modality vector."
        )
    rng = np.random.default_rng(seed)
    for i in range(n):
        z1 = int(rng.integers(0, 2))
        z2 = int(rng.integers(0, 2))
        label = z1 ^ z2
        feats = {name: rng.normal(0.0, 1.0, dim).astype(np.float32) for name, dim in modality_dims.items()}
        if effect_size > 0.0:
            if z1 == 1:  
                feats["mri"][:shift_width] += np.float32(effect_size)
                feats["clinical"][:shift_width] += np.float32(effect_size)
            if z2 == 1:  
                feats["wsi"][:shift_width] += np.float32(effect_size)
                feats["genomics"][:shift_width] += np.float32(effect_size)
        yield f"{prefix}-{i:05d}", feats, z1, z2, label


def generate_unified_fusion_stream(
    n: int,
    seed: int,
    effect_size: float = 0.0,
    shift_width: int = UNIFIED_SHIFT_WIDTH_DEFAULT,
    modality_dims: dict[str, int] = UNIFIED_FUSION_MODALITY_DIMS,
    prevalence: float = 0.5,
    prefix: str = _U4M_PREFIX,
) -> "Iterator[tuple[str, dict[str, np.ndarray], int]]":
    _ = prevalence  
    for pid, feats, _z1, _z2, label in _generate_unified_fusion_raw(
        n, seed, effect_size, shift_width, modality_dims, prefix
    ):
        yield pid, feats, label


def build_stream_manifest(config: dict[str, Any]) -> dict[str, Any]:
    required = {"organ", "stream_name", "n", "seed", "effect_size", "git_commit"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"stream manifest config missing required fields: {sorted(missing)}")
    return {
        "manifest_sha256": compute_synthetic_manifest_hash(config),
        "config": dict(config),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": (
            f"SYNTHETIC plumbing organ '{config['organ']}' stream '{config['stream_name']}': "
            f"n={config['n']} patients, seed={config['seed']}, effect_size={config['effect_size']} — "
            "non-reportable, forward-only characterisation/stratification wiring."
        ),
    }


def build_stream_report(
    organ: str,
    stream_name: str,
    manifest: dict[str, Any],
    feature_names: Sequence[str],
    control_verdict_out: dict[str, Any],
) -> dict[str, Any]:
    cfg = manifest["config"]
    return {
        "reportVersion": REPORT_VERSION,
        "organ": organ,
        "stream": stream_name,
        "provenance": {
            "datasetTag": SYNTHETIC_TAG,
            "manifestSha256": manifest["manifest_sha256"],
            "seed": cfg["seed"],
            "gitCommit": cfg.get("git_commit", "unknown"),
            "generatedAt": manifest["generated_at"],
            "nPatients": cfg["n"],
        },
        "features": list(feature_names),
        "controlVerdict": control_verdict_out,
    }


def selfcheck() -> int:
    from pinksight.eval.e2e_report_contract import assert_synthetic_provenance, control_verdict

    neg_ids, neg_x, neg_y = generate_tabular_stream(20, seed=0, effect_size=0.0)
    pos_ids, pos_x, pos_y = generate_tabular_stream(20, seed=0, effect_size=DEFAULT_EFFECT_SIZE)
    assert set(neg_ids).isdisjoint(pos_ids) is False  
    assert neg_x.shape == pos_x.shape == (20, len(COIMBRA_FEATURES))
    assert set(COIMBRA_FEATURES).isdisjoint(FORBIDDEN_FEATURES), "a Coimbra feature name is FORBIDDEN"

    def _mean_gap(x: np.ndarray, y: np.ndarray) -> float:
        return float(x[y == 1, 0].mean() - x[y == 0, 0].mean())

    assert abs(_mean_gap(pos_x, pos_y)) > abs(_mean_gap(neg_x, neg_y)), "positive shift not injected"

    cfg = {"organ": "trackc-coimbra", "stream_name": "negative_control", "n": 20, "seed": 0,
           "effect_size": 0.0, "git_commit": "unknown"}
    man = build_stream_manifest(cfg)
    assert man["manifest_sha256"] == compute_synthetic_manifest_hash(cfg), "manifest hash not deterministic"
    report = build_stream_report("trackc-coimbra", "negative_control", man, COIMBRA_FEATURES,
                                 control_verdict("negative_control"))
    assert_synthetic_provenance(report, man["manifest_sha256"])  

    from pinksight.eval.e2e_report_contract import SyntheticProvenanceError
    tampered = {**report, "provenance": {**report["provenance"], "datasetTag": "not synthetic"}}
    try:
        assert_synthetic_provenance(tampered, man["manifest_sha256"])
    except SyntheticProvenanceError:
        pass
    else:
        raise AssertionError("provenance gate failed to fire on a stripped tag")

    c_ids = set(neg_ids)  

    f_neg = list(generate_fastmri_stream(30, seed=0, cube_size=6, effect_size=0.0))
    f_pos = list(generate_fastmri_stream(30, seed=0, cube_size=6, effect_size=1.5))
    f_ids = {pid for pid, *_ in f_neg}
    assert f_ids.isdisjoint(c_ids), "SYN-F-* overlaps the SYN-C-* namespace (DD-2 violated)"
    assert all(pid.startswith(_F_PREFIX) for pid in f_ids)
    _fpid, fcube, _flabel = f_neg[0]
    assert fcube.shape == (FASTMRI_CHANNELS, 6, 6, 6), fcube.shape
    assert {y for *_, y in f_neg} <= {0, 1}, "Stream-F carries a non-{benign,malignant} label (no H6/normal)"
    f_prev = float(np.mean([y for *_, y in f_neg]))
    assert 0.1 <= f_prev <= 0.6, f"Stream-F malignant prevalence {f_prev} implausible"
    _assert_fastmri_images_only(FASTMRI_IMAGE_FEATURES)  

    def _blob_region_gap(stream: list[Any]) -> float:  
        q = max(6 // 4, 1)

        def region_mean(cube: np.ndarray) -> float:
            return float(cube[1, q : 3 * q, q : 3 * q, q : 3 * q].mean())

        c1 = [region_mean(cube) for _, cube, y in stream if y == 1]
        c0 = [region_mean(cube) for _, cube, y in stream if y == 0]
        return float(np.mean(c1) - np.mean(c0))

    assert abs(_blob_region_gap(f_pos)) > abs(_blob_region_gap(f_neg)), "Stream-F positive blob not injected"

    b_neg = list(generate_wsi_genomics_stream(30, seed=0, effect_size=0.0, n_tiles_range=(5, 10)))
    b_pos = list(generate_wsi_genomics_stream(30, seed=0, effect_size=1.5, n_tiles_range=(5, 10)))
    b_ids = {pid for pid, *_ in b_neg}
    assert b_ids.isdisjoint(c_ids | f_ids), "SYN-B-* overlaps another synthetic stream (DD-2 violated)"
    assert all(pid.startswith(_B_PREFIX) for pid in b_ids)
    _bpid, bbag, bgenes, _blabel = b_neg[0]
    assert bbag.ndim == 2 and bbag.shape[1] == UNI_DIM, bbag.shape
    assert bgenes.shape == (len(MODALITY_C_GENES),), bgenes.shape
    assert MODALITY_C_GENES == ("TP53", "PIK3CA", "GATA3", "CDH1", "MAP3K1", "PTEN"), "panel drifted"
    assert set(MODALITY_C_GENES).isdisjoint(FORBIDDEN_FEATURES), "a Stream-B gene is a forbidden mirror"

    def _gene0_gap(stream: list[Any]) -> float:  
        c1 = [g[0] for _, _, g, y in stream if y == 1]
        c0 = [g[0] for _, _, g, y in stream if y == 0]
        return float(np.mean(c1) - np.mean(c0))

    assert abs(_gene0_gap(b_pos)) > abs(_gene0_gap(b_neg)), "Stream-B positive gene shift not injected"

    eff = UNIFIED_EFFECT_SIZE_DEFAULT
    u_raw = list(_generate_unified_fusion_raw(
        400, seed=0, effect_size=eff, shift_width=UNIFIED_SHIFT_WIDTH_DEFAULT,
        modality_dims=UNIFIED_FUSION_MODALITY_DIMS, prefix=_U4M_PREFIX,
    ))
    u_ids = {pid for pid, *_ in u_raw}
    assert u_ids.isdisjoint(c_ids | f_ids | b_ids), "SYN-U4M-* overlaps another synthetic stream (DD-2)"
    assert all(pid.startswith(_U4M_PREFIX) for pid in u_ids)
    _upid, ufeats0, _uz1, _uz2, _uy0 = u_raw[0]
    assert set(ufeats0) == set(UNIFIED_FUSION_MODALITY_DIMS), "U4M feature keys != modality_dims"
    for _name, _dim in UNIFIED_FUSION_MODALITY_DIMS.items():
        assert ufeats0[_name].shape == (_dim,), (_name, ufeats0[_name].shape)

    def _u_stack(key: str) -> np.ndarray:
        return np.stack([f[key] for _, f, *_ in u_raw])

    u_mri, u_clin, u_wsi, u_gen = (_u_stack(k) for k in ("mri", "clinical", "wsi", "genomics"))
    u_z1 = np.array([z1 for _, _, z1, _, _ in u_raw])
    u_z2 = np.array([z2 for _, _, _, z2, _ in u_raw])
    u_y = np.array([lab for *_, lab in u_raw])

    assert np.array_equal(u_y, u_z1 ^ u_z2), "U4M label is not exactly z1 XOR z2"

    def _grp_gap(mat: np.ndarray, sel: np.ndarray, axis: int = 0) -> float:
        return float(mat[sel == 1, axis].mean() - mat[sel == 0, axis].mean())

    assert abs(_grp_gap(u_mri, u_y)) < 0.5, f"unexpected y-detectable shift on mri axis0: {_grp_gap(u_mri, u_y)}"
    assert abs(_grp_gap(u_mri, u_z1) - eff) < 1.0, "z1-shift missing/wrong on mri"
    assert abs(_grp_gap(u_clin, u_z1) - eff) < 1.0, "z1-shift missing/wrong on clinical"
    assert abs(_grp_gap(u_wsi, u_z2) - eff) < 1.0, "z2-shift missing/wrong on wsi"
    assert abs(_grp_gap(u_gen, u_z2) - eff) < 1.0, "z2-shift missing/wrong on genomics"
    assert abs(_grp_gap(u_mri, u_z2)) < 0.5, "z2 leaked into mri (cross-group contamination)"
    assert abs(_grp_gap(u_wsi, u_z1)) < 0.5, "z1 leaked into wsi (cross-group contamination)"
    u_neg = list(_generate_unified_fusion_raw(
        200, seed=0, effect_size=0.0, shift_width=UNIFIED_SHIFT_WIDTH_DEFAULT,
        modality_dims=UNIFIED_FUSION_MODALITY_DIMS, prefix=_U4M_PREFIX,
    ))
    n_mri = np.stack([f["mri"] for _, f, *_ in u_neg])
    n_z1 = np.array([z1 for _, _, z1, _, _ in u_neg])
    assert abs(float(n_mri[n_z1 == 1, 0].mean() - n_mri[n_z1 == 0, 0].mean())) < 0.5, \
        "negative control injected a shift it must not"
    _u_pub = next(iter(generate_unified_fusion_stream(4, seed=0, effect_size=eff)))
    assert len(_u_pub) == 3 and _u_pub[0].startswith(_U4M_PREFIX), "public U4M contract drifted"
    assert u_ids.isdisjoint({p for p, *_ in f_neg} | {p for p, *_ in b_neg}), "U4M/F/B ID overlap"

    print(  
        "synthetic_streams selfcheck OK — Track-C/F/B/U4M streams ID-disjoint + forbidden-free + right "
        "shape, each positive control injects a disclosed class shift the negative lacks (tabular / "
        "images-only blob / driver-gene axis / cross-modal XOR interaction), Stream-F carries no normal "
        "analog, U4M label is exactly z1 XOR z2 with z1 in mri+clinical and z2 in wsi+genomics (disjoint "
        "groups, no cross-group leak, no y-detectable marginal), lean manifest hash deterministic, "
        "provenance gate passes/raises."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(selfcheck())
