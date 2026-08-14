"""Stage-0 synthetic generators for the non-Track-A model organs — the ID-disjoint extension of
``synthetic_cohort.py``'s DD-2 control-sentinel pattern so every organ gets its own forward-only 10k
plumbing run:

  * Track-C tabular panels (ADR-0010 standalone LightGBM organs)  -> ``SYN-C-*``
  * Track-B WSI + 6-gene genomics fusion                          -> ``SYN-B-*``   (added by the torch runner)
  * fastMRI-NYU standalone encoder                                -> ``SYN-F-*``   (added by the torch runner)

Same constitution as ``synthetic_cohort.py``: this is fabricated plumbing data, no metric off it is a
scientific result, it is NON-REPORTABLE by construction (``SYNTHETIC_TAG`` + a per-stream manifest hash
gated by ``assert_synthetic_provenance``), every feature key is asserted disjoint from
``FORBIDDEN_FEATURES`` at generation time, and no LOCK is moved. Stream ID prefixes are pairwise
disjoint from ``synthetic_cohort``'s ``SYN-NEG``/``SYN-POS`` and from each other, so two sub-cohorts can
never be confused (DD-2).

Each organ shares ONE forward spine (``fva.shuffle_sentinel.coalition_oof`` -> ``control_verdict``): the
generator is the only per-organ part. Framing is characterisation / at-diagnosis / stratification
language throughout (scanned by ``ci/ledger_lint.py``) — for Track-C the ADR-0010 detection/incidence
framing is the path-scoped carve-out, still synthetic and still non-reportable.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

import numpy as np

from pinksight import FORBIDDEN_FEATURES
from pinksight.data.synthetic_cohort import compute_synthetic_manifest_hash
from pinksight.eval.e2e_report_contract import REPORT_VERSION, SYNTHETIC_TAG

# The disclosed positive-control class shift. Calibrated for the tabular geometry (9 Gaussian axes):
# 0.5 -> real sentinel AUROC ~0.85 (a clear-but-not-degenerate effect) while the permutation-null
# shuffle collapses to ~0.48. A larger shift (e.g. 1.2) drives real to ~0.995 and forms a feature
# cluster that stresses the single-draw shuffle — see scripts/e2e_synthetic_common.permutation_null_oof.
DEFAULT_EFFECT_SIZE = 0.5

# --- Track-C tabular panels -------------------------------------------------------------------------
# Coimbra metabolic/inflammatory panel (confirmed on disk — explore/tabular_risk/src/coimbra_loader.py).
# A second panel (BCSC screening) is a one-line addition: pass its own feature-name tuple.
COIMBRA_FEATURES: tuple[str, ...] = (
    "Age", "BMI", "Glucose", "Insulin", "HOMA", "Leptin", "Adiponectin", "Resistin", "MCP.1",
)

_C_PREFIX = "SYN-C"


def _balanced_labels(n: int, rng: np.random.Generator) -> np.ndarray:
    """A balanced, shuffled label vector carrying ZERO feature information (the negative-control
    invariant) while guaranteeing both classes are present so grouped-CV folds are well-posed."""
    y = np.zeros(n, dtype=int)
    y[: n // 2] = 1
    return rng.permutation(y)


def _assert_forbidden_free(feature_names: Sequence[str]) -> None:
    """A synthetic panel may NEVER carry a label-defining FORBIDDEN feature (LOCK-2 discipline)."""
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
    """One synthetic Track-C tabular sub-cohort: ``(patient_ids, X (n, d), y (n,))``.

    ``effect_size == 0`` => negative control: every feature column is drawn independent of the label,
    so a downstream sentinel AUROC ~ 0.50 is the correct expectation and a tight departure at scale is a
    leak signature. ``effect_size > 0`` => positive control: class-1 rows get a disclosed ``+effect_size``
    shift on every axis (a deliberate small effect, NOT a leak) that the sentinel should recover well
    above chance and lose under a label shuffle. Values are abstract standardized draws, NOT physical
    units. Patient IDs are ``{prefix}-{i:05d}`` — pairwise disjoint from every other synthetic stream.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if effect_size < 0.0:
        raise ValueError(f"effect_size must be >= 0, got {effect_size}")
    _assert_forbidden_free(feature_names)
    rng = np.random.default_rng(seed)
    y = _balanced_labels(n, rng)
    x = rng.normal(0.0, 1.0, (n, len(feature_names))).astype(np.float64)
    if effect_size > 0.0:
        x[y == 1] += effect_size  # disclosed class-conditional shift on class 1 (mirrors _make_separable)
    pids = np.array([f"{prefix}-{i:05d}" for i in range(n)])
    return pids, x, y


def _prevalence_labels(n: int, prevalence: float, rng: np.random.Generator) -> np.ndarray:
    """A shuffled label vector at ~``prevalence`` positives with BOTH classes guaranteed present (so
    grouped-CV folds are well-posed). Carries ZERO feature information — the negative-control invariant;
    prevalence is a realistic cohort marginal, never a signal."""
    if not 0.0 < prevalence < 1.0:
        raise ValueError(f"prevalence must be in (0,1), got {prevalence}")
    n_pos = min(max(int(round(n * prevalence)), 1), n - 1)
    y = np.zeros(n, dtype=int)
    y[:n_pos] = 1
    return rng.permutation(y)


# --- Stream F: fastMRI-NYU standalone encoder (images-only; ADR-0016) -------------------------------
# A synthetic NYU-shaped sub-cohort for the images-only H-char (malignant-vs-benign) encoder. NO
# biomarker/clinical column is EVER an input (the encoder is images-only; ADR-0016 Fix #6). The 51
# verified-normal / H6 anomaly head is NOT modelled — labels are strictly {benign, malignant}, so the
# synthetic F-stream has zero "normal" analog in any batch (the hard-interlock mirror). IDs SYN-F-* are
# pairwise disjoint from every other synthetic stream.
_F_PREFIX = "SYN-F"
FASTMRI_CHANNELS = 2  # pre, post DCE — matches data.dataset.n_channels("pre_post")
FASTMRI_MALIG_PREVALENCE = 0.36  # H-char malignant prevalence (~18/50 shipped NYU test)
# images-only descriptor surface for the report `features` block (never a biomarker column name).
FASTMRI_IMAGE_FEATURES: tuple[str, ...] = ("dce_pre_channel", "dce_post_channel")


def _assert_fastmri_images_only(feature_names: Sequence[str]) -> None:
    """Structural LOCK-2 guard (ADR-0016 Fix #6): the fastMRI biomarker columns must be quarantined in
    ``FORBIDDEN_FEATURES`` and must NEVER appear as a Stream-F feature. The encoder is images-only, so
    the only admissible feature descriptors are image channels — asserted disjoint from the ledger."""
    from pinksight.data.fastmri_nyu import FORBIDDEN_NYU_BIOMARKERS  # lazy: keep this module pandas-free

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
    """A synthetic (2, cube, cube, cube) pre/post DCE pair, IMAGES-ONLY, whole-breast (no lesion crop —
    the shipped fastMRI H-char recipe). ``blob`` (>0 only for a class-1 positive) adds a fixed-location
    enhancing region to ``post`` — the disclosed class signal (mirrors ``synthetic_cohort._mri_pair``),
    not a leak."""
    pre = rng.normal(0.5, 0.1, (cube_size, cube_size, cube_size)).astype(np.float32)
    post = pre + rng.normal(0.0, 0.1, (cube_size, cube_size, cube_size)).astype(np.float32)
    if blob > 0.0:
        q = max(cube_size // 4, 1)
        post[q : 3 * q, q : 3 * q, q : 3 * q] += np.float32(blob)  # disclosed enhancing region
    return np.stack([pre, post], axis=0).astype(np.float32)


def generate_fastmri_stream(
    n: int,
    seed: int,
    cube_size: int = 16,
    effect_size: float = 0.0,
    prevalence: float = FASTMRI_MALIG_PREVALENCE,
    prefix: str = _F_PREFIX,
) -> "Iterator[tuple[str, np.ndarray, int]]":
    """One synthetic fastMRI-NYU sub-cohort: yields ``(patient_id, cube (2,cube,cube,cube), label)`` one
    at a time (no bulk cube array persisted — firewall #5). IMAGES-ONLY. ``label`` = malignant(1) vs
    benign(0); there is NO "normal" analog (H6 / the 51 verified-normals are NOT modelled — the hard
    interlock mirror). ``effect_size == 0`` => negative control (images label-independent, AUROC ~ 0.50);
    ``> 0`` => positive control injecting the disclosed enhancing blob into class-1. IDs ``SYN-F-*``.
    """
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


# --- Stream B: Track-B WSI + genomics fusion (standalone; LOCK-1) ------------------------------------
# A synthetic intra-cohort Track-B sub-cohort: frozen tile-bag embeddings (UNI2-h dim) + a
# pre-registered 6-gene mRNA panel, LumA-vs-Basal characterisation. IDs SYN-B-* pairwise disjoint from
# every other stream. Standalone / intra-cohort ONLY — LOCK-1 cross-institution generalisation stays
# forbidden; nothing here is ever pooled or juxtaposed with a Duke number.
_B_PREFIX = "SYN-B"
UNI_DIM = 1536  # UNI2-h frozen tile-embedding dim (mirrors encode_bags_uni.py:47 FOUNDATION_DIMS["uni"])
# Pre-registered 6-gene mRNA panel — mirrors trackb_fusion_wsi_genomics.MODALITY_C_GENES VERBATIM.
# GATA3 is KEPT (it is the real pre-registered panel). Asserted disjoint from the ESR1/PGR/ERBB2/MKI67
# gene mirrors that already live in FORBIDDEN_FEATURES.
MODALITY_C_GENES: tuple[str, ...] = ("TP53", "PIK3CA", "GATA3", "CDH1", "MAP3K1", "PTEN")
PAM50_BASAL_PREVALENCE = 0.26  # Basal(1) vs LumA(0) prevalence (~165/640)
# variable synthetic bag depth. A NOMINAL range (the plan sanctions "fix a nominal") kept modest so the
# n=10k run stays $0-local-CPU-fast — the mean-pool-over-variable-tiles contract is exercised either way;
# the exact tile magnitude is a throughput knob, not a plumbing-verdict input.
_B_NTILES_RANGE = (128, 512)
# positive-control signal knobs (disclosed): a driver-gene axis shift + a small per-tile bag-mean offset
# that survives mean-pooling. Scaled so the recovered real AUROC is clear-but-not-degenerate (a global
# 1536-dim offset separates trivially, so the per-tile offset is deliberately tiny).
_B_GENE_SHIFT = 3.0    # disclosed shift into ONE driver-gene axis (gene 0 == TP53), log2-ish units
_B_BAG_OFFSET = 0.008  # disclosed per-tile bag-mean offset for class 1 (survives mean-pool over UNI_DIM)


def _assert_genes_forbidden_free(genes: Sequence[str]) -> None:
    """A synthetic genomic panel may NEVER carry an IHC-defining gene mirror (LOCK-2 discipline):
    ESR1/PGR/ERBB2/MKI67 are quarantined in FORBIDDEN_FEATURES and must stay out of every input."""
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
    """One synthetic Track-B sub-cohort: yields ``(patient_id, bag (n_tiles, 1536), genes (6,), label)``
    one patient at a time — the bag is mean-pooled + discarded by the runner, so no bulk bag array is
    ever persisted (firewall #5). ``label`` = PAM50 Basal(1) vs LumA(0). ``effect_size == 0`` => negative
    control (bag + genes label-independent, ΔAUC ~ 0); ``> 0`` => positive control (a disclosed shift
    into ONE driver-gene axis PLUS a small per-tile bag-mean offset for class 1). IDs ``SYN-B-*``.
    """
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
        genes = rng.normal(8.0, 2.0, len(MODALITY_C_GENES)).astype(np.float64)  # log2(x+1)-ish draws
        if effect_size > 0.0 and label == 1:
            bag += np.float32(effect_size * _B_BAG_OFFSET)  # disclosed bag-mean offset (survives pooling)
            genes[0] += effect_size * _B_GENE_SHIFT          # disclosed driver-gene axis shift (TP53)
        yield f"{prefix}-{i:05d}", bag, genes, label


# --- Stream U4M: unified all-4-modality cross-attention fusion (TRAINED organ; SYNTHETIC — NOT RESULT) -
# The ONE synthetic organ in this file whose downstream driver GRADIENT-TRAINS (`.backward()`/`opt.step()`)
# rather than running forward-only. The carve-out that licenses this (E-D0a): the driver trains ONLY the
# `FusionModel` fusion module + its heads on the directly-synthesized embeddings this generator emits —
# zero encoder invoked, zero raw pixel/tile/panel data — per the sanctioned `fusion.py` docstring
# ("only this fusion module + its heads are trained"), NOT the DD-1 firewall's forbidden
# encoder-on-fabricated-images case. This generator itself is pure numpy forward draws (no torch).
#
# The label is a CROSS-MODAL-ONLY interaction: `y = z1 XOR z2` with `z1, z2 ~ Bernoulli(0.5)` i.i.d.,
# `z1` encoded ONLY in the (mri, clinical) pair and `z2` ONLY in the (wsi, genomics) pair — DISJOINT
# modality groups. Because XOR is not linearly separable (Minsky & Papert 1969) and `y` is exactly
# independent of `z1` alone and of `z2` alone, every single modality — and any LINEAR concat of all four
# — is provably at chance in expectation, while a genuine bilinear cross-modal interaction (which
# `CrossAttentionFusion`'s Q·Kᵀ attention structurally provides) can recover it. IDs SYN-U4M-* are
# pairwise disjoint from every other synthetic stream.
_U4M_PREFIX = "SYN-U4M"
# The exact modality_dims dict passed to FusionModel(modality_dims=..., fused_dim=128, n_heads=4):
# mri=512 (mri_encoder.EMBED_DIM), clinical=128 (FTTransformer d_block, n_blocks=2), wsi=1536 (UNI2-h),
# genomics=128 (GenomicsEncoder.embed_dim default).
UNIFIED_FUSION_MODALITY_DIMS: dict[str, int] = {"mri": 512, "clinical": 128, "wsi": 1536, "genomics": 128}
# The disclosed positive-control interaction knobs. shift_width=16: a clearly-recoverable sub-block on the
# first 16 axes of each modality (matches Stream-B's "spread across several axes to survive pooling"
# precedent). effect_size=3.0: a three-SD shift on every informative axis — clear-but-not-degenerate,
# matching Stream-B's `_B_GENE_SHIFT = 3.0` precedent scale. Both are DISCLOSED signals, NOT leaks.
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
    """Private raw generator: yields ``(patient_id, feats, z1, z2, label)`` one patient at a time.

    Exposes the two latent bits ``z1``/``z2`` so ``selfcheck`` can inspect the interaction construction
    directly (present-and-identical z-shift per modality pair, cross-group cleanliness, y == z1 XOR z2)
    without re-deriving them by inference from the embeddings. The PUBLIC generator wraps this and drops
    ``z1``/``z2`` to keep the public contract clean. ONE shared RNG drives the whole cohort (sequential
    per-patient draws), so the stream is a deterministic function of ``seed``.
    """
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
            if z1 == 1:  # z1 lives ONLY in the (mri, clinical) pair — identical disclosed shift on both
                feats["mri"][:shift_width] += np.float32(effect_size)
                feats["clinical"][:shift_width] += np.float32(effect_size)
            if z2 == 1:  # z2 lives ONLY in the (wsi, genomics) pair — DISJOINT from z1's group
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
    """One synthetic unified-fusion sub-cohort: yields ``(patient_id, {"mri":[512], "clinical":[128],
    "wsi":[1536], "genomics":[128]}, label)`` one patient at a time (no bulk array persisted — firewall
    #5). SYNTHETIC — NOT A RESULT. The downstream driver trains ONLY the FusionModel fusion module + its
    heads on these directly-synthesized embeddings (zero encoder, zero images) — the sanctioned
    ``fusion.py`` carve-out, not the DD-1 firewall's forbidden encoder-on-fabricated-images case (E-D0a).

    ``effect_size == 0.0`` => negative control: neither latent is encoded anywhere, so ``y`` carries zero
    information in any modality and the trained fusion should sit at chance. ``effect_size > 0.0`` =>
    positive control: the cross-modal XOR interaction (``z1`` in mri+clinical, ``z2`` in wsi+genomics,
    ``y = z1 XOR z2``) that no unimodal or linear-concat baseline can recover but cross-attention can.

    ``label = z1 XOR z2`` with both latents Bernoulli(0.5) — always ~50/50 balanced by construction. The
    ``prevalence`` param is accepted for interface symmetry with the other ``synthetic_streams.py``
    generators but is a **no-op** here: XOR of two independent fair coins is always balanced, so prevalence
    cannot be steered. IDs ``SYN-U4M-*``, pairwise disjoint from every other synthetic stream.
    """
    _ = prevalence  # documented no-op (XOR of two fair coins is always balanced); accepted for symmetry
    for pid, feats, _z1, _z2, label in _generate_unified_fusion_raw(
        n, seed, effect_size, shift_width, modality_dims, prefix
    ):
        yield pid, feats, label


# --- Shared lean manifest + report (organ-agnostic; NOT the desktop CharacterisationReport surface) --
def build_stream_manifest(config: dict[str, Any]) -> dict[str, Any]:
    """Lean manifest for a non-Track-A organ: sha256 of the sorted-key config + timestamp + a one-line
    CONSORT-style summary. Reuses ``synthetic_cohort.compute_synthetic_manifest_hash`` so the provenance
    gate is byte-identical machinery. ``config`` must carry the fields the hash is computed over."""
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
    """A lean, non-reportable report for a plumbing organ that is NOT the desktop characterisation
    surface (Track-B / Track-C / fastMRI). It carries the provenance gate + the controlVerdict + the
    organ's feature manifest — nothing shaped like a subtype result — and still passes the SAME
    ``assert_synthetic_provenance`` gate (firewall #2: a synthetic number is never a result)."""
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
    """Runnable check (repo convention): the Track-C tabular streams are ID-disjoint from each other and
    forbidden-free; the positive control injects a class-mean shift the negative control lacks; the lean
    manifest hash is deterministic and its report passes the provenance gate. No torch, no disk. Exits 0."""
    from pinksight.eval.e2e_report_contract import assert_synthetic_provenance, control_verdict

    neg_ids, neg_x, neg_y = generate_tabular_stream(20, seed=0, effect_size=0.0)
    pos_ids, pos_x, pos_y = generate_tabular_stream(20, seed=0, effect_size=DEFAULT_EFFECT_SIZE)
    assert set(neg_ids).isdisjoint(pos_ids) is False  # same prefix+seed => same IDs (paired controls)
    assert neg_x.shape == pos_x.shape == (20, len(COIMBRA_FEATURES))
    assert set(COIMBRA_FEATURES).isdisjoint(FORBIDDEN_FEATURES), "a Coimbra feature name is FORBIDDEN"

    # The positive control must actually inject a class-mean shift the negative control lacks.
    def _mean_gap(x: np.ndarray, y: np.ndarray) -> float:
        return float(x[y == 1, 0].mean() - x[y == 0, 0].mean())

    assert abs(_mean_gap(pos_x, pos_y)) > abs(_mean_gap(neg_x, neg_y)), "positive shift not injected"

    cfg = {"organ": "trackc-coimbra", "stream_name": "negative_control", "n": 20, "seed": 0,
           "effect_size": 0.0, "git_commit": "unknown"}
    man = build_stream_manifest(cfg)
    assert man["manifest_sha256"] == compute_synthetic_manifest_hash(cfg), "manifest hash not deterministic"
    report = build_stream_report("trackc-coimbra", "negative_control", man, COIMBRA_FEATURES,
                                 control_verdict("negative_control"))
    assert_synthetic_provenance(report, man["manifest_sha256"])  # correct tag+hash => passes

    # Deliberate-violation: a stripped tag must RAISE.
    from pinksight.eval.e2e_report_contract import SyntheticProvenanceError
    tampered = {**report, "provenance": {**report["provenance"], "datasetTag": "not synthetic"}}
    try:
        assert_synthetic_provenance(tampered, man["manifest_sha256"])
    except SyntheticProvenanceError:
        pass
    else:
        raise AssertionError("provenance gate failed to fire on a stripped tag")

    c_ids = set(neg_ids)  # SYN-C-* namespace (paired controls share IDs); tracked for disjointness below

    # --- Stream F: fastMRI-NYU images-only encoder cohort ---------------------------------------------
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
    _assert_fastmri_images_only(FASTMRI_IMAGE_FEATURES)  # must not raise (images-only, forbidden-free)

    def _blob_region_gap(stream: list[Any]) -> float:  # isolate the disclosed enhancing region
        q = max(6 // 4, 1)

        def region_mean(cube: np.ndarray) -> float:
            return float(cube[1, q : 3 * q, q : 3 * q, q : 3 * q].mean())

        c1 = [region_mean(cube) for _, cube, y in stream if y == 1]
        c0 = [region_mean(cube) for _, cube, y in stream if y == 0]
        return float(np.mean(c1) - np.mean(c0))

    assert abs(_blob_region_gap(f_pos)) > abs(_blob_region_gap(f_neg)), "Stream-F positive blob not injected"

    # --- Stream B: Track-B WSI + genomics cohort -----------------------------------------------------
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

    def _gene0_gap(stream: list[Any]) -> float:  # the disclosed driver-gene axis (gene 0 == TP53)
        c1 = [g[0] for _, _, g, y in stream if y == 1]
        c0 = [g[0] for _, _, g, y in stream if y == 0]
        return float(np.mean(c1) - np.mean(c0))

    assert abs(_gene0_gap(b_pos)) > abs(_gene0_gap(b_neg)), "Stream-B positive gene shift not injected"

    # --- Stream U4M: unified all-4-modality cross-attention fusion cohort (TRAINED organ) -------------
    eff = UNIFIED_EFFECT_SIZE_DEFAULT
    u_raw = list(_generate_unified_fusion_raw(
        400, seed=0, effect_size=eff, shift_width=UNIFIED_SHIFT_WIDTH_DEFAULT,
        modality_dims=UNIFIED_FUSION_MODALITY_DIMS, prefix=_U4M_PREFIX,
    ))
    u_ids = {pid for pid, *_ in u_raw}
    # (a) SYN-U4M-* IDs disjoint from every other synthetic prefix.
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

    # (d) y is exactly z1 XOR z2 for every generated patient.
    assert np.array_equal(u_y, u_z1 ^ u_z2), "U4M label is not exactly z1 XOR z2"

    def _grp_gap(mat: np.ndarray, sel: np.ndarray, axis: int = 0) -> float:
        return float(mat[sel == 1, axis].mean() - mat[sel == 0, axis].mean())

    # (b) NO y-detectable shift on a label-group axis directly (y is exactly independent of z1 alone, so
    #     grouping mri's z1-axis by y shows ~zero gap — only z1/z2-detectable, never y-detectable).
    assert abs(_grp_gap(u_mri, u_y)) < 0.5, f"unexpected y-detectable shift on mri axis0: {_grp_gap(u_mri, u_y)}"
    # (c) the z1-shift IS present and ~identical on mri AND clinical; the z2-shift on wsi AND genomics.
    assert abs(_grp_gap(u_mri, u_z1) - eff) < 1.0, "z1-shift missing/wrong on mri"
    assert abs(_grp_gap(u_clin, u_z1) - eff) < 1.0, "z1-shift missing/wrong on clinical"
    assert abs(_grp_gap(u_wsi, u_z2) - eff) < 1.0, "z2-shift missing/wrong on wsi"
    assert abs(_grp_gap(u_gen, u_z2) - eff) < 1.0, "z2-shift missing/wrong on genomics"
    # cross-group cleanliness: z2 must NOT shift z1's group (mri) and z1 must NOT shift z2's group (wsi).
    assert abs(_grp_gap(u_mri, u_z2)) < 0.5, "z2 leaked into mri (cross-group contamination)"
    assert abs(_grp_gap(u_wsi, u_z1)) < 0.5, "z1 leaked into wsi (cross-group contamination)"
    # negative control (effect_size=0): zero shift anywhere — y carries no information in any modality.
    u_neg = list(_generate_unified_fusion_raw(
        200, seed=0, effect_size=0.0, shift_width=UNIFIED_SHIFT_WIDTH_DEFAULT,
        modality_dims=UNIFIED_FUSION_MODALITY_DIMS, prefix=_U4M_PREFIX,
    ))
    n_mri = np.stack([f["mri"] for _, f, *_ in u_neg])
    n_z1 = np.array([z1 for _, _, z1, _, _ in u_neg])
    assert abs(float(n_mri[n_z1 == 1, 0].mean() - n_mri[n_z1 == 0, 0].mean())) < 0.5, \
        "negative control injected a shift it must not"
    # public wrapper yields (pid, feats, label) only — z1/z2 never exposed on the public contract.
    _u_pub = next(iter(generate_unified_fusion_stream(4, seed=0, effect_size=eff)))
    assert len(_u_pub) == 3 and _u_pub[0].startswith(_U4M_PREFIX), "public U4M contract drifted"
    assert u_ids.isdisjoint({p for p, *_ in f_neg} | {p for p, *_ in b_neg}), "U4M/F/B ID overlap"

    print(  # noqa: T201
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
