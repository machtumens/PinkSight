"""FVA (Fusion-Value-Attribution) config dataclass — paths, seeds, split, verdict thresholds.

The thresholds here are FROZEN — they are the exact pre-registered values already living in
``h6_modality_audit.py`` (Shapley efficiency, MRI additive/non-additive) and ``h4_info_ceiling.py``
(ceiling / residual). This module only carries them; it never chooses them (LAW L-1, C2 pre-reg).

Nothing in FVA is a gate on the science: FVA is a *diagnostic protocol*. The verdict flags are
evaluated, never tuned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# --- Pre-registered verdict thresholds (FROZEN — mirror h6 + h4 constants verbatim) ---------------
# h6 (MODALITY_AUDIT) ΔAUC verdict thresholds
NONADDITIVE_MAX = 0.02   # MRI-NON-ADDITIVE: ΔAUC(MRI|clinical) <= +0.02 AND paired p >= 0.05
ADDITIVE_MIN = 0.03      # MRI-ADDITIVE: ΔAUC >= +0.03 AND paired p < 0.05
EFFICIENCY_TOL = 1e-6    # Shapley efficiency axiom: |sum(phi) - (v_all - v_empty)| < 1e-6

# h4 (INFO_CEILING) ceiling/residual verdict thresholds
CEILING_AUROC_MAX = 0.62      # CEILING-CONFIRMED: best AUROC <= 0.62 AND DeLong UB < 0.75
CEILING_UB_MAX = 0.75
RESIDUAL_AUROC_MIN = 0.65     # RESIDUAL-SIGNAL: any estimator >= 0.65 AND DeLong LB > 0.55
RESIDUAL_LB_MIN = 0.55

# h4 replication STOP-gate anchors (structural integrity checks — NOT tuneable parameters)
G1_FLOOR = 0.567          # radiomics-LR anchor (STOP-gate A) — within 0.02
EMB_ANCHOR = 0.514        # latent-probe subtype anchor (STOP-gate B) — within 0.03
RADIOMICS_TOL = 0.02
EMB_TOL = 0.03


@dataclass(frozen=True)
class FVAConfig:
    """Configuration for a Fusion-Value-Attribution diagnostic run.

    Frozen so a config can never be mutated mid-run (immutability — a verdict must be reproducible
    from a single config object). Paths default to the project layout; seeds/n_splits default to the
    LOCK-2 3-seed 5-fold convention.
    """

    feat_csv: Path = ROOT / "data/processed/radiomics_features.csv"
    manifest: Path = ROOT / "data/manifest_v1.csv"
    emb_dir: Path = ROOT / "reports/G2_imaging/embeddings"
    clin_xlsx: Path = ROOT / "data/raw/Clinical_and_Other_Features.xlsx"
    split_yaml: Path = ROOT / "configs/split_v2.yaml"
    out_dir: Path = ROOT / "reports/G2_imaging/FVA"

    seeds: tuple[int, ...] = (0, 1, 2)
    n_splits: int = 5

    # verdict thresholds (frozen; carried, not chosen)
    nonadditive_max: float = NONADDITIVE_MAX
    additive_min: float = ADDITIVE_MIN
    efficiency_tol: float = EFFICIENCY_TOL
    ceiling_auroc_max: float = CEILING_AUROC_MAX
    ceiling_ub_max: float = CEILING_UB_MAX
    residual_auroc_min: float = RESIDUAL_AUROC_MIN
    residual_lb_min: float = RESIDUAL_LB_MIN

    # replication STOP-gate anchors
    g1_floor: float = G1_FLOOR
    emb_anchor: float = EMB_ANCHOR
    radiomics_tol: float = RADIOMICS_TOL
    emb_tol: float = EMB_TOL

    # streams to attribute over (the FVA player set); default is the Track A 3-player set
    players: tuple[str, ...] = ("clinical", "radiomics", "mri")

    # extra per-run metadata (free-form, kept in the report header)
    meta: dict = field(default_factory=dict)
