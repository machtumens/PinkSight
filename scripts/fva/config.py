
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

NONADDITIVE_MAX = 0.02   
ADDITIVE_MIN = 0.03      
EFFICIENCY_TOL = 1e-6    

CEILING_AUROC_MAX = 0.62      
CEILING_UB_MAX = 0.75
RESIDUAL_AUROC_MIN = 0.65     
RESIDUAL_LB_MIN = 0.55

G1_FLOOR = 0.567          
EMB_ANCHOR = 0.514        
RADIOMICS_TOL = 0.02
EMB_TOL = 0.03


@dataclass(frozen=True)
class FVAConfig:

    feat_csv: Path = ROOT / "data/processed/radiomics_features.csv"
    manifest: Path = ROOT / "data/manifest_v1.csv"
    emb_dir: Path = ROOT / "reports/G2_imaging/embeddings"
    clin_xlsx: Path = ROOT / "data/raw/Clinical_and_Other_Features.xlsx"
    split_yaml: Path = ROOT / "configs/split_v2.yaml"
    out_dir: Path = ROOT / "reports/G2_imaging/FVA"

    seeds: tuple[int, ...] = (0, 1, 2)
    n_splits: int = 5

    nonadditive_max: float = NONADDITIVE_MAX
    additive_min: float = ADDITIVE_MIN
    efficiency_tol: float = EFFICIENCY_TOL
    ceiling_auroc_max: float = CEILING_AUROC_MAX
    ceiling_ub_max: float = CEILING_UB_MAX
    residual_auroc_min: float = RESIDUAL_AUROC_MIN
    residual_lb_min: float = RESIDUAL_LB_MIN

    g1_floor: float = G1_FLOOR
    emb_anchor: float = EMB_ANCHOR
    radiomics_tol: float = RADIOMICS_TOL
    emb_tol: float = EMB_TOL

    players: tuple[str, ...] = ("clinical", "radiomics", "mri")

    meta: dict = field(default_factory=dict)
