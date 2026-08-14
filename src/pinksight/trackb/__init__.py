"""Track B (TCGA WSI + genomics) — fixture-tested, DORMANT PoC harness (decisions.md [5.1]-[5.4]).

HARD-GATED (LOCK-6): no real TCGA data, no real UNI/CONCH weights, no training until Track A clears
external validation (G5). Subtype characterisation ONLY — never survival / growth-rate / early
detection. NOT patient-matched to MRI; the ~84-patient Duke∩TCGA overlap MUST be de-duplicated
before any "external" claim.
"""

from pinksight.trackb.gate import (
    TRACKB_GATE_OPEN,
    TrackBGateClosedError,
    assert_gate_open,
)
from pinksight.trackb.genomics import GenomicsEncoder
from pinksight.trackb.head import TrackBSubtypeHead
from pinksight.trackb.mil import GatedAttentionMIL
from pinksight.trackb.modality_dropout import ModalityDropoutFusion
from pinksight.trackb.tiles import (
    FOUNDATION_DIMS,
    SyntheticTileEncoder,
    UNITileEncoder,
    load_tile_encoder,
)

__all__ = [
    "TRACKB_GATE_OPEN",
    "TrackBGateClosedError",
    "assert_gate_open",
    "GenomicsEncoder",
    "TrackBSubtypeHead",
    "GatedAttentionMIL",
    "ModalityDropoutFusion",
    "FOUNDATION_DIMS",
    "SyntheticTileEncoder",
    "UNITileEncoder",
    "load_tile_encoder",
]
