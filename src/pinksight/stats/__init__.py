"""P09 statistics harness — paired bootstrap ΔAUC + DeLong across folds/seeds + power/MDE."""

from pinksight.stats.compare import paired_bootstrap_delta_auc, stats_report
from pinksight.stats.temperature import (
    overconfident_selfcheck,
    recompute_floor_ece,
    temperature_scale_ece,
)

__all__ = [
    "paired_bootstrap_delta_auc",
    "stats_report",
    "temperature_scale_ece",
    "overconfident_selfcheck",
    "recompute_floor_ece",
]
