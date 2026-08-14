"""Piece A — PinkSight cohort/modality -> harness dispatcher (routing table only, ZERO gradient).

This is a static lookup table over the project's EXISTING, separately-trained CPU harnesses. It maps
a (cohort, available-modality-set) tuple to the harness script that owns that organ. It is NOT a
checkpoint zoo, NOT a trained network, and holds NO trainable parameters.

THE "NO >1-COHORT GRADIENT" CONTRACT (ADR-0011 precedent — generalised, never relaxed):
  Every entry is INFERENCE-TIME ROUTING ONLY. No arrow trains a shared parameter across cohorts.
  `cross_cohort_gradient` is `False` on every result, by construction — this table is the same
  "no >1-cohort gradient" rule ADR-0011 established for the pCR task-head slot (frozen-trunk /
  separate-weights), expressed as a routing table instead of restated per-organ. Each organ is a
  standalone, own-cohort module; the dispatcher only *selects* one, it never fuses them.

Framing (LOCK-1): routing a Duke (Track A) cohort and a TCGA-BRCA (Track B) cohort through the same
table does NOT juxtapose their numbers or imply cross-institution transfer — the dispatcher returns a
script path, never a comparison.

Unknown or NOT-WIRED combos return the "NOT WIRED — confirm path or backlog" sentinel and never raise
or silently guess a path.

Run:  uv run python scripts/pinksight_dispatch.py --selfcheck
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

_NOT_WIRED = "NOT WIRED — confirm path or backlog"

Status = Literal["WIRED", "NOT WIRED — confirm path or backlog"]


@dataclass(frozen=True)
class DispatchResult:
    """Resolution of one (cohort, modalities) lookup.

    harness_script: path to the owning harness (None when not wired).
    status: "WIRED" or the NOT-WIRED sentinel.
    cross_cohort_gradient: ALWAYS False — documents the routing-only contract (no arrow trains a
      shared parameter across cohorts).
    note: short human-readable descriptor / not-wired reason (additive; not part of the routing key).
    """

    harness_script: str | None
    status: Status
    cross_cohort_gradient: bool = False
    note: str = ""


# WIRED entries — (cohort, frozenset(modalities)) -> existing harness script path. Statuses in the
# comments are pulled from docs/map/e5-novel-heads-arms.md's scoreboard (cited, not re-derived here).
COHORT_HARNESS_REGISTRY: dict[tuple[str, frozenset[str]], str] = {
    ("duke", frozenset({"mri", "clinical"})): "scripts/train_g3_hierarchical.py",  # Track A G3 (BUILT)
    ("tcga_brca", frozenset({"wsi"})): "scripts/trackb_mil_cv.py",  # UNI2-h ABMIL (BUILT, 0.9675)
    ("tcga_brca", frozenset({"wsi", "genomics"})): "scripts/trackb_fusion_wsi_genomics.py",  # Piece B
    ("metabric", frozenset({"purity"})): "scripts/novel_heads/arm2_purity_admixture.py",  # arm2 GREENLIGHT
    ("tcga_brca", frozenset({"hrd"})): "scripts/novel_heads/arm4_hrd_brcaness.py",  # arm4 GREENLIGHT
    ("cptac_brca", frozenset({"proteomic"})): "scripts/novel_heads/arm8_proteogenomic_discordance.py",  # arm8 PILOT (tension)
    ("cdd_cesm", frozenset({"cesm"})): "scripts/novel_heads/arm1_cesm_iodine_radiomics.py",  # arm1 GREENLIGHT
    ("cmmd", frozenset({"ffdm"})): "scripts/novel_heads/arm9_cmmd_modality_transfer.py",  # arm9 KILL (null-replication)
    # ADR-0016 (RATIFIED) standalone NYU DCE encoder. NYU-INTERNAL NO-GO honest null (H-char
    # malig-vs-benign ensemble AUROC 0.599, DeLong [0.4303, 0.7676], lower bound < 0.60 -> NO-GO;
    # EVL-confirmed leak-free). Own-cohort, NYU-only, no Duke gradient (never juxtaposed with Duke).
    ("fastmri_nyu", frozenset({"dce"})): "scripts/train_fastmri_nyu.py",
    # ADR-0010 (Scope C) public-benchmark tabular ENSEMBLE companion panel (Coimbra/BCSC/METABRIC);
    # ensemble NOT fusion; zero shared patients; the entrypoint is a routing wrapper (no gradient).
    ("track_c", frozenset({"tabular"})): "scripts/track_c_tabular_panel.py",
}

# Explicit NOT-WIRED combos — documented so their absence is visible in the table, not silently
# missing. Each maps to the reason. `dispatch()` returns the NOT-WIRED sentinel for these.
#   - duke clinical+recurrence : the ADR-0006 clinical-companion recurrence-stratification organ has
#     no standalone scripts/ entrypoint on disk.
#   - duke clinical : E9 finding (04-08-26) — the plan named `train_imaging_mvp.py --clinical-only-path`,
#     but train_imaging_mvp.py is a PURE-IMAGING script with NO --clinical-only-path flag (argparse has
#     no clinical-only mode), and no other clean clinical-only Duke *training* entrypoint exists in
#     scripts/. Marked NOT WIRED rather than left pointing at a non-existent CLI contract.
NOT_WIRED_COMBOS: dict[tuple[str, frozenset[str]], str] = {
    ("duke", frozenset({"clinical", "recurrence"})): "ADR-0006 clinical-companion organ has no standalone scripts/ entrypoint",
    ("duke", frozenset({"clinical"})): "E9: train_imaging_mvp.py has no --clinical-only-path flag (pure-imaging script); no clinical-only Duke training entrypoint found",
}


def dispatch(cohort: str, modalities: Iterable[str]) -> DispatchResult:
    """Resolve (cohort, modalities) to a harness. Normalises modalities to a frozenset, looks up the
    registry, and returns WIRED with the script path or the NOT-WIRED sentinel. Never raises on an
    unknown key; never silently guesses a path.
    """
    key = (cohort, frozenset(modalities))
    if key in COHORT_HARNESS_REGISTRY:
        return DispatchResult(
            harness_script=COHORT_HARNESS_REGISTRY[key],
            status="WIRED",
            cross_cohort_gradient=False,
            note="inference-time routing only (no >1-cohort gradient)",
        )
    if key in NOT_WIRED_COMBOS:
        return DispatchResult(
            harness_script=None,
            status=_NOT_WIRED,
            cross_cohort_gradient=False,
            note=NOT_WIRED_COMBOS[key],
        )
    return DispatchResult(
        harness_script=None,
        status=_NOT_WIRED,
        cross_cohort_gradient=False,
        note="unknown (cohort, modalities) combo — not in the registry; never silently guessed",
    )


def selfcheck() -> int:
    """Resolve every registry key and assert every WIRED script path exists on disk; confirm the
    documented NOT-WIRED combos and an unknown combo degrade gracefully. No data required.
    """
    n_wired = 0
    for (cohort, modalities), script in COHORT_HARNESS_REGISTRY.items():
        res = dispatch(cohort, modalities)
        assert res.status == "WIRED", f"registry entry not WIRED: ({cohort}, {sorted(modalities)})"
        assert res.harness_script == script
        assert res.cross_cohort_gradient is False
        assert Path(res.harness_script).exists(), (
            f"WIRED entry ({cohort}, {sorted(modalities)}) -> {res.harness_script} does NOT exist on disk"
        )
        n_wired += 1

    for cohort, modalities in NOT_WIRED_COMBOS:
        res = dispatch(cohort, modalities)
        assert res.status == _NOT_WIRED and res.harness_script is None

    unknown = dispatch("nonexistent_cohort", frozenset({"nonexistent_modality"}))
    assert unknown.status == _NOT_WIRED and unknown.harness_script is None

    print(
        f"[pinksight_dispatch] selfcheck OK — {n_wired} WIRED entries all resolve to existing script "
        f"paths; {len(NOT_WIRED_COMBOS)} documented NOT-WIRED combos + unknown combos degrade "
        f"gracefully (no raise, no silent guess); cross_cohort_gradient=False on every result."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="PinkSight cohort/modality -> harness dispatcher (routing only)")
    ap.add_argument("--selfcheck", action="store_true", help="resolve every registry key + assert paths exist")
    args = ap.parse_args(argv)
    if args.selfcheck:
        return selfcheck()
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
