
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

    harness_script: str | None
    status: Status
    cross_cohort_gradient: bool = False
    note: str = ""


COHORT_HARNESS_REGISTRY: dict[tuple[str, frozenset[str]], str] = {
    ("duke", frozenset({"mri", "clinical"})): "scripts/train_g3_hierarchical.py",  
    ("tcga_brca", frozenset({"wsi"})): "scripts/trackb_mil_cv.py",  
    ("tcga_brca", frozenset({"wsi", "genomics"})): "scripts/trackb_fusion_wsi_genomics.py",  
    ("metabric", frozenset({"purity"})): "scripts/novel_heads/arm2_purity_admixture.py",  
    ("tcga_brca", frozenset({"hrd"})): "scripts/novel_heads/arm4_hrd_brcaness.py",  
    ("cptac_brca", frozenset({"proteomic"})): "scripts/novel_heads/arm8_proteogenomic_discordance.py",  
    ("cdd_cesm", frozenset({"cesm"})): "scripts/novel_heads/arm1_cesm_iodine_radiomics.py",  
    ("cmmd", frozenset({"ffdm"})): "scripts/novel_heads/arm9_cmmd_modality_transfer.py",  
    ("fastmri_nyu", frozenset({"dce"})): "scripts/train_fastmri_nyu.py",
    ("track_c", frozenset({"tabular"})): "scripts/track_c_tabular_panel.py",
}

NOT_WIRED_COMBOS: dict[tuple[str, frozenset[str]], str] = {
    ("duke", frozenset({"clinical", "recurrence"})): "ADR-0006 clinical-companion organ has no standalone scripts/ entrypoint",
    ("duke", frozenset({"clinical"})): "E9: train_imaging_mvp.py has no --clinical-only-path flag (pure-imaging script); no clinical-only Duke training entrypoint found",
}


def dispatch(cohort: str, modalities: Iterable[str]) -> DispatchResult:
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
