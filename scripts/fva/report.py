"""FVA report renderer: intermediate diagnostic dicts -> JSON + human-readable text (C2-6).

Accepts the four FVA diagnostic components (Shapley attribution, shuffle sentinel, conditional-
independence MI, information ceiling), evaluates the frozen verdict flags, and emits both a
machine-readable JSON and a human-readable text summary.

JSON keys (per C2-6): ``shapley``, ``shuffle_sentinel``, ``conditional_indep_mi``, ``info_ceiling``,
``verdict_flags``, ``fva_verdict``.

The verdict flags are EVALUATED against the frozen thresholds in FVAConfig — never chosen here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from fva.config import FVAConfig


@dataclass
class FVAReport:
    """A rendered FVA diagnostic result (the ``run_fva`` return value)."""

    shapley: dict = field(default_factory=dict)
    shuffle_sentinel: dict = field(default_factory=dict)
    conditional_indep_mi: dict = field(default_factory=dict)
    info_ceiling: dict = field(default_factory=dict)
    verdict_flags: dict = field(default_factory=dict)
    fva_verdict: str = "UNSET"
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "probe": "FVA",
            "meta": self.meta,
            "shapley": self.shapley,
            "shuffle_sentinel": self.shuffle_sentinel,
            "conditional_indep_mi": self.conditional_indep_mi,
            "info_ceiling": self.info_ceiling,
            "verdict_flags": self.verdict_flags,
            "fva_verdict": self.fva_verdict,
        }


def evaluate_verdict_flags(
    shapley: dict,
    info_ceiling: dict,
    mri_delta: float | None,
    mri_p: float | None,
    cfg: FVAConfig,
) -> dict:
    """Evaluate the frozen FVA verdict flags. Never chooses thresholds — reads them from cfg.

    Returns a dict of boolean/string flags:
      - efficiency_ok: Shapley Σφ == v(N)-v(∅) within cfg.efficiency_tol
      - mri_verdict: MRI-ADDITIVE / MRI-NON-ADDITIVE / AMBIGUOUS (if mri_delta/mri_p supplied)
      - ceiling_verdict: CEILING-CONFIRMED / RESIDUAL-SIGNAL / AMBIGUOUS (if info_ceiling supplied)
    """
    flags: dict = {}
    if shapley:
        flags["efficiency_ok"] = bool(shapley.get("efficiency_ok", False))
        flags["efficiency_gap"] = shapley.get("efficiency_gap")

    if mri_delta is not None and mri_p is not None:
        if mri_delta <= cfg.nonadditive_max and mri_p >= 0.05:
            flags["mri_verdict"] = "MRI-NON-ADDITIVE"
        elif mri_delta >= cfg.additive_min and mri_p < 0.05:
            flags["mri_verdict"] = "MRI-ADDITIVE"
        else:
            flags["mri_verdict"] = "AMBIGUOUS"
        flags["mri_delta_auc"] = round(float(mri_delta), 4)
        flags["mri_paired_p"] = round(float(mri_p), 4)

    if info_ceiling and "best_learned_estimator" in info_ceiling:
        best = info_ceiling["best_learned_estimator"]
        best_auc = best["auroc_mean"]
        best_lb = best["delong_lb_mean"]
        best_ub = best["delong_ub_mean"]
        if best_auc <= cfg.ceiling_auroc_max and best_ub < cfg.ceiling_ub_max:
            flags["ceiling_verdict"] = "CEILING-CONFIRMED"
        elif best_auc >= cfg.residual_auroc_min and best_lb > cfg.residual_lb_min:
            flags["ceiling_verdict"] = "RESIDUAL-SIGNAL"
        else:
            flags["ceiling_verdict"] = "AMBIGUOUS"
        flags["best_auroc"] = round(float(best_auc), 4)

    return flags


def synthesize_verdict(flags: dict) -> str:
    """Combine component verdicts into one FVA headline verdict.

    A fusion claim PASSES the FVA diagnostic only when every present component is consistent with
    the modality being non-additive at the ceiling. The headline is descriptive, not a gate on the
    science — it summarises what the four diagnostics jointly say.
    """
    if not flags.get("efficiency_ok", True):
        return "FVA-INVALID (Shapley efficiency axiom violated)"
    parts = []
    if "mri_verdict" in flags:
        parts.append(flags["mri_verdict"])
    if "ceiling_verdict" in flags:
        parts.append(flags["ceiling_verdict"])
    if not parts:
        return "FVA-INCOMPLETE (no verdict components supplied)"
    return " / ".join(parts)


def render(report: FVAReport, out_dir: Path, tag: str) -> tuple[Path, Path]:
    """Write ``fva_{tag}_REPORT.json`` + ``.txt`` to out_dir; return the two paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"fva_{tag}_REPORT.json"
    txt_path = out_dir / f"fva_{tag}_REPORT.txt"

    json_path.write_text(json.dumps(report.to_dict(), indent=2))

    lines = ["=== FVA (Fusion-Value-Attribution) diagnostic report ===", ""]
    if report.meta:
        lines.append(f"tag: {tag}")
        for k, v in report.meta.items():
            lines.append(f"  {k}: {v}")
        lines.append("")
    if report.shapley:
        lines.append("-- Shapley attribution --")
        lines.append(f"  phi: {report.shapley.get('phi')}")
        lines.append(f"  efficiency_ok: {report.shapley.get('efficiency_ok')} "
                     f"(gap={report.shapley.get('efficiency_gap'):.2e})")
        lines.append("")
    if report.shuffle_sentinel:
        lines.append("-- Shuffle sentinel (LOCK-2 integrity) --")
        for name, d in report.shuffle_sentinel.items():
            lines.append(f"  {name:<10} shuffle={d['shuffle_auroc_mean']:.4f} "
                         f"vs real={d['real_auroc_mean']:.4f} at_chance={d['shuffle_at_chance']}")
        lines.append("")
    if report.conditional_indep_mi:
        lines.append("-- Conditional-independence proxy --")
        for k, v in report.conditional_indep_mi.items():
            if k != "note":
                lines.append(f"  {k}: {v}")
        lines.append("")
    if report.info_ceiling and "best_learned_estimator" in report.info_ceiling:
        best = report.info_ceiling["best_learned_estimator"]
        lines.append("-- Information ceiling --")
        lines.append(f"  best estimator: {best['name']} AUROC={best['auroc_mean']:.4f} "
                     f"CI=[{best['delong_lb_mean']:.4f},{best['delong_ub_mean']:.4f}]")
        lines.append("")
    lines.append("-- Verdict flags --")
    for k, v in report.verdict_flags.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append(f"FVA VERDICT: {report.fva_verdict}")
    txt_path.write_text("\n".join(lines) + "\n")

    return json_path, txt_path
