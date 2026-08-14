"""Track B PoC demo — regenerate a synthetic subtype metrics.json into reports/EXP-fixture-trackb/.

Track B is HARD-GATED (LOCK-6, dormant until Track A clears G5). This produces NO real number:
it is PoC · synthetic · not patient-matched · gate-closed. The cohort is NOT patient-matched to
MRI; the ~84-patient Duke∩TCGA overlap MUST be de-duplicated before any "external" claim.

Reuses pinksight.eval.build_metrics_json (the single source of numbers) on the synthetic Track B
cohort fixture. Run:  PYTHONPATH=src .venv/bin/python scripts/trackb_demo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from fixtures import trackb_synthetic as fx  # noqa: E402
from pinksight.eval import build_metrics_json  # noqa: E402

OUT = Path("reports/EXP-fixture-trackb")
_LABEL = "PoC · synthetic · not patient-matched · gate-closed"


def main() -> None:
    cohort = fx.cohort_fixture()
    rng = np.random.default_rng(0)
    # Ki-67 is NOT a Track B head (Track B is subtype-only); pass a placeholder so build_metrics_json
    # runs, and label it as non-applicable in the manifest note below.
    ki67 = {"y_true": rng.uniform(0, 60, fx.N_PATIENTS),
            "pred": rng.uniform(0, 60, fx.N_PATIENTS), "thresh": 14.0}

    doc = build_metrics_json(cohort, ki67, OUT, figures=True)

    # TICKET-006 (CLAIM LEDGER): build_metrics_json hardcodes a Ki-67 block + a "...+ Ki-67
    # stratification" task string. Track B is billed SUBTYPE-ONLY ([5.2]) — strip the Ki-67 block and
    # rewrite the task BEFORE anything is written to disk, so no forbidden framing ever lands in
    # metrics.json. Fixed at the call site (ablation.py is owned by another agent).
    doc.pop("ki67", None)
    doc["task"] = "subtype characterisation ONLY (Luminal-like vs TNBC)"

    # Stamp the honesty labelling onto the emitted doc + a sidecar manifest.
    doc["track"] = "B (TCGA WSI + genomics)"
    doc["status"] = _LABEL
    doc["gate"] = "HARD-GATED (LOCK-6 / decisions.md [5.1]) — dormant until Track A clears G5"
    doc["ki67_note"] = "Ki-67 block is a placeholder — Track B is SUBTYPE ONLY ([5.2]); ignore it."
    doc["ledger"] = "subtype characterisation ONLY; NO survival / growth-rate / early detection"
    doc["patient_match"] = ("NOT patient-matched to MRI; ~84-patient Duke∩TCGA overlap MUST be "
                            "de-duplicated before any external claim")
    doc["rung_meaning"] = fx.RUNG_MEANING  # Track B modality rungs mapped onto the eval LADDER keys
    (OUT / "metrics.json").write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")

    top = doc["ablation_ladder"]["cross_attn"]["auroc"]
    print(f"[trackb_demo] {_LABEL}")  # noqa: T201 — script output is allowed
    print(f"[trackb_demo] wrote {OUT/'metrics.json'}")  # noqa: T201
    print(f"[trackb_demo] PoC subtype AUROC (wsi+genomics, synthetic): "  # noqa: T201
          f"{top['value']} CI95 {top['ci95']}")


if __name__ == "__main__":
    main()
