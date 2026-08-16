
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from fixtures import trackb_synthetic as fx  
from pinksight.eval import build_metrics_json  

OUT = Path("reports/EXP-fixture-trackb")
_LABEL = "PoC · synthetic · not patient-matched · gate-closed"


def main() -> None:
    cohort = fx.cohort_fixture()
    rng = np.random.default_rng(0)
    ki67 = {"y_true": rng.uniform(0, 60, fx.N_PATIENTS),
            "pred": rng.uniform(0, 60, fx.N_PATIENTS), "thresh": 14.0}

    doc = build_metrics_json(cohort, ki67, OUT, figures=True)

    doc.pop("ki67", None)
    doc["task"] = "subtype characterisation ONLY (Luminal-like vs TNBC)"

    doc["track"] = "B (TCGA WSI + genomics)"
    doc["status"] = _LABEL
    doc["gate"] = "HARD-GATED (LOCK-6 / decisions.md [5.1]) — dormant until Track A clears G5"
    doc["ki67_note"] = "Ki-67 block is a placeholder — Track B is SUBTYPE ONLY ([5.2]); ignore it."
    doc["ledger"] = "subtype characterisation ONLY; NO survival / growth-rate / early detection"
    doc["patient_match"] = ("NOT patient-matched to MRI; ~84-patient Duke∩TCGA overlap MUST be "
                            "de-duplicated before any external claim")
    doc["rung_meaning"] = fx.RUNG_MEANING  
    (OUT / "metrics.json").write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")

    top = doc["ablation_ladder"]["cross_attn"]["auroc"]
    print(f"[trackb_demo] {_LABEL}")  
    print(f"[trackb_demo] wrote {OUT/'metrics.json'}")  
    print(f"[trackb_demo] PoC subtype AUROC (wsi+genomics, synthetic): "  
          f"{top['value']} CI95 {top['ci95']}")


if __name__ == "__main__":
    main()
