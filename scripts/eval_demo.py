"""P08 demo: regenerate reports/EXP-fixture/metrics.json + figures from the synthetic fixture.

  PYTHONPATH=src .venv/bin/python scripts/eval_demo.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from fixtures.synthetic import classification_fixture, ki67_fixture  # noqa: E402

from pinksight.eval.ablation import build_metrics_json  # noqa: E402

if __name__ == "__main__":
    out = ROOT / "reports" / "EXP-fixture"
    doc = build_metrics_json(classification_fixture(), ki67_fixture(), out, figures=True)
    print(f"wrote {out/'metrics.json'} + figures/")  # noqa: T201
    print("ladder AUROC: "  # noqa: T201
          + ", ".join(f"{r}={doc['ablation_ladder'][r]['auroc']['value']}"
                      for r in doc["ladder_order"]))
    print(f"ΔAUROC cross-attn vs unimodal = {doc['delta_auroc_crossattn_vs_unimodal']}")  # noqa: T201
