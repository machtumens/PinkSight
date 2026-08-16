
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from fixtures.synthetic import classification_fixture, ki67_fixture  

from pinksight.eval.ablation import build_metrics_json  

if __name__ == "__main__":
    out = ROOT / "reports" / "EXP-fixture"
    doc = build_metrics_json(classification_fixture(), ki67_fixture(), out, figures=True)
    print(f"wrote {out/'metrics.json'} + figures/")  
    print("ladder AUROC: "  
          + ", ".join(f"{r}={doc['ablation_ladder'][r]['auroc']['value']}"
                      for r in doc["ladder_order"]))
    print(f"ΔAUROC cross-attn vs unimodal = {doc['delta_auroc_crossattn_vs_unimodal']}")  
