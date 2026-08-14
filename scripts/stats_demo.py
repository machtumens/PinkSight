"""P09 demo: write reports/EXP-fixture/stats.json + append a stats block to metrics.json.

  PYTHONPATH=src .venv/bin/python scripts/stats_demo.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from fixtures.synthetic import classification_fixture  # noqa: E402

from pinksight.stats.compare import stats_report  # noqa: E402

if __name__ == "__main__":
    fx = classification_fixture()
    rep = stats_report(fx["y_true"], fx["probs"]["unimodal"], fx["probs"]["cross_attn"])
    out = ROOT / "reports" / "EXP-fixture"
    out.mkdir(parents=True, exist_ok=True)
    (out / "stats.json").write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n")

    mpath = out / "metrics.json"  # append the stats block if the eval demo has run
    if mpath.exists():
        doc = json.loads(mpath.read_text())
        doc["stats_fusion_vs_unimodal"] = rep
        mpath.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")

    print(f"wrote {out/'stats.json'} (+ appended to metrics.json)")  # noqa: T201
    print(f"ΔAUROC={rep['delta_auroc_mean']}  bootstrap CI={rep['bootstrap_delta_ci95_mean']}  "  # noqa: T201
          f"DeLong p={rep['delong_combined_p']}  power@margin={rep['power_to_detect_margin']}")
    print(f"meets pre-registered margin: {rep['meets_prereg_margin']}")  # noqa: T201
