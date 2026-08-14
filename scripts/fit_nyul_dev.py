"""Phase A1: fit the Nyul standard histogram on the FULL dev split's pre-contrast volumes.

LOCK-2 (train-only): only dev-split patients feed the fit; the sealed Avanto test never does.
Mirrors build_golden_fixture's fit (pre-contrast N4+iso volumes) but over all 613 resolvable dev
patients instead of the 2-patient fixture — the old landmarks were degenerate (Session-09 risk).
Overwrites configs/nyul_standard_v1.npy; re-run build_golden_fixture afterwards to re-freeze the hash.

    PYTHONPATH=src .venv/bin/python scripts/fit_nyul_dev.py            # full dev
    PYTHONPATH=src .venv/bin/python scripts/fit_nyul_dev.py --limit 30 # smoke
"""

import argparse
from pathlib import Path

import pandas as pd

from pinksight.data.phase_stack import select_phase_stack
from pinksight.data.preprocess import (
    NYUL_LANDMARKS,
    compose,
    fit_nyul,
    load_series,
    n4_correct,
    resample_iso,
    to_rcs,
)

DUKE = Path("data/duke_breast_cancer_mri")


def dev_patients(manifest: Path, exclusions: Path, limit: int | None) -> list[str]:
    m = pd.read_csv(manifest)
    dev = m[m["split"] == "dev"]["patient_id"].tolist()
    excl = (
        set(pd.read_csv(exclusions, sep="\t")["patient_id"]) if exclusions.exists() else set()
    )
    dev = [p for p in dev if p not in excl]
    return dev[:limit] if limit else dev


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest_v1.csv"))
    ap.add_argument("--exclusions", type=Path, default=Path("data/processed/_phase_stack_exclusions.tsv"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    dev = dev_patients(args.manifest, args.exclusions, args.limit)
    chain = compose([n4_correct, resample_iso])
    vols, skipped = [], []
    for i, p in enumerate(dev):
        try:
            vols.append(to_rcs(chain(load_series(select_phase_stack(DUKE / p).pre.path))))
        except Exception as e:  # noqa: BLE001 — a straggler must not abort the fit
            skipped.append((p, str(e)[:80]))
        if (i + 1) % 25 == 0:
            print(f"  N4+iso {i + 1}/{len(dev)}")  # noqa: T201
    nyul = fit_nyul(vols)
    NYUL_LANDMARKS.parent.mkdir(parents=True, exist_ok=True)
    nyul.save_standard_histogram(str(NYUL_LANDMARKS))
    print(f"Nyul fit on {len(vols)} dev pre-contrast volumes -> {NYUL_LANDMARKS}")  # noqa: T201
    if skipped:
        print(f"skipped {len(skipped)}: {skipped[:5]}")  # noqa: T201


if __name__ == "__main__":
    main()
