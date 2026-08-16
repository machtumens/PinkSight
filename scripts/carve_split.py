
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from audit_ki67 import load as load_clinical_raw  

SUBTYPE_MAP = {0: "luminal_like", 3: "tnbc"}
TNBC = "tnbc"
DEFAULT_HOLDOUT = ["Avanto"]
TNBC_PREV_TOLERANCE = 0.05  


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clinical_subtypes(path: Path) -> pd.DataFrame:
    df = load_clinical_raw(path)
    out = pd.DataFrame(
        {
            "patient_id": df["Patient ID"].astype(str).str.strip(),
            "subtype": pd.to_numeric(df["Mol Subtype"], errors="coerce").map(SUBTYPE_MAP),
        }
    )
    return out.dropna(subset=["subtype"]).reset_index(drop=True)


def patient_scanner(path: Path) -> pd.DataFrame:
    m = pd.read_csv(path, dtype=str)
    m["scanner"] = m["ManufacturerModelName"].astype(str).str.strip()
    m = m[~m["scanner"].str.startswith("git@")]
    mode = m.groupby("PatientID")["scanner"].agg(lambda s: s.value_counts().index[0])
    multi = int((m.groupby("PatientID")["scanner"].nunique() > 1).sum())
    out = mode.rename_axis("patient_id").reset_index()
    out.attrs["multi_scanner_patients"] = multi
    return out


def build_manifest(clinical: pd.DataFrame, scanners: pd.DataFrame) -> pd.DataFrame:
    man = clinical.merge(scanners, on="patient_id", how="inner")
    man.attrs["multi_scanner_patients"] = scanners.attrs.get("multi_scanner_patients", 0)
    return man.sort_values("patient_id").reset_index(drop=True)


def scanner_table(man: pd.DataFrame) -> pd.DataFrame:
    g = man.assign(is_tnbc=man.subtype.eq(TNBC)).groupby("scanner")
    t = g.agg(n=("patient_id", "size"), tnbc=("is_tnbc", "sum"))
    t["luminal_like"] = t.n - t.tnbc
    t["tnbc_prev"] = (t.tnbc / t.n).round(3)
    return t.sort_values("n", ascending=False)


def carve(man: pd.DataFrame, holdout_scanners: list[str]) -> pd.DataFrame:
    unknown = set(holdout_scanners) - set(man.scanner.unique())
    if unknown:
        raise ValueError(f"holdout scanner(s) not in data: {sorted(unknown)}")
    man = man.copy()
    man["split"] = man.scanner.isin(holdout_scanners).map({True: "test", False: "dev"})
    return man


def _balance(man: pd.DataFrame, split: str) -> dict:
    s = man[man.split == split]
    n = len(s)
    tnbc = int(s.subtype.eq(TNBC).sum())
    return {"n": n, "luminal_like": n - tnbc, "tnbc": tnbc,
            "tnbc_prev": round(tnbc / n, 3) if n else 0.0}


def freeze(man: pd.DataFrame, holdout: list[str], clin_path: Path, scan_path: Path,
           seed: int) -> dict:
    overall_prev = man.subtype.eq(TNBC).mean()
    test_bal = _balance(man, "test")
    dev_bal = _balance(man, "dev")
    drift = abs(test_bal["tnbc_prev"] - overall_prev)
    return {
        "schema": "pinksight.split/v2",
        "axis": "scanner",  
        "seed": seed,
        "holdout_scanners": holdout,
        "subtype_scope": {"luminal_like": "Mol Subtype == 0", "tnbc": "Mol Subtype == 3",
                          "dropped": "Mol Subtype in {1,2} (out of LOCK-3 scope)"},
        "provenance": {"clinical_sha256": sha256(clin_path),
                       "scanners_sha256": sha256(scan_path)},
        "caveat_multi_scanner_patients": man.attrs.get("multi_scanner_patients", 0),
        "tnbc_prev_overall": round(float(overall_prev), 3),
        "tnbc_prev_drift_test": round(float(drift), 3),
        "tnbc_balance_ok": bool(drift <= TNBC_PREV_TOLERANCE),
        "counts": {"total_in_scope": len(man), "dev": dev_bal, "test": test_bal},
        "test": sorted(man.loc[man.split == "test", "patient_id"]),
        "dev": sorted(man.loc[man.split == "dev", "patient_id"]),
    }


def run(clin_path: Path, scan_path: Path, holdout: list[str], seed: int,
        out_manifest: Path, out_yaml: Path) -> int:
    man = carve(build_manifest(clinical_subtypes(clin_path), patient_scanner(scan_path)), holdout)
    frozen = freeze(man, holdout, clin_path, scan_path, seed)

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    man[["patient_id", "scanner", "subtype", "split"]].to_csv(out_manifest, index=False)
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    out_yaml.write_text(yaml.safe_dump(frozen, sort_keys=False))

    print("=== per-scanner table (in-scope: luminal-like vs TNBC) ===")
    print(scanner_table(man).to_string())
    print(f"\nholdout (test) = {holdout}")
    print(f"  test: {frozen['counts']['test']}")
    print(f"  dev : {frozen['counts']['dev']}")
    print(f"  TNBC prev overall={frozen['tnbc_prev_overall']} "
          f"test={frozen['counts']['test']['tnbc_prev']} "
          f"drift={frozen['tnbc_prev_drift_test']} "
          f"({'OK' if frozen['tnbc_balance_ok'] else 'WARN >5pp — pick another scanner'})")
    print(f"\nwrote {out_manifest}  and  {out_yaml}")
    return 0 if frozen["tnbc_balance_ok"] else 1


def selfcheck() -> int:
    clin = pd.DataFrame({"patient_id": ["p1", "p2", "p3", "p4"],
                         "subtype": ["luminal_like", "tnbc", "luminal_like", "tnbc"]})
    scan = pd.DataFrame({"patient_id": ["p1", "p2", "p3", "p4"],
                         "scanner": ["A", "A", "B", "B"]})
    scan.attrs["multi_scanner_patients"] = 0
    man = carve(build_manifest(clin, scan), ["B"])
    assert sorted(man.loc[man.split == "test", "patient_id"]) == ["p3", "p4"], man
    assert sorted(man.loc[man.split == "dev", "patient_id"]) == ["p1", "p2"], man
    test, dev = set(man[man.split == "test"].patient_id), set(man[man.split == "dev"].patient_id)
    assert not (test & dev), "patient leaked across splits"
    t = scanner_table(man)
    assert t.loc["B", "tnbc"] == 1 and t.loc["B", "tnbc_prev"] == 0.5, t
    try:
        carve(man, ["NOPE"])
    except ValueError:
        pass
    else:
        raise AssertionError("unknown scanner should raise")
    print("selfcheck OK — carve, scanner table, disjointness, unknown-scanner guard")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clinical", type=Path,
                    default=Path("data/raw/Clinical_and_Other_Features.xlsx"))
    ap.add_argument("--scanners", type=Path, default=Path("data/raw/duke_scan_metadata.csv"))
    ap.add_argument("--holdout", nargs="+", default=DEFAULT_HOLDOUT,
                    help="scanner(s) to hold out entirely (ManufacturerModelName)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-manifest", type=Path, default=Path("data/manifest_v1.csv"))
    ap.add_argument("--out-yaml", type=Path, default=Path("configs/split_v2.yaml"))
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args(argv)

    if args.selfcheck:
        return selfcheck()
    for p in (args.clinical, args.scanners):
        if not p.exists():
            print(f"Not found: {p} (data is git-ignored / not downloaded).", file=sys.stderr)
            return 2
    return run(args.clinical, args.scanners, args.holdout, args.seed,
               args.out_manifest, args.out_yaml)


if __name__ == "__main__":
    raise SystemExit(main())
