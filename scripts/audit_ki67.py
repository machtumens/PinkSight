
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

KI67_PAT = re.compile(r"ki.?67", re.IGNORECASE)
SUBTYPE_PAT = re.compile(r"subtype|luminal|triple|tnbc|molecular", re.IGNORECASE)
MRI_PAT = re.compile(r"mri|dce|scan|series", re.IGNORECASE)


def numeric_ki67(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(
        series.astype(str).str.extract(r"(-?\d+\.?\d*)", expand=False), errors="coerce"
    )
    return vals[(vals >= 0) & (vals <= 100)]


def find_cols(df: pd.DataFrame, pat: re.Pattern) -> list[str]:
    return [c for c in df.columns if pat.search(str(c))]


def audit(df: pd.DataFrame) -> dict:
    total = len(df)
    mri_cols = find_cols(df, MRI_PAT)
    subtype_cols = find_cols(df, SUBTYPE_PAT)
    ki67_cols = find_cols(df, KI67_PAT)

    has_mri = df[mri_cols].notna().any(axis=1).sum() if mri_cols else total  
    has_subtype = df[subtype_cols].notna().any(axis=1).sum() if subtype_cols else 0
    ki67_usable = 0
    if ki67_cols:
        mask = pd.Series(False, index=df.index)
        for c in ki67_cols:
            mask |= numeric_ki67(df[c]).reindex(df.index).notna()
        ki67_usable = int(mask.sum())

    subtype_dist: dict[str, int] = {}
    if subtype_cols:
        vc = df[subtype_cols[0]].value_counts(dropna=True)
        subtype_dist = {str(k): int(v) for k, v in vc.items()}

    return {
        "total_rows": total,
        "mri_cols": mri_cols,
        "subtype_cols": subtype_cols,
        "ki67_cols": ki67_cols,
        "has_mri": int(has_mri),
        "has_subtype": int(has_subtype),
        "subtype_dist": subtype_dist,
        "ki67_usable_N": ki67_usable,
    }


def render(r: dict) -> str:
    lines = [
        "=== G0 Ki-67 + cohort audit ===",
        f"total rows                 : {r['total_rows']}",
        f"Ki-67 columns found        : {r['ki67_cols'] or 'NONE (expected — not enumerated)'}",
        f"subtype columns found      : {r['subtype_cols'] or 'NONE'}",
        "N-WATERFALL:",
        f"  has DCE-MRI (proxy)      : {r['has_mri']}",
        f"  has subtype label        : {r['has_subtype']}",
        f"  subtype value dist       : {r['subtype_dist'] or 'NONE'}",
        f"  USABLE NUMERIC Ki-67 N   : {r['ki67_usable_N']}   <-- the number that gates Head 2",
        "Duke n MAMA-MIA overlap    : TODO (de-duplicate before any 'external' claim — LOCK-2)",
        "",
        "Decision rule: small N -> demote Ki-67 to sub-analysis, keep subtype primary (decisions.md O-2).",
    ]
    return "\n".join(lines)


def selfcheck() -> int:
    df = pd.DataFrame(
        {
            "Patient ID": [1, 2, 3, 4, 5],
            "MRI Series": ["a", "b", None, "d", "e"],
            "Molecular Subtype": ["LumA", "TNBC", "LumA", None, "TNBC"],
            "Ki-67 (%)": ["20%", "n/a", "55", "", "5.5"],  
        }
    )
    r = audit(df)
    assert r["ki67_usable_N"] == 3, r
    assert r["has_subtype"] == 4, r
    assert r["ki67_cols"] == ["Ki-67 (%)"], r
    assert r["subtype_dist"] == {"LumA": 2, "TNBC": 2}, r
    assert numeric_ki67(pd.Series(["150", "-3", "30"])).tolist() == [30.0], "range filter"
    grouped = pd.DataFrame(
        [
            ["Group A", None],
            ["Patient ID", "Ki-67 (%)"],
            [None, "0=lo,1=hi"],
            ["p1", "30"],
            ["p2", "x"],
        ]
    )
    rg = audit(_resolve_header(grouped))
    assert rg["total_rows"] == 2 and rg["ki67_usable_N"] == 1, rg
    print("selfcheck OK — counting + 3-row-header logic sound")
    return 0


def _resolve_header(raw: pd.DataFrame) -> pd.DataFrame:
    if len(raw) > 3 and raw.iloc[0].isna().mean() > 0.3 and raw.iloc[1].notna().mean() > 0.5:
        df = raw.iloc[3:].copy()
        df.columns = [str(c) for c in raw.iloc[1]]
    else:
        df = raw.iloc[1:].copy()
        df.columns = [str(c) for c in raw.iloc[0]]
    return df.reset_index(drop=True)


def load(path: Path) -> pd.DataFrame:
    raw = (
        pd.read_excel(path, header=None)
        if path.suffix.lower() in {".xlsx", ".xls"}
        else pd.read_csv(path, header=None)
    )
    return _resolve_header(raw)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--table", type=Path, help="Duke Clinical_and_Other_Features table (.xlsx/.csv)"
    )
    ap.add_argument("--selfcheck", action="store_true", help="run synthetic self-check, no data")
    args = ap.parse_args(argv)

    if args.selfcheck:
        return selfcheck()
    if not args.table:
        print(
            "No --table given. Run with --selfcheck, or pass the Duke table path.", file=sys.stderr
        )
        return 2
    if not args.table.exists():
        print(
            f"Table not found: {args.table} (data is git-ignored / not yet downloaded).",
            file=sys.stderr,
        )
        return 2

    print(render(audit(load(args.table))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
