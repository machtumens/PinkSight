
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import audit_ki67


def test_selfcheck_passes():
    assert audit_ki67.selfcheck() == 0


def test_numeric_ki67_range_filter():
    import pandas as pd

    out = audit_ki67.numeric_ki67(pd.Series(["20%", "150", "-1", "0", "100"]))
    assert sorted(out.tolist()) == [0.0, 20.0, 100.0]


def test_no_table_returns_error_code():
    assert audit_ki67.main(["--table", "data/does_not_exist.xlsx"]) == 2
