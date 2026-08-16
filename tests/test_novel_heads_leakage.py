from __future__ import annotations

import re

import pandas as pd
import pytest

from pinksight import FORBIDDEN_FEATURES

COVERED_ARMS = (1, 2, 4, 6, 8, 9, 10)


def _canonical(token: str) -> str:
    return re.sub(r"[^a-z0-9]", "", token.lower())


_LEDGER_TOKENS = {_canonical(f) for f in FORBIDDEN_FEATURES}
_ALIAS_TOKENS = {
    _canonical(a)
    for a in (
        "ER_STATUS",
        "PR_STATUS",
        "HER2_STATUS",
        "HER-2",
        "HER2_IHC",
        "ER_IHC",
        "PR_IHC",
        "estrogen_receptor",
        "progesterone_receptor",
        "Ki-67",
        "Ki_67",
        "MIB1",
        "MIB-1",
        "ki67_index",
        "proliferation_index_ki67",
        "Mol_Subtype",
        "Molecular_Subtype",
        "PAM50",
        "PAM50_Subtype",
        "CLAUDIN_SUBTYPE",
        "intrinsic_subtype",
        "subtype_pam50",
        "Oncotype",
        "Oncotype_DX",
        "OncotypeDX",
        "RS_SCORE",
        "recurrence_score",
    )
}
_FORBIDDEN_TOKENS = _LEDGER_TOKENS | _ALIAS_TOKENS


class LeakageError(AssertionError):
    pass


def assert_no_forbidden_inputs(columns) -> None:
    leaked = sorted({str(c) for c in columns if _canonical(str(c)) in _FORBIDDEN_TOKENS})
    if leaked:
        raise LeakageError(
            f"LEAKAGE: forbidden classifier INPUT column(s) present (LOCK-2): {leaked}. "
            f"ER/PR/HER2/Ki-67/Mol-Subtype/Oncotype (and aliases/gene mirrors) are label surrogates "
            f"— exclude from inputs; they may only be the prediction TARGET."
        )


def _clean_wave1_expression() -> pd.DataFrame:
    return pd.DataFrame(
        {"TP53": [0.1, -0.3], "CDH1": [1.2, 0.4], "PIK3CA": [-0.5, 0.9], "GATA3": [0.2, 0.1]}
    )


def _dirty_wave1_expression() -> pd.DataFrame:
    df = _clean_wave1_expression()
    df["MKI67"] = [3.1, 2.8]
    return df


def _clean_wave2_radiomics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "original_firstorder_Mean": [10.0, 12.0],
            "original_glcm_Contrast": [0.3, 0.5],
            "original_shape_Sphericity": [0.8, 0.7],
        }
    )


def _dirty_wave2_radiomics() -> pd.DataFrame:
    df = _clean_wave2_radiomics()
    df["ER_STATUS"] = [1, 0]
    return df


@pytest.mark.leakage
def test_clean_wave1_matrix_passes():
    assert_no_forbidden_inputs(_clean_wave1_expression().columns)


@pytest.mark.leakage
def test_clean_wave2_matrix_passes():
    assert_no_forbidden_inputs(_clean_wave2_radiomics().columns)


@pytest.mark.leakage
def test_dirty_wave1_matrix_raises():
    with pytest.raises(LeakageError, match="MKI67"):
        assert_no_forbidden_inputs(_dirty_wave1_expression().columns)


@pytest.mark.leakage
def test_dirty_wave2_matrix_raises():
    with pytest.raises(LeakageError, match="ER_STATUS"):
        assert_no_forbidden_inputs(_dirty_wave2_radiomics().columns)


@pytest.mark.leakage
@pytest.mark.parametrize(
    "forbidden_col",
    [
        "ER",
        "PR",
        "HER2",
        "Ki-67",
        "Ki67",
        "Mol Subtype",
        "Oncotype",
        "PAM50",
        "ESR1",
        "ERBB2",
        "HER-2",
        "PR_STATUS",
        "CLAUDIN_SUBTYPE",
        "OncotypeDX",
    ],
)
def test_each_forbidden_field_and_alias_is_caught(forbidden_col):
    df = _clean_wave1_expression()
    df[forbidden_col] = [1, 0]
    with pytest.raises(LeakageError):
        assert_no_forbidden_inputs(df.columns)


@pytest.mark.leakage
def test_forbidden_ledger_is_nonempty():
    assert {"ER", "PR", "HER2"} <= FORBIDDEN_FEATURES
    assert {"ESR1", "ERBB2", "MKI67"} <= FORBIDDEN_FEATURES


@pytest.mark.leakage
def test_dual_wave_scope_documented():
    assert set(COVERED_ARMS) == {1, 2, 4, 6, 8, 9, 10}, (
        "AC-1 covers BOTH waves — Wave 1 arms 2,4,6,8,10 AND Wave 2 arms 1,9"
    )
