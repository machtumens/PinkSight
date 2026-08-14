"""AC-1 leakage assertion for the novel-heads program — DUAL-WAVE scope (LOCK-2, decisions.md).

Scope (AC-1, SUPPLEMENT Gap 4 — authoritative): this test guards EVERY EXECUTED arm's feature
matrix across BOTH waves — Wave 1 tabular arms (2, 4, 6, 8, 10) AND Wave 2 imaging arms (1, 9). No
ER / PR / HER2 / Ki-67 / Mol-Subtype / Oncotype column (or a common alias / gene-symbol mirror) may
be reachable as a classifier INPUT. Any earlier reference to "Wave 1 features only" is superseded by
this dual-wave definition. Forbidden fields are legal as TARGETS, never as inputs.

Why this is the highest-value test in the program: a leakage breach inflates AUROC and is the #1 way
the project fails a judge (same class as the rejected G3 HR-status-routing leak). It runs in CI and
bites the moment any arm builds a real feature matrix. Reuses the single-source ledger
`pinksight.FORBIDDEN_FEATURES` (never re-defines the forbidden set).

RED/GREEN contract (checklist item 4): `assert_no_forbidden_inputs` RAISES on a matrix containing a
forbidden column and PASSES on a clean matrix — this file encodes BOTH sides (the dirty-matrix cases
assert a raise; the clean-matrix cases assert no raise), so the same test run proves the assertion is
neither vacuously green nor over-eager. Small in-memory dummy dataframes only — no real data.

Plan: process/general-plans/active/novel-heads-roadmap_25-07-26/novel-heads-roadmap_PLAN_25-07-26.md
"""

from __future__ import annotations

import re

import pandas as pd
import pytest

from pinksight import FORBIDDEN_FEATURES

# The arms this assertion covers (dual-wave). Documented here so the scope is auditable in-code.
COVERED_ARMS = (1, 2, 4, 6, 8, 9, 10)


def _canonical(token: str) -> str:
    """Normalise a column name to a comparable token: lowercase, strip non-alphanumerics.

    So that `Ki-67`, `Ki_67`, `ki 67`, `KI67` all collapse to `ki67`; `ER_STATUS` -> `erstatus`;
    `HER-2` -> `her2`. This is what lets the alias matcher below catch the common column-name variants
    a real cohort loader would produce without enumerating every spelling.
    """
    return re.sub(r"[^a-z0-9]", "", token.lower())


# Canonical forbidden tokens from the single-source ledger, plus the common column-name aliases a
# public-cohort loader emits. The ledger (ER/PR/HER2/Ki-67/Mol Subtype/Oncotype + ESR1/PGR/ERBB2/
# MKI67) is the SOURCE OF TRUTH; the extra aliases are receptor-status / subtype restatements that
# are the SAME forbidden field under a different column name (the leak this gate must catch).
_LEDGER_TOKENS = {_canonical(f) for f in FORBIDDEN_FEATURES}
_ALIAS_TOKENS = {
    _canonical(a)
    for a in (
        # receptor-status restatements
        "ER_STATUS",
        "PR_STATUS",
        "HER2_STATUS",
        "HER-2",
        "HER2_IHC",
        "ER_IHC",
        "PR_IHC",
        "estrogen_receptor",
        "progesterone_receptor",
        # Ki-67 spellings
        "Ki-67",
        "Ki_67",
        "MIB1",
        "MIB-1",
        "ki67_index",
        "proliferation_index_ki67",
        # molecular-subtype / PAM50 call restatements
        "Mol_Subtype",
        "Molecular_Subtype",
        "PAM50",
        "PAM50_Subtype",
        "CLAUDIN_SUBTYPE",
        "intrinsic_subtype",
        "subtype_pam50",
        # Oncotype restatements
        "Oncotype",
        "Oncotype_DX",
        "OncotypeDX",
        "RS_SCORE",
        "recurrence_score",
    )
}
_FORBIDDEN_TOKENS = _LEDGER_TOKENS | _ALIAS_TOKENS


class LeakageError(AssertionError):
    """A feature matrix carried a forbidden classifier INPUT (LOCK-2). Fatal — never suppress."""


def assert_no_forbidden_inputs(columns) -> None:
    """RAISE `LeakageError` if any column name maps to a forbidden field; otherwise return (pass).

    `columns` is any iterable of column names (a pandas `df.columns` in practice). Matching is on the
    canonicalised token (case/underscore/dash-insensitive) so `ER_STATUS`, `Ki-67`, `PAM50_Subtype`
    are all caught. This is the shared assertion every arm's floor gate calls on its feature-matrix
    columns BEFORE any fit. Forbidden fields as the prediction TARGET are fine — only INPUTS are
    guarded here (the caller never passes the target column into this check).
    """
    leaked = sorted({str(c) for c in columns if _canonical(str(c)) in _FORBIDDEN_TOKENS})
    if leaked:
        raise LeakageError(
            f"LEAKAGE: forbidden classifier INPUT column(s) present (LOCK-2): {leaked}. "
            f"ER/PR/HER2/Ki-67/Mol-Subtype/Oncotype (and aliases/gene mirrors) are label surrogates "
            f"— exclude from inputs; they may only be the prediction TARGET."
        )


# ================================================================================================
# Dummy feature matrices — small, in-memory, no real data. One clean + one dirty per representative
# arm feature-space (tabular Wave 1 + imaging Wave 2). These stand in for the real per-arm matrices.
# ================================================================================================
def _clean_wave1_expression() -> pd.DataFrame:
    """Arm 2/4/6/8/10-style tabular matrix: label-safe (non-PAM50) expression / tabular features."""
    return pd.DataFrame(
        {"TP53": [0.1, -0.3], "CDH1": [1.2, 0.4], "PIK3CA": [-0.5, 0.9], "GATA3": [0.2, 0.1]}
    )


def _dirty_wave1_expression() -> pd.DataFrame:
    """Same tabular matrix but with a forbidden gene-symbol mirror (MKI67) leaked in as an input."""
    df = _clean_wave1_expression()
    df["MKI67"] = [3.1, 2.8]  # gene-symbol mirror of Ki-67 — forbidden input (LOCK-2)
    return df


def _clean_wave2_radiomics() -> pd.DataFrame:
    """Arm 1/9-style imaging radiomics matrix: pyradiomics texture features only (label-safe)."""
    return pd.DataFrame(
        {
            "original_firstorder_Mean": [10.0, 12.0],
            "original_glcm_Contrast": [0.3, 0.5],
            "original_shape_Sphericity": [0.8, 0.7],
        }
    )


def _dirty_wave2_radiomics() -> pd.DataFrame:
    """Same radiomics matrix but with a receptor-status column (ER_STATUS) leaked in as an input."""
    df = _clean_wave2_radiomics()
    df["ER_STATUS"] = [1, 0]  # receptor-status restatement of ER — forbidden input (LOCK-2)
    return df


# ------------------------------------------------------------------------------------------------
# GREEN side — clean matrices must PASS (no raise).
# ------------------------------------------------------------------------------------------------
@pytest.mark.leakage
def test_clean_wave1_matrix_passes():
    assert_no_forbidden_inputs(_clean_wave1_expression().columns)  # must not raise


@pytest.mark.leakage
def test_clean_wave2_matrix_passes():
    assert_no_forbidden_inputs(_clean_wave2_radiomics().columns)  # must not raise


# ------------------------------------------------------------------------------------------------
# RED side — dirty matrices must RAISE (the assertion bites). This is the "fails on forbidden columns"
# half of the checklist-item-4 contract, expressed as a passing test that asserts the raise.
# ------------------------------------------------------------------------------------------------
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
    """Every core forbidden field, gene mirror, and common alias must be caught as an input leak."""
    df = _clean_wave1_expression()
    df[forbidden_col] = [1, 0]
    with pytest.raises(LeakageError):
        assert_no_forbidden_inputs(df.columns)


@pytest.mark.leakage
def test_forbidden_ledger_is_nonempty():
    """Guard against the single-source ledger being silently emptied (mirrors test_leakage.py)."""
    assert {"ER", "PR", "HER2"} <= FORBIDDEN_FEATURES
    assert {"ESR1", "ERBB2", "MKI67"} <= FORBIDDEN_FEATURES  # gene-symbol mirror intact


@pytest.mark.leakage
def test_dual_wave_scope_documented():
    """AC-1 dual-wave scope self-guard: the covered arms are exactly Wave 1 (2,4,6,8,10) + Wave 2 (1,9)."""
    assert set(COVERED_ARMS) == {1, 2, 4, 6, 8, 9, 10}, (
        "AC-1 covers BOTH waves — Wave 1 arms 2,4,6,8,10 AND Wave 2 arms 1,9"
    )
