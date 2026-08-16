
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pinksight import FORBIDDEN_FEATURES

_CLINICAL_XLSX = Path("data/raw/Clinical_and_Other_Features.xlsx")
_GRADE_COL = "Nottingham grade"
_EXPECTED_NHG1 = 113  
_EXPECTED_NHG3 = 207  
_EXPECTED_BINARY_N = 320  


def feature_columns() -> set[str]:
    from pinksight.models.clinical_encoder import FEATURES

    return set(FEATURES)


def test_forbidden_fields_excluded_from_classifier():
    leaked = feature_columns() & {f.lower() for f in FORBIDDEN_FEATURES} | (
        feature_columns() & FORBIDDEN_FEATURES
    )
    assert not leaked, f"LEAKAGE: forbidden fields in classifier inputs: {leaked}"


def test_forbidden_ledger_is_nonempty():
    assert {"ER", "PR", "HER2"} <= FORBIDDEN_FEATURES


def test_fusion_modality_b_excludes_forbidden():
    MODALITY_B_FEATURES = frozenset(["TBL_SCORE", "ANEUPLOIDY_SCORE", "TMB_NONSYNONYMOUS", "AGE"])
    FORBIDDEN_HISTOLOGY = frozenset(
        [
            "ONCOTREE_CODE", "CANCER_TYPE", "CANCER_TYPE_DETAILED", "TUMOR_TYPE",
            "ICD_O_3_HISTOLOGY", "ICD_O_3_SITE", "SUBTYPE",
        ]
    )
    FORBIDDEN_SURVIVAL = frozenset(
        [
            "OS_MONTHS", "DSS_MONTHS", "DFS_MONTHS", "PFS_MONTHS",
            "OS_STATUS", "DSS_STATUS", "DFS_STATUS", "PFS_STATUS",
        ]
    )
    all_forbidden = FORBIDDEN_HISTOLOGY | FORBIDDEN_SURVIVAL | FORBIDDEN_FEATURES
    leaked = MODALITY_B_FEATURES & all_forbidden
    assert not leaked, f"LEAKAGE: modality-B bundle contains forbidden column(s): {leaked}"


def test_pam50_proliferation_never_feeds_subtype_head():
    pytest.importorskip("torch", reason="firewall test imports the ml-stack fusion script")
    import sys

    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    sys.path.insert(0, str(scripts_dir / "novel_heads"))
    import trackb_fusion_wsi_genomics as tb

    struct = tb._selfcheck_firewall_structural()
    assert struct["subtype_x_cols"] == 1 + len(tb.MODALITY_C_GENES), (
        f"subtype design matrix must be exactly 1 + {len(tb.MODALITY_C_GENES)} cols "
        f"(TITAN logit + panel), got {struct['subtype_x_cols']} — proliferation score leaked in as a col"
    )
    assert struct["bit_identical"], (
        "synthetic firewall check: proliferation on/off changed the subtype OOF — firewall breached"
    )

    manifest = Path("data/pathology/features/tcga_brca_titan_manifest.csv")
    expr = Path("data/genomics/tcga/brca_tcga_pan_can_atlas_2018/data_mrna_seq_v2_rsem.txt")
    if not (manifest.exists() and expr.exists()):
        pytest.skip("TITAN manifest / mRNA matrix absent (git-ignored) — real firewall equivalence skipped")

    import numpy as np

    r_off = tb.run(compute_proliferation=False)
    r_on = tb.run(compute_proliferation=True)
    off = np.asarray(r_off["oof_scores_fusion"], dtype=float)
    on = np.asarray(r_on["oof_scores_fusion"], dtype=float)
    assert np.array_equal(off, on), (
        "FIREWALL BREACH: subtype OOF differs with vs without the proliferation head — the two "
        "estimators are NOT decoupled (LOCK-2 violation). Do NOT weaken this test; fix the code (E2)."
    )
    assert "proliferation" not in r_off, "compute_proliferation=False must NOT emit a proliferation sub-dict"
    assert "proliferation" in r_on, "compute_proliferation=True must emit the proliferation sub-dict"


def test_dispatcher_registry_resolves_known_combos():
    import sys

    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    import pinksight_dispatch as dp

    not_wired = "NOT WIRED — confirm path or backlog"

    for (cohort, modalities), script in dp.COHORT_HARNESS_REGISTRY.items():
        res = dp.dispatch(cohort, modalities)
        assert res.status == "WIRED", f"({cohort}, {sorted(modalities)}) expected WIRED, got {res.status}"
        assert res.harness_script == script
        assert Path(res.harness_script).exists(), (
            f"WIRED entry ({cohort}, {sorted(modalities)}) -> {res.harness_script} does not exist on disk"
        )
        assert res.cross_cohort_gradient is False

    for cohort, modalities in dp.NOT_WIRED_COMBOS:
        res = dp.dispatch(cohort, modalities)
        assert res.status == not_wired
        assert res.harness_script is None
        assert res.cross_cohort_gradient is False

    unknown = dp.dispatch("nonexistent_cohort", frozenset({"nonexistent_modality"}))
    assert unknown.status == not_wired
    assert unknown.harness_script is None
    assert unknown.cross_cohort_gradient is False


_TRACKC_METABRIC_FEATURES = frozenset(
    {
        "AGE_AT_DIAGNOSIS", "TUMOR_SIZE", "GRADE", "TUMOR_STAGE",
        "LYMPH_NODES_EXAMINED_POSITIVE",
        "ER_STATUS", "PR_STATUS", "HER2_STATUS", "CLAUDIN_SUBTYPE",
        "HISTOLOGICAL_SUBTYPE", "INFERRED_MENOPAUSAL_STATE", "CELLULARITY",
    }
)
_TRACKC_METABRIC_SUBTYPE_LEAK = frozenset(
    {"ER_STATUS", "PR_STATUS", "HER2_STATUS", "CLAUDIN_SUBTYPE"}
)


def test_metabric_subtype_features_absent_from_track_a_classifier():
    from pinksight.models.clinical_encoder import FEATURES

    track_a = set(FEATURES)
    leaked = track_a & _TRACKC_METABRIC_FEATURES
    assert not leaked, (
        f"LEAKAGE: Track-C METABRIC feature(s) reached the Track-A Duke classifier inputs: {leaked}"
    )
    receptor_leak = track_a & _TRACKC_METABRIC_SUBTYPE_LEAK
    assert not receptor_leak, (
        f"LEAKAGE (IHC/subtype): METABRIC receptor/subtype column(s) in Track-A inputs: "
        f"{receptor_leak}"
    )


def test_metabric_receptor_columns_map_to_forbidden_ledger():
    receptor_root = {
        "ER_STATUS": "ER",
        "PR_STATUS": "PR",
        "HER2_STATUS": "HER2",
    }
    for col, root in receptor_root.items():
        assert root in FORBIDDEN_FEATURES, (
            f"ledger drift: {root} (root of METABRIC {col}) missing from FORBIDDEN_FEATURES"
        )
    assert {"Mol Subtype", "Molecular Subtype"} & FORBIDDEN_FEATURES, (
        "ledger drift: molecular-subtype family (CLAUDIN_SUBTYPE's leak class) missing from "
        "FORBIDDEN_FEATURES"
    )


def roc_auc(scores, labels) -> float:
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC needs both classes present")
    ranks = np.empty(len(scores), float)
    ranks[scores.argsort()] = np.arange(1, len(scores) + 1)  
    return (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


@pytest.mark.leakage
def test_label_shuffle_probe_synthetic():
    rng = np.random.default_rng(0)
    n = 400
    labels = np.r_[np.zeros(n // 2), np.ones(n // 2)].astype(int)
    scores = labels + rng.normal(0, 0.3, n)  
    assert roc_auc(scores, labels) > 0.90, "probe broken: real signal not detected"
    shuffled = rng.permutation(labels)
    assert roc_auc(scores, shuffled) < 0.60, "LEAKAGE SENTINEL: shuffled-label AUC >= 0.60"


def _load_grade_series() -> pd.Series:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from audit_ki67 import load as load_clinical  

    df = load_clinical(_CLINICAL_XLSX)
    df.columns = [str(c) for c in df.columns]
    col = df[_GRADE_COL]
    if isinstance(col, pd.DataFrame):  
        col = col.iloc[:, 0]
    return pd.to_numeric(col, errors="coerce")


def test_grade_is_target_not_feature():
    grade_names = {"nottingham grade", "grade", "nottingham"}

    forbidden_lower = {f.lower() for f in FORBIDDEN_FEATURES}
    assert grade_names.isdisjoint(forbidden_lower), (
        f"grade must NOT be in FORBIDDEN_FEATURES (it is the Arm A TARGET, not an IHC label): "
        f"{grade_names & forbidden_lower}"
    )

    pytest.importorskip("torch", reason="pinksight.data.dataset imports torch+monai for NpyVolumeDataset")
    from pinksight.data.dataset import NpyVolumeDataset

    ds = NpyVolumeDataset([("Breast_MRI_999", 1)], proc_dir=Path("data/processed"))
    assert ds.items == [("Breast_MRI_999", 1)], "dataset must carry (pid, label) — grade is the target y"
    from pinksight.data.dataset import CHANNEL_POLICIES

    assert "grade" not in {c.lower() for c in CHANNEL_POLICIES}, (
        "imaging channel policies must be MRI phases only — no clinical/grade channel"
    )


def test_grade_label_counts():
    if not _CLINICAL_XLSX.exists():
        pytest.skip(f"{_CLINICAL_XLSX} not present (git-ignored) — grade-count gate cannot run")

    grade = _load_grade_series()
    n1 = int((grade == 1).sum())
    n2 = int((grade == 2).sum())
    n3 = int((grade == 3).sum())
    assert n1 == _EXPECTED_NHG1, f"NHG1 count drifted: got {n1}, expected {_EXPECTED_NHG1}"
    assert n3 == _EXPECTED_NHG3, f"NHG3 count drifted: got {n3}, expected {_EXPECTED_NHG3}"
    assert n1 + n3 == _EXPECTED_BINARY_N, (
        f"binary NHG1+NHG3 N drifted: got {n1 + n3}, expected {_EXPECTED_BINARY_N} "
        f"(NHG2 dropped = {n2})"
    )
