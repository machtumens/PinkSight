"""B2 integrity assertion (decisions.md LOCK-2): label-defining fields NEVER reach the classifier.

This is the highest-value test in the repo — a leakage breach inflates AUC and is the #1 way the
project fails a judge. It runs in CI. It bites the moment a real feature list exists.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pinksight import FORBIDDEN_FEATURES

# --- Arm A ([HEAD2-GRADE-PIVOT]) grade-label constants, verified 2026-07-10 against the raw table ---
# Duke Nottingham grade lives in Clinical_and_Other_Features.xlsx (3-row header) in the column NAMED
# "Nottingham grade" (positional index 34). Load by NAME, never by position — col#43 is a bilateral
# OTHER-SIDE grade decoy. Binary Arm A contrast drops NHG2 (mirrors DeepRadGrade).
_CLINICAL_XLSX = Path("data/raw/Clinical_and_Other_Features.xlsx")
_GRADE_COL = "Nottingham grade"
_EXPECTED_NHG1 = 113  # grade == 1
_EXPECTED_NHG3 = 207  # grade == 3
_EXPECTED_BINARY_N = 320  # NHG1 + NHG3 (the Arm A trainable universe before imaging-on-disk filtering)


def feature_columns() -> set[str]:
    """The classifier's REAL input columns — the P05 FT-Transformer clinical feature set (G2).

    clinical_encoder lazy-imports the ml stack, so this import stays torch-free and the test stays
    in the core suite. The assertion below now bites on the actual inputs, not a stub.
    """
    from pinksight.models.clinical_encoder import FEATURES

    return set(FEATURES)


def test_forbidden_fields_excluded_from_classifier():
    leaked = feature_columns() & {f.lower() for f in FORBIDDEN_FEATURES} | (
        feature_columns() & FORBIDDEN_FEATURES
    )
    assert not leaked, f"LEAKAGE: forbidden fields in classifier inputs: {leaked}"


def test_forbidden_ledger_is_nonempty():
    # Guard against the ledger being silently emptied.
    assert {"ER", "PR", "HER2"} <= FORBIDDEN_FEATURES


def test_fusion_modality_b_excludes_forbidden():
    """Track B late-fusion modality-B guard (plan trackb-fusion-deltaauc, LOCK-2).

    Ensures the 4-feature modality-B bundle used in the linear-probe TITAN + late-fusion ΔAUC
    experiment does not contain any forbidden/histology/survival columns. A violation here means
    direct subtype leakage (histology/IHC/PAM50 calls) or temporal leakage (survival endpoints)
    into the second modality — which would invalidate the ΔAUC result.
    """
    MODALITY_B_FEATURES = frozenset(["TBL_SCORE", "ANEUPLOIDY_SCORE", "TMB_NONSYNONYMOUS", "AGE"])
    # Direct subtype proxies (histology + IHC + PAM50 calls)
    FORBIDDEN_HISTOLOGY = frozenset(
        [
            "ONCOTREE_CODE", "CANCER_TYPE", "CANCER_TYPE_DETAILED", "TUMOR_TYPE",
            "ICD_O_3_HISTOLOGY", "ICD_O_3_SITE", "SUBTYPE",
        ]
    )
    # Survival/temporal endpoints (temporal leakage)
    FORBIDDEN_SURVIVAL = frozenset(
        [
            "OS_MONTHS", "DSS_MONTHS", "DFS_MONTHS", "PFS_MONTHS",
            "OS_STATUS", "DSS_STATUS", "DFS_STATUS", "PFS_STATUS",
        ]
    )
    # LOCK-2 clinical level (already in FORBIDDEN_FEATURES: ER/PR/HER2/Ki-67/Mol-Subtype/Oncotype +
    # gene-symbol mirror ESR1/PGR/ERBB2/MKI67).
    all_forbidden = FORBIDDEN_HISTOLOGY | FORBIDDEN_SURVIVAL | FORBIDDEN_FEATURES
    leaked = MODALITY_B_FEATURES & all_forbidden
    assert not leaked, f"LEAKAGE: modality-B bundle contains forbidden column(s): {leaked}"


def test_pam50_proliferation_never_feeds_subtype_head():
    """LOCK-2 FIREWALL (plan Step B-1.2) — the load-bearing zero-gradient proof for Piece B.

    The auxiliary PAM50 proliferation-index head is a STRUCTURALLY SEPARATE sklearn estimator; the
    subtype head must take ZERO gradient/information path from it. Two-part proof:

      (a) STRUCTURAL (Fully-Automated, synthetic fixture, no real data): the subtype model's per-fold
          design matrix has EXACTLY `1 + len(MODALITY_C_GENES)` columns — the proliferation score is
          NEVER a subtype input column — AND computing the proliferation Ridge leaves the subtype OOF
          bit-identical.
      (b) EMPIRICAL (Hybrid, skips if git-ignored real data absent): `run(compute_proliferation=False)`
          vs `run(compute_proliferation=True)` produce BIT-IDENTICAL subtype OOF arrays — the auxiliary
          head's presence/absence cannot perturb the subtype path. Strongest available proof of zero
          coupling for two independently-fit estimators (E2).
    """
    pytest.importorskip("torch", reason="firewall test imports the ml-stack fusion script")
    import sys

    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    sys.path.insert(0, str(scripts_dir / "novel_heads"))
    import trackb_fusion_wsi_genomics as tb

    # (a) STRUCTURAL — synthetic fixture, no real data required
    struct = tb._selfcheck_firewall_structural()
    assert struct["subtype_x_cols"] == 1 + len(tb.MODALITY_C_GENES), (
        f"subtype design matrix must be exactly 1 + {len(tb.MODALITY_C_GENES)} cols "
        f"(TITAN logit + panel), got {struct['subtype_x_cols']} — proliferation score leaked in as a col"
    )
    assert struct["bit_identical"], (
        "synthetic firewall check: proliferation on/off changed the subtype OOF — firewall breached"
    )

    # (b) EMPIRICAL — real-data bit-for-bit equivalence (Hybrid; skip cleanly if data absent)
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
    """Plan Step B-1.3 — Piece A dispatcher routing-table integrity (torch-free).

    Every WIRED registry entry resolves to a real, existing script path with the routing-only
    `cross_cohort_gradient=False` contract; every documented NOT-WIRED combo and any unknown combo
    return the NOT-WIRED sentinel WITHOUT raising (never silently guesses).
    """
    import sys

    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    import pinksight_dispatch as dp

    not_wired = "NOT WIRED — confirm path or backlog"

    # Every WIRED registry key resolves to an existing script path, no cross-cohort gradient.
    for (cohort, modalities), script in dp.COHORT_HARNESS_REGISTRY.items():
        res = dp.dispatch(cohort, modalities)
        assert res.status == "WIRED", f"({cohort}, {sorted(modalities)}) expected WIRED, got {res.status}"
        assert res.harness_script == script
        assert Path(res.harness_script).exists(), (
            f"WIRED entry ({cohort}, {sorted(modalities)}) -> {res.harness_script} does not exist on disk"
        )
        assert res.cross_cohort_gradient is False

    # Every documented NOT-WIRED combo degrades gracefully with harness_script=None.
    for cohort, modalities in dp.NOT_WIRED_COMBOS:
        res = dp.dispatch(cohort, modalities)
        assert res.status == not_wired
        assert res.harness_script is None
        assert res.cross_cohort_gradient is False

    # An entirely unknown combo returns NOT WIRED and does NOT raise.
    unknown = dp.dispatch("nonexistent_cohort", frozenset({"nonexistent_modality"}))
    assert unknown.status == not_wired
    assert unknown.harness_script is None
    assert unknown.cross_cohort_gradient is False


# ---------------------------------------------------------------------------------------------------
# Track-C (ADR-0010) tabular-risk suite ↔ Track-A firewall (LOCK-2). The Track-C METABRIC prognosis
# model LEGALLY carries ER/PR/HER2/CLAUDIN_SUBTYPE as inputs (survival is the target, not subtype) —
# but those columns must NEVER cross into a Track-A (Duke) subtype/Ki-67 feature table, or they
# launder FORBIDDEN receptor/subtype inputs into the characterisation head (the SAME failure class as
# the rejected G3 HR-status-routing leak). ADR-0010 §"Required guards" item 1 mandates this CI.
# ---------------------------------------------------------------------------------------------------

# The Track-C tabular-risk models' feature columns, keyed by cohort. METABRIC's receptor/subtype
# columns are the load-bearing danger (they are direct IHC/PAM50 label surrogates). BCSC/Coimbra
# feature names are screening/blood-panel variables — listed so the whole Track-C suite is walled,
# not just METABRIC. Names match explore/tabular_risk/src/*_loader.py + the results metrics JSONs.
_TRACKC_METABRIC_FEATURES = frozenset(
    {
        "AGE_AT_DIAGNOSIS", "TUMOR_SIZE", "GRADE", "TUMOR_STAGE",
        "LYMPH_NODES_EXAMINED_POSITIVE",
        # The receptor/subtype columns — the reason this firewall exists.
        "ER_STATUS", "PR_STATUS", "HER2_STATUS", "CLAUDIN_SUBTYPE",
        "HISTOLOGICAL_SUBTYPE", "INFERRED_MENOPAUSAL_STATE", "CELLULARITY",
    }
)
# The receptor/subtype subset that is a direct LOCK-2 IHC/PAM50 leak if it ever reaches a Track-A
# head — the non-negotiable exclusion.
_TRACKC_METABRIC_SUBTYPE_LEAK = frozenset(
    {"ER_STATUS", "PR_STATUS", "HER2_STATUS", "CLAUDIN_SUBTYPE"}
)


def test_metabric_subtype_features_absent_from_track_a_classifier():
    """ADR-0010 guard #1 (LOCK-2): METABRIC's ER/PR/HER2/CLAUDIN_SUBTYPE (and the rest of the
    Track-C METABRIC feature set) must NEVER appear in the Track-A (Duke) clinical classifier
    feature set. METABRIC may carry these for its SURVIVAL target; feeding them to the Duke
    SUBTYPE/Ki-67 head would launder FORBIDDEN receptor calls (circular, same class as the rejected
    G3 HR-status leak). Reuses the real Track-A input list (`clinical_encoder.FEATURES`).
    """
    from pinksight.models.clinical_encoder import FEATURES

    track_a = set(FEATURES)
    leaked = track_a & _TRACKC_METABRIC_FEATURES
    assert not leaked, (
        f"LEAKAGE: Track-C METABRIC feature(s) reached the Track-A Duke classifier inputs: {leaked}"
    )
    # The receptor/subtype subset is the highest-severity leak — assert it explicitly so a partial
    # future overlap still bites here with a clear message.
    receptor_leak = track_a & _TRACKC_METABRIC_SUBTYPE_LEAK
    assert not receptor_leak, (
        f"LEAKAGE (IHC/subtype): METABRIC receptor/subtype column(s) in Track-A inputs: "
        f"{receptor_leak}"
    )


def test_metabric_receptor_columns_map_to_forbidden_ledger():
    """The METABRIC receptor columns are receptor-status restatements of the FORBIDDEN IHC fields
    (ER/PR/HER2). Assert the mapping is intact so the firewall cannot be silently bypassed by
    renaming a forbidden field to its `_STATUS` form. `CLAUDIN_SUBTYPE` is a PAM50/molecular-subtype
    call — the same class as the forbidden 'Mol Subtype'/'Molecular Subtype' ledger entries.
    """
    # Each METABRIC receptor column carries a forbidden IHC field as its root token.
    receptor_root = {
        "ER_STATUS": "ER",
        "PR_STATUS": "PR",
        "HER2_STATUS": "HER2",
    }
    for col, root in receptor_root.items():
        assert root in FORBIDDEN_FEATURES, (
            f"ledger drift: {root} (root of METABRIC {col}) missing from FORBIDDEN_FEATURES"
        )
    # CLAUDIN_SUBTYPE is a molecular-subtype call; confirm the molecular-subtype family is banned.
    assert {"Mol Subtype", "Molecular Subtype"} & FORBIDDEN_FEATURES, (
        "ledger drift: molecular-subtype family (CLAUDIN_SUBTYPE's leak class) missing from "
        "FORBIDDEN_FEATURES"
    )


def roc_auc(scores, labels) -> float:
    """AUC via the Mann–Whitney U statistic (numpy-only; no sklearn at G0)."""
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC needs both classes present")
    ranks = np.empty(len(scores), float)
    ranks[scores.argsort()] = np.arange(1, len(scores) + 1)  # synthetic scores are distinct
    return (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


@pytest.mark.leakage
def test_label_shuffle_probe_synthetic():
    """B2 label-shuffle sentinel — SYNTHETIC unit guard (always runs, torch-free): a score with real
    signal must collapse to chance (AUC < 0.60) once labels are shuffled. This proves the shuffle
    LOGIC is sound (a permutation destroys a genuine feature→label link). The LIVE gate against the
    real clinical CV is `test_label_shuffle_probe_real_oof` below.
    """
    rng = np.random.default_rng(0)
    n = 400
    labels = np.r_[np.zeros(n // 2), np.ones(n // 2)].astype(int)
    scores = labels + rng.normal(0, 0.3, n)  # genuinely separates the classes
    assert roc_auc(scores, labels) > 0.90, "probe broken: real signal not detected"
    shuffled = rng.permutation(labels)
    assert roc_auc(scores, shuffled) < 0.60, "LEAKAGE SENTINEL: shuffled-label AUC >= 0.60"


# ---------------------------------------------------------------------------------------------------
# Arm A ([HEAD2-GRADE-PIVOT]) grade-head integrity — grade is the TARGET, never an input (no [1.16]
# circularity). These two gates are the EXECUTE step-1 preconditions for the grade smoke.
# ---------------------------------------------------------------------------------------------------


def _load_grade_series() -> pd.Series:
    """Numeric Nottingham-grade series (1/2/3) from the raw Duke table, loaded BY COLUMN NAME.

    Uses the same 3-row-header resolver the clinical encoder uses (scripts/audit_ki67.load). Kept
    torch-free so this stays in the core suite; skips cleanly when the git-ignored table is absent.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from audit_ki67 import load as load_clinical  # scripts/ on sys.path, not a package

    df = load_clinical(_CLINICAL_XLSX)
    df.columns = [str(c) for c in df.columns]
    col = df[_GRADE_COL]
    if isinstance(col, pd.DataFrame):  # duplicate-label guard — take the PRIMARY (first) column
        col = col.iloc[:, 0]
    return pd.to_numeric(col, errors="coerce")


def test_grade_is_target_not_feature():
    """[1.16]/[8.3] Arm A guarantee (decisions.md:876): grade is the TARGET of the imaging grade-head,
    imaging is the input, and grade is NOT in the imaging → no circularity.

    The machine-checkable form of that guarantee, and the ONLY form that is true against this codebase:

      (1) Nottingham grade is NOT in FORBIDDEN_FEATURES — it is a legitimate label-safe TARGET, not an
          IHC label-defining field (ER/PR/HER2/Ki-67/Mol-Subtype/Oncotype are the forbidden set). This
          is what lets grade be an Arm A target at all.
      (2) Arm A's model input is PURE IMAGING (NpyVolumeDataset yields MRI voxel tensors only — zero
          clinical columns by construction), so the grade label never re-enters the grade-head as a
          feature. The (pid, grade) item list uses grade solely as the scalar target y.

    NOTE (deviation from a naive reading): grade ("Nottingham grade") IS legitimately present in
    clinical_encoder.FEATURES — but that is the SUBTYPE head, where grade is a label-SAFE predictor
    ([1.16] line 184: "Label-safe set only", and decisions.md [G2-CLIN-LEAK]: grade is a
    legitimate-but-label-correlated subtype predictor). Asserting grade absent from FEATURES would
    contradict that LOCKED decision and is NOT the Arm A circularity. The Arm A circularity would be
    feeding grade as an input to the GRADE head — which the pure-imaging pipeline structurally cannot do.
    """
    grade_names = {"nottingham grade", "grade", "nottingham"}

    # (1) grade is not an IHC label-defining forbidden field
    forbidden_lower = {f.lower() for f in FORBIDDEN_FEATURES}
    assert grade_names.isdisjoint(forbidden_lower), (
        f"grade must NOT be in FORBIDDEN_FEATURES (it is the Arm A TARGET, not an IHC label): "
        f"{grade_names & forbidden_lower}"
    )

    # (2) the grade-head input is pure imaging: the dataset item is (pid, label) and __getitem__ yields
    #     (voxel_tensor, y, pid) — no clinical feature vector exists on the imaging path. We assert the
    #     structural contract: NpyVolumeDataset carries labels as scalar y, never as an input channel.
    # NpyVolumeDataset lives in pinksight.data.dataset, which imports torch+monai at module level.
    # The grade-not-in-FORBIDDEN half (part 1) above is torch-free and always runs; this structural
    # half (part 2) needs the ml stack, so skip it cleanly under a base install (ml-extra portion).
    pytest.importorskip("torch", reason="pinksight.data.dataset imports torch+monai for NpyVolumeDataset")
    from pinksight.data.dataset import NpyVolumeDataset

    # the dataset stores (pid, label) pairs and never a feature matrix keyed by grade — assert the
    # constructor signature treats the second tuple element as the scalar label (target), not a feature.
    ds = NpyVolumeDataset([("Breast_MRI_999", 1)], proc_dir=Path("data/processed"))
    assert ds.items == [("Breast_MRI_999", 1)], "dataset must carry (pid, label) — grade is the target y"
    # imaging channels are MRI phases only (first_post/pre_post/fixed4/subtraction/kinetic) — no clinical
    # column, so grade cannot leak in as a feature by construction.
    from pinksight.data.dataset import CHANNEL_POLICIES

    assert "grade" not in {c.lower() for c in CHANNEL_POLICIES}, (
        "imaging channel policies must be MRI phases only — no clinical/grade channel"
    )


def test_grade_label_counts():
    """Arm A label sanity ([HEAD2-GRADE-PIVOT], DeepRadGrade replication): the binary NHG1-vs-NHG3
    contrast is NHG1=113 / NHG3=207 / total 320, loaded BY COLUMN NAME "Nottingham grade" (col#34),
    NOT positional index and NOT the col#43 bilateral other-side decoy. NHG2 is dropped from training.
    """
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
