"""Tier-1 leakage-probe helpers (ADDITIVE — decisions.md LOCK-2, integrity commitments).

Three trustworthiness questions on the G2 clinical-only headline (Luminal-like vs TNBC), all
answered by feeding the EXISTING patient-grouped CV (`clinical_encoder.cross_val_auroc`) — never a
new model, never a touched split, never a relaxed leak guard:

  1. DeLong 95% CI on the headline AUROC          → reuse `pinksight.metrics.delong_ci` (already
                                                     self-checked: CI brackets AUC, matches Mann-Whitney).
  2. patient-level label-shuffle sentinel          → `shuffle_labels_patientwise` + real OOF preds;
                                                     shuffled AUROC must collapse to ~chance.
  3. PROXY-HIGH feature-ablation ΔAUROC            → `ablate_columns` drops named features from the
                                                     (x_num, x_cat, cards) triple, subset → the SAME CV.

Why column-subset (not a new model): the ablation question is "how much of 0.64 rides on this
feature", so the ONLY thing that may change between arms is which columns enter — the CV, the
train-only impute/standardize, the patient-disjoint fold assertion, and the FTT construction all stay
byte-identical to the scored path. `ablate_columns` therefore returns exactly the arguments
`cross_val_auroc` already takes.

pandas/torch-free: pure numpy + the encoder's public constants, so the light-weight helpers here stay
importable in the core (torch-free) test suite; only the CV call itself pulls the ml stack (lazily,
inside `clinical_encoder`).
"""

from __future__ import annotations

import numpy as np

from pinksight.models.clinical_encoder import FEATURES_CAT, FEATURES_NUM

# ---------------------------------------------------------------------------------------------------
# PROXY-HIGH ablation arms. Each arm names the AT-DIAGNOSIS, label-SAFE features to DROP from the 9.
# These are receptor-CORRELATED proxies (a leakage audit flagged their unquantified contribution to
# the 0.64), NOT label-defining fields — dropping them probes how modality-independent the signal is.
#   * Nottingham grade          — composite histologic grade; its mitotic component tracks proliferation
#   * Race and Ethnicity        — demographic proxy correlated with subtype prevalence
#   * the two NODAL features     — nodal burden correlates with aggressive (often TNBC) disease
# `full` (drop nothing) reproduces the headline; `minus_all_proxyhigh` drops every proxy at once. If
# `minus_grade`/`minus_all_proxyhigh` collapses toward the G1 radiomics floor (0.567), the headline is
# largely a grade→subtype correlation, not standalone clinical signal (verdict stated in the report).
_NODAL_FEATURES = (
    "Staging(Nodes)#(Nx replaced by -1)[N]",  # numeric nodal stage
    "Lymphadenopathy or Suspicious Nodes",  # categorical nodal finding
)
PROXY_HIGH_ARMS: dict[str, tuple[str, ...]] = {
    "full": (),  # all 9 features — the 0.64 baseline
    "minus_grade": ("Nottingham grade",),
    "minus_race": ("Race and Ethnicity",),
    "minus_nodal": _NODAL_FEATURES,
    "minus_all_proxyhigh": ("Nottingham grade", "Race and Ethnicity", *_NODAL_FEATURES),
}


def ablate_columns(
    x_num: np.ndarray,
    x_cat: np.ndarray,
    cards: list[int],
    drop: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Return (x_num, x_cat, cards) with the named features removed — the ONLY thing an ablation arm
    changes. Column order follows FEATURES_NUM / FEATURES_CAT (the order `load_xy` stacks), so the
    kept columns of `x_cat` stay aligned with the kept entries of `cards`.

    Never rebuilds features and never touches the label/split — the caller passes the result straight
    into the existing `cross_val_auroc`. Raises on an unknown feature name (typo guard) so a silent
    no-op ablation can't masquerade as a real arm.
    """
    drop_set = set(drop)
    known = set(FEATURES_NUM) | set(FEATURES_CAT)
    unknown = drop_set - known
    if unknown:
        raise ValueError(f"ablate_columns: unknown feature(s) {sorted(unknown)} (not in the 9-feature set)")

    num_keep = [i for i, name in enumerate(FEATURES_NUM) if name not in drop_set]
    cat_keep = [j for j, name in enumerate(FEATURES_CAT) if name not in drop_set]

    x_num_sub = x_num[:, num_keep] if num_keep else np.empty((x_num.shape[0], 0), dtype=x_num.dtype)
    x_cat_sub = x_cat[:, cat_keep] if cat_keep else np.empty((x_cat.shape[0], 0), dtype=x_cat.dtype)
    cards_sub = [cards[j] for j in cat_keep]
    return x_num_sub, x_cat_sub, cards_sub


def shuffle_labels_patientwise(
    y: list[int],
    groups: list,
    seed: int,
) -> list[int]:
    """Permute labels at the PATIENT level: a patient's label moves as a unit (never split within a
    patient). Returns a new label list aligned to the input rows.

    The label-shuffle sentinel: run the real CV on these permuted labels and any residual signal
    (AUROC >~0.55) is a leak path, not learning. Patient-level (not row-level) is the correct null for
    a patient-grouped CV — it destroys the feature→label link while preserving the group structure, so
    the fold assertion still holds and the null can't accidentally learn a per-patient shortcut. (In
    this cohort each patient owns exactly one row, so patient- and row-level shuffles coincide; doing
    it group-wise keeps it correct if the cohort ever gains multi-row patients.)
    """
    y_arr = np.asarray(y)
    groups_arr = np.asarray(groups)
    uniq_patients = list(dict.fromkeys(groups_arr.tolist()))  # stable first-seen order

    # each patient's (single, in this cohort) label — assert intra-patient label consistency
    pat_label: dict = {}
    for p in uniq_patients:
        labs = set(y_arr[groups_arr == p].tolist())
        if len(labs) != 1:
            raise ValueError(f"patient {p!r} has inconsistent labels {labs} — cannot patient-shuffle")
        pat_label[p] = labs.pop()

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq_patients))
    shuffled_pat_label = {uniq_patients[i]: pat_label[uniq_patients[perm[i]]] for i in range(len(uniq_patients))}
    return [int(shuffled_pat_label[p]) for p in groups_arr.tolist()]


def selfcheck() -> int:
    """No data, no torch: prove the two pure helpers hold their invariants.

    ablate_columns — drops the right columns, keeps cards aligned, rejects unknown names.
    shuffle_labels_patientwise — preserves the label MULTISET (a permutation), moves patients as units,
    and is deterministic per seed.
    """
    # ablate: synthetic 3-row matrix mirroring the real (4 num, 5 cat) layout
    xn = np.arange(3 * len(FEATURES_NUM), dtype=float).reshape(3, len(FEATURES_NUM))
    xc = np.arange(3 * len(FEATURES_CAT)).reshape(3, len(FEATURES_CAT))
    cards = [10 + j for j in range(len(FEATURES_CAT))]

    xn2, xc2, cards2 = ablate_columns(xn, xc, cards, ("Nottingham grade",))
    assert xn2.shape == (3, len(FEATURES_NUM) - 1), "grade drop should remove one numeric col"
    assert xc2.shape == (3, len(FEATURES_CAT)), "grade drop must not touch categoricals"
    assert cards2 == cards, "cards unchanged when only a numeric col is dropped"

    xn3, xc3, cards3 = ablate_columns(xn, xc, cards, ("Race and Ethnicity",))
    assert xc3.shape == (3, len(FEATURES_CAT) - 1) and len(cards3) == len(FEATURES_CAT) - 1, (
        "race drop should remove one categorical col AND its card"
    )
    # the dropped card is Race's position in FEATURES_CAT
    race_idx = FEATURES_CAT.index("Race and Ethnicity")
    assert cards3 == [c for j, c in enumerate(cards) if j != race_idx], "wrong card removed for race"

    # all-proxyhigh drops grade (num) + nodal-N (num) + race (cat) + lymphadenopathy (cat)
    xn4, xc4, cards4 = ablate_columns(xn, xc, cards, PROXY_HIGH_ARMS["minus_all_proxyhigh"])
    assert xn4.shape[1] == len(FEATURES_NUM) - 2, "all-proxyhigh drops 2 numeric (grade + nodal-N)"
    assert xc4.shape[1] == len(FEATURES_CAT) - 2, "all-proxyhigh drops 2 categoricals (race + lymphadenopathy)"

    try:
        ablate_columns(xn, xc, cards, ("Not A Real Feature",))
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("ablate_columns must reject unknown feature names")

    # patient-shuffle: 4 patients, one row each; permutation preserves the label multiset
    y = [0, 0, 1, 1]
    groups = ["A", "B", "C", "D"]
    s0 = shuffle_labels_patientwise(y, groups, 0)
    assert sorted(s0) == sorted(y), "shuffle must be a permutation (label multiset preserved)"
    assert s0 == shuffle_labels_patientwise(y, groups, 0), "same seed must be deterministic"

    # patient-as-unit: a 2-row patient keeps its two rows' label identical after shuffle
    y2 = [0, 0, 1, 1]
    groups2 = ["A", "A", "B", "B"]  # 2 patients, 2 rows each, consistent within patient
    s2 = shuffle_labels_patientwise(y2, groups2, 3)
    assert s2[0] == s2[1] and s2[2] == s2[3], "a patient's rows must share one shuffled label"
    assert sorted(s2) == sorted(y2), "2-row-patient shuffle must still preserve the multiset"

    print("leakage_probe selfcheck OK — ablate_columns aligns cards & rejects typos; patient-shuffle preserves multiset")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(selfcheck())
