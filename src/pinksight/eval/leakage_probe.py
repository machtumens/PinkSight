
from __future__ import annotations

import numpy as np

from pinksight.models.clinical_encoder import FEATURES_CAT, FEATURES_NUM

_NODAL_FEATURES = (
    "Staging(Nodes)#(Nx replaced by -1)[N]",  
    "Lymphadenopathy or Suspicious Nodes",  
)
PROXY_HIGH_ARMS: dict[str, tuple[str, ...]] = {
    "full": (),  
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
    y_arr = np.asarray(y)
    groups_arr = np.asarray(groups)
    uniq_patients = list(dict.fromkeys(groups_arr.tolist()))  

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
    race_idx = FEATURES_CAT.index("Race and Ethnicity")
    assert cards3 == [c for j, c in enumerate(cards) if j != race_idx], "wrong card removed for race"

    xn4, xc4, cards4 = ablate_columns(xn, xc, cards, PROXY_HIGH_ARMS["minus_all_proxyhigh"])
    assert xn4.shape[1] == len(FEATURES_NUM) - 2, "all-proxyhigh drops 2 numeric (grade + nodal-N)"
    assert xc4.shape[1] == len(FEATURES_CAT) - 2, "all-proxyhigh drops 2 categoricals (race + lymphadenopathy)"

    try:
        ablate_columns(xn, xc, cards, ("Not A Real Feature",))
    except ValueError:
        pass
    else:  
        raise AssertionError("ablate_columns must reject unknown feature names")

    y = [0, 0, 1, 1]
    groups = ["A", "B", "C", "D"]
    s0 = shuffle_labels_patientwise(y, groups, 0)
    assert sorted(s0) == sorted(y), "shuffle must be a permutation (label multiset preserved)"
    assert s0 == shuffle_labels_patientwise(y, groups, 0), "same seed must be deterministic"

    y2 = [0, 0, 1, 1]
    groups2 = ["A", "A", "B", "B"]  
    s2 = shuffle_labels_patientwise(y2, groups2, 3)
    assert s2[0] == s2[1] and s2[2] == s2[3], "a patient's rows must share one shuffled label"
    assert sorted(s2) == sorted(y2), "2-row-patient shuffle must still preserve the multiset"

    print("leakage_probe selfcheck OK — ablate_columns aligns cards & rejects typos; patient-shuffle preserves multiset")  
    return 0


if __name__ == "__main__":
    raise SystemExit(selfcheck())
