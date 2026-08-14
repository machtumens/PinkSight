"""DeLong CI + ECE (CLAUDE.md eval rule / red-team M1). Torch-free — pure numpy core suite."""

import numpy as np

from pinksight.metrics import delong_ci, ece, selfcheck


def test_selfcheck_passes():
    assert selfcheck() == 0


def test_delong_perfect_separation():
    y = np.r_[np.zeros(50), np.ones(50)].astype(int)
    auc, lo, hi = delong_ci(y, y.astype(float))  # scores == labels → AUC 1.0
    assert auc == 1.0 and lo <= 1.0 <= hi


def test_delong_ci_brackets_and_bounded():
    rng = np.random.default_rng(1)
    y = np.r_[np.zeros(300), np.ones(300)].astype(int)
    s = y + rng.normal(0, 0.6, 600)
    auc, lo, hi = delong_ci(y, s)
    assert 0.0 <= lo <= auc <= hi <= 1.0
    assert lo < auc < hi  # nonzero-width CI for a noisy, non-perfect classifier


def test_delong_single_class_returns_nan():
    # TICKET-012: single-class OOF must degrade to NaN (not raise) so ablation over a
    # small/imbalanced fold surfaces a clear NaN instead of an opaque crash mid-aggregate.
    auc, lo, hi = delong_ci(np.ones(10, int), np.random.default_rng(0).random(10))
    assert np.isnan(auc) and np.isnan(lo) and np.isnan(hi)


def test_ece_calibrated_vs_miscalibrated():
    assert ece(np.zeros(200, int), np.full(200, 0.99)) > 0.9  # confidently wrong
    assert ece(np.r_[np.zeros(100), np.ones(100)].astype(int), np.r_[np.zeros(100), np.ones(100)]) < 1e-9


def test_delong_ci_width_matches_bootstrap():
    """Pin the CI WIDTH, not just the point estimate.

    The other DeLong tests check `auc` against Mann-Whitney and that `lo <= auc <= hi`. A
    constant-factor error in the structural-components variance passes all of them while making
    every reported interval systematically too wide or too narrow -- and CI width is a decision
    criterion here (LOCK-2: "CI lower bound clears 0.50", "CI excludes zero"), not decoration.

    The oracle is a bootstrap percentile CI over the same scores: an independent variance
    estimate rather than a memorised constant, so it survives refactoring of the DeLong
    internals. Seeds are pinned, so this is deterministic -- it cannot flake.

    Sensitivity (measured): a 2x or 0.5x variance error is caught; a ~15% error is NOT.
    This is a guard against gross breakage, not a precision audit.
    """
    n, n_boot = 600, 300
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(n // 2), np.ones(n // 2)].astype(int)
    s = y + rng.normal(0, 1.0, n)  # AUC ~0.77 -- the project's realistic operating range

    auc, lo, hi = delong_ci(y, s)
    delong_width = hi - lo
    assert 0.0 < lo < auc < hi < 1.0, "CI must be interior (no clipping) for this fixture"

    # Independent oracle: stratified bootstrap over patients, percentile CI on the AUC.
    boot = np.random.default_rng(12345)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    aucs = np.empty(n_boot)
    for i in range(n_boot):
        p = boot.choice(pos, len(pos), replace=True)
        q = boot.choice(neg, len(neg), replace=True)
        yy = np.r_[np.ones(len(p), int), np.zeros(len(q), int)]
        aucs[i] = delong_ci(yy, np.r_[s[p], s[q]])[0]
    b_lo, b_hi = np.percentile(aucs, [2.5, 97.5])
    boot_width = b_hi - b_lo

    ratio = delong_width / boot_width
    assert abs(ratio - 1.0) < 0.20, (
        f"DeLong CI width {delong_width:.4f} disagrees with bootstrap {boot_width:.4f} "
        f"(ratio {ratio:.3f}) -- suspect the structural-components variance in delong_ci()"
    )
