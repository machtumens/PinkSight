"""P05 clinical encoder — leak-guard tests (LOCK-2). Torch-free: the ml stack is lazy-imported,
so these run in the core suite without `--extra ml`."""

import pytest

from pinksight import FORBIDDEN_FEATURES
from pinksight.models import clinical_encoder as ce


def test_features_disjoint_from_forbidden():
    assert not (set(ce.FEATURES) & set(FORBIDDEN_FEATURES))
    assert ce.selfcheck() == 0


def test_leak_guard_bites_on_poisoned_inputs(monkeypatch):
    """The guard must FAIL if a label-defining field ever enters FEATURES — not just pass today."""
    monkeypatch.setattr(ce, "FEATURES", ce.FEATURES + ("Mol Subtype",))
    with pytest.raises(ce.LeakageError):
        ce._assert_leak_free()
