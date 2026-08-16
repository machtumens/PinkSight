
import pytest

from pinksight import FORBIDDEN_FEATURES
from pinksight.models import clinical_encoder as ce


def test_features_disjoint_from_forbidden():
    assert not (set(ce.FEATURES) & set(FORBIDDEN_FEATURES))
    assert ce.selfcheck() == 0


def test_leak_guard_bites_on_poisoned_inputs(monkeypatch):
    monkeypatch.setattr(ce, "FEATURES", ce.FEATURES + ("Mol Subtype",))
    with pytest.raises(ce.LeakageError):
        ce._assert_leak_free()
