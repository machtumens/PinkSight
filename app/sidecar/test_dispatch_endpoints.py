
import pytest

pytest.importorskip("fastapi", reason="fastapi is an ml/arms/trackb-extra dep — skip cleanly under a base install; sidecar HTTP tests need fastapi TestClient")
import os

import pinksight_dispatch
from fastapi.testclient import TestClient
from main import (
    app,  
)

client = TestClient(app)

DISPATCH_FIELDS = {"cohort", "modalities", "harnessScript", "status", "crossCohortGradient", "note"}


def test_dispatch_endpoint_wired_row():
    resp = client.get("/dispatch", params={"cohort": "duke", "modalities": "mri,clinical"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "WIRED"
    assert body["crossCohortGradient"] is False
    assert body["harnessScript"] == "scripts/train_g3_hierarchical.py"


def test_dispatch_endpoint_not_wired_row():
    resp = client.get("/dispatch", params={"cohort": "duke", "modalities": "clinical"})
    assert resp.status_code == 200
    body = resp.json()
    assert "NOT WIRED" in body["status"]
    assert body["harnessScript"] is None


def test_infer_echoes_dispatch_block():
    resp = client.post(
        "/infer",
        json={"studyId": "X", "cohort": "duke", "modalities": ["mri", "clinical"], "live": False},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["dispatch"]["status"] == "WIRED"
    assert body["audit"]["generatedAt"] == "mocked"


def test_dispatch_payload_schema_closed_no_numeric_field():
    rows = list(pinksight_dispatch.COHORT_HARNESS_REGISTRY) + list(pinksight_dispatch.NOT_WIRED_COMBOS)
    assert len(rows) == 12, "expected 10 WIRED + 2 NOT-WIRED = 12 documented routing rows"
    for cohort, modalities in rows:
        resp = client.get(
            "/dispatch", params={"cohort": cohort, "modalities": ",".join(sorted(modalities))}
        )
        assert resp.status_code == 200, f"row ({cohort}, {sorted(modalities)}) did not return 200"
        assert set(resp.json().keys()) == DISPATCH_FIELDS, (
            f"row ({cohort}, {sorted(modalities)}) returned unexpected fields: "
            f"{set(resp.json().keys()) ^ DISPATCH_FIELDS}"
        )


def test_infer_live_true_uses_synthetic_path():
    prev = os.environ.get("PINKSIGHT_SIDECAR_LIVE")
    os.environ["PINKSIGHT_SIDECAR_LIVE"] = "1"
    try:
        resp = client.post("/infer", json={"studyId": "X", "live": True})
        assert resp.status_code == 200
        assert resp.json()["provenance"]["datasetTag"] == "SYNTHETIC — NOT A RESULT"
    finally:
        if prev is None:
            os.environ.pop("PINKSIGHT_SIDECAR_LIVE", None)
        else:
            os.environ["PINKSIGHT_SIDECAR_LIVE"] = prev
