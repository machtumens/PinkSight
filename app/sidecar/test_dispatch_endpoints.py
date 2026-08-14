"""Pytest coverage for the sidecar's new /dispatch endpoint + /infer dispatch-echo (Group A).

Unlike the framework-free ``test_sidecar.py``, this drives the actual HTTP layer via
``fastapi.testclient.TestClient``. Run:

    uv run --with fastapi --with uvicorn --with pydantic --with httpx pytest \
        desktop/sidecar/test_dispatch_endpoints.py -q

``from main import app`` injects ``scripts/`` onto ``sys.path`` as a module-load side effect, so the
``import pinksight_dispatch`` below resolves without a second path hack.
"""

import pytest

pytest.importorskip("fastapi", reason="fastapi is an ml/arms/trackb-extra dep — skip cleanly under a base install; sidecar HTTP tests need fastapi TestClient")
import os

import pinksight_dispatch
from fastapi.testclient import TestClient
from main import (
    app,  # importing main injects scripts/ onto sys.path (used by the import just below)
)

client = TestClient(app)

# The response schema is CLOSED: exactly these 6 fields, for every routing-table row. No numeric
# prediction field can ever appear — this is the direct, mechanical proof of AC5 (no per-organ number).
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
    # env var unset in this test process -> mock path taken; audit.generatedAt is the mock branch tell.
    assert body["audit"]["generatedAt"] == "mocked"


def test_dispatch_payload_schema_closed_no_numeric_field():
    # Every documented routing-table row (10 WIRED + 2 NOT-WIRED) returns EXACTLY the 6 closed fields —
    # no AUROC/Pearson-r/CV numeric field can leak in for any row. Schema-closed proof of AC5.
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
    # With the env gate ON and live_override True, /infer takes the synthetic seam. The adapter degrades
    # to a shape-complete SYNTHETIC-tagged placeholder when torch/monai is absent, so this passes on a
    # box with only pinksight+numpy (no ml extra) — still provably NOT the mock fixture.
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
