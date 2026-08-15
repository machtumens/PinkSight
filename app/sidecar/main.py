"""
PinkSight inference sidecar — FastAPI on 127.0.0.1:8756.

This is the seam where PyTorch/MONAI lives. Right now it returns a MOCKED
characterisation so the desktop UI is demoable end-to-end. Swap `build_report`
for a real model call when the G2 encoder exists — the response contract stays.

Claim-ledger enforcement: `assert_ledger_safe` rejects any payload containing
forbidden framing. Integrity is a code path, not a promise.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except ImportError:  # allow the test + guard to run without FastAPI installed
    FastAPI = None  # type: ignore
    BaseModel = object  # type: ignore

# Pure, read-only cohort/modality -> harness dispatcher (scripts/). Same sys.path pattern as
# pinksight_report_adapter.py: main.py -> sidecar/ -> desktop/ -> repo root (parents[2]). Pure stdlib,
# imported unconditionally at module scope (no import guard needed).
_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from pinksight_dispatch import dispatch  # noqa: E402 — sys.path injected just above

# Forbidden framing from docs/CLAIM_LEDGER.md claim ledger. Substring match, case-insensitive.
FORBIDDEN_TERMS = [
    "early detection",
    "pre-detection",
    "growth rate",
    "doubling time",
    "tumour kinetics",
    "tumor kinetics",
    "screening",
    "cross-institution",
]


def assert_ledger_safe(payload: dict) -> None:
    """Raise if any string field drifts into forbidden framing."""
    blob = " ".join(str(v) for v in _walk_strings(payload)).lower()
    for term in FORBIDDEN_TERMS:
        if re.search(re.escape(term), blob):
            raise ValueError(f"claim-ledger violation: forbidden term '{term}' in output")


def _walk_strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _walk_strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk_strings(v)


def build_report(
    study_id: str,
    *,
    live_override: bool | None = None,
    cohort: str | None = None,
    modalities: list[str] | None = None,
) -> dict:
    """Characterisation report for one study.

    Default (byte-identical to the original demo): a fixed MOCK characterisation. When the environment
    variable ``PINKSIGHT_SIDECAR_LIVE=1`` is set, return one real FORWARD-ONLY synthetic pipeline pass
    (Stages 0-6 at n=1) via ``pinksight_report_adapter`` instead — still non-reportable by construction
    (``provenance.datasetTag == "SYNTHETIC — NOT A RESULT"``), never a scientific result. The gate is
    default-OFF, so the existing demo behaviour is unchanged unless a run explicitly opts in.

    ``live_override`` (per-request) can only ever NARROW what the env gate already permits, never widen
    it: live mode needs BOTH ``PINKSIGHT_SIDECAR_LIVE=1`` AND (``live_override`` is None or True). When
    ``live_override`` is None (every existing caller), behaviour is byte-identical to today.

    When both ``cohort`` and ``modalities`` are given, an additive ``dispatch`` routing block (pure,
    read-only lookup) is echoed into the report; when either is absent, the report is unchanged.
    """
    env_live = os.environ.get("PINKSIGHT_SIDECAR_LIVE") == "1"
    use_live = env_live if live_override is None else (env_live and live_override)
    if use_live:
        # Live: one real forward-only synthetic pipeline pass. The adapter degrades to a mock when the
        # ml stack (torch/monai) is absent, so this branch never crashes the sidecar on import.
        from pinksight_report_adapter import get_synthetic_report

        report = get_synthetic_report(study_id)
    else:
        report = {
            "studyId": study_id,
            "subtype": {
                "label": "Luminal A",
                "probability": 0.71,
                "uncertainty": [0.58, 0.83],
                "abstained": False,
            },
            "ki67Descriptor": (
                "Descriptive companion — imaging correlate of Ki-67 index at "
                "diagnosis (aggressiveness snapshot, not kinetics)."
            ),
            "nottinghamGrade": {"label": "NHG1", "probability": 0.68, "uncertainty": [0.55, 0.80]},
            "calibration": {"ece": 0.041, "band": "good"},
            "modalities": [
                {"modality": "clinical", "present": True, "contribution": 0.135},
                {"modality": "mri", "present": True, "contribution": -0.025},
                {"modality": "path", "present": False, "contribution": 0.0},
                {"modality": "genomic", "present": False, "contribution": 0.0},
            ],
            "audit": {
                "modelHash": "sha256:demo0000",
                "seed": 42,
                "split": "split_v2 (scanner holdout)",
                "generatedAt": "mocked",
            },
        }
    if cohort is not None and modalities is not None:
        result = dispatch(cohort, modalities)
        report["dispatch"] = {
            "cohort": cohort,
            "modalities": list(modalities),
            "harnessScript": result.harness_script,
            "status": result.status,
            "crossCohortGradient": result.cross_cohort_gradient,
            "note": result.note,
        }
    assert_ledger_safe(report)  # never ship a ledger-violating payload (runs on whichever path)
    return report


if FastAPI is not None:
    app = FastAPI(title="PinkSight sidecar")

    # Dev-only CORS: `npm run dev` serves the UI from localhost:1420, a cross-origin caller to this
    # sidecar (127.0.0.1:8756). Packaged Tauri fetches in a same trust context and never sends these
    # Origins, so this allowance is inert in production. ponytail: three fixed origins, never "*".
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:1420", "http://127.0.0.1:1420"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    class InferRequest(BaseModel):
        studyId: str
        cohort: str | None = None
        modalities: list[str] | None = None
        live: bool | None = None

    @app.post("/infer")
    def infer(req: "InferRequest"):
        try:
            return build_report(
                req.studyId,
                live_override=req.live,
                cohort=req.cohort,
                modalities=req.modalities,
            )
        except ValueError as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/dispatch")
    def get_dispatch(cohort: str, modalities: str):
        # Routing decision only — mirrors the dispatch block echoed into /infer, exposed as its own GET
        # so the picker can check routing without triggering any run. Never returns a per-organ number.
        # assert_ledger_safe runs on the payload unconditionally, matching /infer's guard pattern.
        try:
            mods = [m for m in modalities.split(",") if m]
            result = dispatch(cohort, mods)
            payload = {
                "cohort": cohort,
                "modalities": mods,
                "harnessScript": result.harness_script,
                "status": result.status,
                "crossCohortGradient": result.cross_cohort_gradient,
                "note": result.note,
            }
            assert_ledger_safe(payload)
            return payload
        except ValueError as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    def health():
        return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8756)
