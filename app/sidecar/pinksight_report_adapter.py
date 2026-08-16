from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from pinksight.eval.e2e_report_contract import (
    KI67_DESCRIPTOR_DEFAULT,
    SYNTHETIC_TAG,
    build_report,
)

_VALID_STREAMS = ("negative_control", "positive_control")

try:
    _SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"  
    if str(_SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS_DIR))
    from e2e_synthetic_harness_run import run_stream as _run_stream  

    _HARNESS_IMPORT_ERROR: Exception | None = None
except Exception as exc:  
    _run_stream = None  
    _HARNESS_IMPORT_ERROR = exc


def _degraded_report(study_id: str, stream: str) -> dict[str, Any]:
    manifest = {
        "manifest_sha256": "0" * 64,
        "generated_at": "unavailable (ml stack absent)",
        "config": {
            "n": 1,
            "seed": 0,
            "cube_size": 16,
            "channels": "pre_post",
            "effect_size": 1.2 if stream == "positive_control" else 0.0,
            "stream_name": stream,
            "git_commit": "unknown",
        },
    }
    subtype_out = {"label": "Luminal A", "probability": 0.5, "uncertainty": [0.4, 0.6], "abstained": False}
    modalities = [
        {"modality": "clinical", "present": True, "contribution": 0.0},
        {"modality": "mri", "present": True, "contribution": 0.0},
        {"modality": "path", "present": False, "contribution": 0.0},
        {"modality": "genomic", "present": False, "contribution": 0.0},
    ]
    control_block = {
        "stream": stream,
        "verdict": "NOT_ASSESSED",
        "expected": "control sentinel needs a patient-grouped CV cohort (skipped at tiny N)",
        "note": (
            "ml stack (torch/monai) unavailable on this host — the forward-only synthetic plumbing "
            "chain was not run; returning a shape-complete non-reportable placeholder"
        ),
    }
    return build_report(
        stream,
        manifest,
        subtype_out,
        {"descriptor": KI67_DESCRIPTOR_DEFAULT, "stratum": "not_assessed"},
        None,
        modalities,
        None,
        control_block,
        study_id=study_id,
    )


def get_synthetic_report(study_id: str, stream: str = "positive_control") -> dict[str, Any]:
    if stream not in _VALID_STREAMS:
        raise ValueError(f"stream must be one of {_VALID_STREAMS}, got {stream!r}")

    if _run_stream is None:
        return _degraded_report(study_id, stream)

    res = _run_stream(stream, n=1, seed=0, xai_subsample=0)
    report = res["report"]
    return {**report, "studyId": study_id}


def selfcheck() -> int:
    report = get_synthetic_report("SELFCHECK-0001", stream="positive_control")
    required = ("studyId", "subtype", "ki67Descriptor", "nottinghamGrade", "calibration",
                "modalities", "audit", "provenance")
    missing = [k for k in required if k not in report]
    assert not missing, f"report missing desktop-required fields: {missing}"
    assert report["studyId"] == "SELFCHECK-0001", "studyId label was not applied"
    assert report["provenance"]["datasetTag"] == SYNTHETIC_TAG, "provenance tag must stay SYNTHETIC"
    mode = "degraded (ml stack absent)" if _run_stream is None else "live (forward-only n=1 chain)"
    print(f"pinksight_report_adapter selfcheck OK — {mode}; report SYNTHETIC-tagged, shape-complete.")  
    return 0


if __name__ == "__main__":
    raise SystemExit(selfcheck())
