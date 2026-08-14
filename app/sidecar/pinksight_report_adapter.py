"""Desktop sidecar <-> E2E-synthetic-harness bridge (Stage 7 wiring).

``get_synthetic_report`` runs the documented PinkSight v2 pipeline (Stages 0-6) FORWARD-ONLY at
n=1, reusing Teammate A's ``scripts/e2e_synthetic_harness_run.run_stream`` verbatim (no separate
single-patient code path), and returns the shared-contract report dict so the desktop UI can render
one real pipeline pass instead of the fixed mock.

INTEGRITY (constitutional — same firewall as the harness):
  * NON-REPORTABLE BY CONSTRUCTION. The returned report keeps ``provenance.datasetTag == SYNTHETIC_TAG``
    and the generation manifest hash. Only the top-level ``studyId`` label is swapped to the caller's
    id so the UI shows a sensible study label — the provenance block still honestly says
    "SYNTHETIC — NOT A RESULT". A synthetic plumbing pass is NEVER a scientific result.
  * CHARACTERISATION FRAMING ONLY. At-diagnosis subtype characterisation, Ki-67 stratification, and
    explainable fusion — never detection, proliferation dynamics, or cross-institution transfer.
  * FORWARD-ONLY, $0-LOCAL, CPU. n=1 forward passes through tiny random-init frozen models; no
    ``.backward()``, no optimizer, no training on synthetic data.

ROBUSTNESS. The heavy pipeline import (``run_stream`` pulls torch/monai) is wrapped in the SAME
try/except-degrade-to-mock pattern ``main.py`` uses for FastAPI/pydantic: a demo box may not have the
ml stack installed, and the sidecar must never crash on import. When the ml stack is absent,
``get_synthetic_report`` returns a shape-complete, still-SYNTHETIC-tagged, NOT_ASSESSED placeholder
built via the torch-free ``e2e_report_contract.build_report``.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# The shared contract module is torch-free by design (stdlib + numpy only), so these are safe at
# module load even on a demo box without the ml extra — they back both the live and the degrade path.
from pinksight.eval.e2e_report_contract import (
    KI67_DESCRIPTOR_DEFAULT,
    SYNTHETIC_TAG,
    build_report,
)

_VALID_STREAMS = ("negative_control", "positive_control")

# Heavy pipeline import, guarded. ``run_stream`` transitively imports torch/monai + every pipeline
# organ; on a box without the ml stack this raises, and we degrade to a mock instead of crashing.
try:
    _SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"  # sidecar -> desktop -> repo root
    if str(_SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS_DIR))
    from e2e_synthetic_harness_run import run_stream as _run_stream  # type: ignore

    _HARNESS_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001 — any import failure (missing torch/monai/pinksight) degrades
    _run_stream = None  # type: ignore
    _HARNESS_IMPORT_ERROR = exc


def _degraded_report(study_id: str, stream: str) -> dict[str, Any]:
    """Shape-complete, non-reportable placeholder for when the ml stack is unavailable.

    Built with the torch-free ``build_report`` so the report is byte-shape-identical to a live one
    (same required keys, same SYNTHETIC provenance tag) — only the pipeline numbers are absent
    (control verdict NOT_ASSESSED). Never masquerades as a real result.
    """
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
    """Run the Stage 0-6 pipeline FORWARD-ONLY at n=1 and return the shared-contract report dict.

    ``study_id`` labels the returned report's top-level ``studyId`` (so the UI shows a sensible label);
    the ``provenance`` block is untouched and still carries ``datasetTag == SYNTHETIC_TAG`` — the report
    is non-reportable by construction. ``stream`` selects the ``positive_control`` (default) or
    ``negative_control`` synthetic sub-cohort.

    XAI is intentionally skipped at the desktop path (``xai_subsample=0``) — it would write ``.npy``
    saliency maps as a side effect and is meaningless on a single random-init forward pass. The control
    sentinel is NOT_ASSESSED at n=1 (below the CV minimum), which ``run_stream`` handles gracefully.

    Degrades to a shape-complete NOT_ASSESSED placeholder (never crashes) when the ml stack is absent.
    """
    if stream not in _VALID_STREAMS:
        raise ValueError(f"stream must be one of {_VALID_STREAMS}, got {stream!r}")

    if _run_stream is None:
        return _degraded_report(study_id, stream)

    res = _run_stream(stream, n=1, seed=0, xai_subsample=0)
    report = res["report"]
    # Swap only the display label; provenance.datasetTag stays SYNTHETIC_TAG (non-reportable).
    return {**report, "studyId": study_id}


def selfcheck() -> int:
    """Runnable check (mirrors the repo's ``selfcheck`` convention): a synthetic report is produced,
    carries the non-reportable SYNTHETIC tag, keeps the requested studyId, and has every desktop-
    required field. Works on both the live and the degraded path. Exits 0."""
    report = get_synthetic_report("SELFCHECK-0001", stream="positive_control")
    required = ("studyId", "subtype", "ki67Descriptor", "nottinghamGrade", "calibration",
                "modalities", "audit", "provenance")
    missing = [k for k in required if k not in report]
    assert not missing, f"report missing desktop-required fields: {missing}"
    assert report["studyId"] == "SELFCHECK-0001", "studyId label was not applied"
    assert report["provenance"]["datasetTag"] == SYNTHETIC_TAG, "provenance tag must stay SYNTHETIC"
    mode = "degraded (ml stack absent)" if _run_stream is None else "live (forward-only n=1 chain)"
    print(f"pinksight_report_adapter selfcheck OK — {mode}; report SYNTHETIC-tagged, shape-complete.")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(selfcheck())
