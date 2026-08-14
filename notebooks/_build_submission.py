"""Build the jury submission notebooks.

Reads each self-contained organ script, strips every `#` comment, wraps the code
in a clean notebook with one plain-language intro cell, and writes it to this
folder. Stdlib only. Re-run with: python3 submission/_build_submission.py
"""
import io
import json
import re
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent

NOT_CLAIM = (
    "This is a wiring check on **fake (synthetic) data**. "
    "It is **not** a scientific result. "
    "It uses no real patient data and never compares one group against another."
)

# organ -> (nice title, one-line "what it shows", extra note)
ORGANS = {
    "scripts/e2e_synthetic_harness_run.py": (
        "Track-A control check (MRI-style)",
        "Makes fake patients with fake MRI-style cubes, then checks the pipeline: "
        "a real-signal run should score well and a shuffled-label run should score "
        "about chance. If both happen, the wiring is correct.",
        "Needs `torch`.",
    ),
    "scripts/e2e_synthetic_trackb_run.py": (
        "Track-B control check (slides + genes)",
        "Same control check for the slide + gene stream, using fake numbers.",
        "Needs `pandas`.",
    ),
    "scripts/e2e_synthetic_trackc_run.py": (
        "Track-C control check (clinical table)",
        "Same control check for the clinical-table stream, using a fake table.",
        "Needs `pandas`.",
    ),
    "scripts/e2e_synthetic_fastmri_run.py": (
        "fastMRI control check (image cubes)",
        "Same control check for the fastMRI image stream, using fake cubes.",
        "Needs `torch`.",
    ),
    "scripts/e2e_synthetic_fused_10k_validation_run.py": (
        "Fused 4-stream trainability check",
        "Joins all four fake streams into one fake patient and trains a small "
        "fusion model. Shows the fusion wiring trains end to end on fake data.",
        "Needs `torch`.",
    ),
    "scripts/e2e_synthetic_companion_index_run.py": (
        "Run-of-runs index",
        "Runs each organ's control check and lists every verdict on its own line. "
        "The verdicts are never added up or compared.",
        "Needs `torch`.",
    ),
    "scripts/eval_demo.py": (
        "Scoring helper demo",
        "Shows how the scoring/metrics helper works on one tiny fixed example.",
        "Run after `pip install -e .`.",
    ),
    "scripts/stats_demo.py": (
        "Stats helper demo",
        "Shows how the statistics helper (confidence intervals) works on one tiny "
        "fixed example.",
        "Run after `pip install -e .`.",
    ),
    "scripts/xai_demo.py": (
        "Explainability helper demo",
        "Shows how the explainability helper works on one tiny fixed example.",
        "Run after `pip install -e .`.",
    ),
    "scripts/trackb_demo.py": (
        "Track-B helper demo",
        "Shows how the Track-B slide-bag helper works on one tiny fixed example.",
        "Run after `pip install -e .`.",
    ),
    "scripts/xai_counterfactual_grade.py": (
        "Counterfactual explainability self-check",
        "Runs the counterfactual-explainability self-check on built-in fixtures.",
        "Run after `pip install -e .`.",
    ),
}


def strip_comments(source: str) -> str:
    """Remove every `#` comment, keep code and layout. Falls back to raw source."""
    try:
        out = []
        last_line, last_col = 1, 0
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            ttype, tstr, (sl, sc), (el, ec), _ = tok
            if sl > last_line:
                last_col = 0
            if sc > last_col:
                out.append(" " * (sc - last_col))
            if ttype != tokenize.COMMENT:
                out.append(tstr)
            last_line, last_col = el, ec
        text = "".join(out)
    except (tokenize.TokenError, IndentationError):
        text = source
    text = re.sub(r"\n{3,}", "\n\n", text)  # collapse noise
    return text.strip("\n") + "\n"


def as_source(text: str) -> list[str]:
    """nbformat wants a list of lines, each keeping its newline."""
    lines = text.splitlines(keepends=True)
    return lines or [""]


def split_cells(code: str) -> list[str]:
    """Split into a setup cell plus one cell per top-level def/class block."""
    boundary = re.compile(r"^(def |class |@|if __name__)")
    cells, cur = [], []
    for line in code.splitlines(keepends=True):
        if boundary.match(line) and cur and "".join(cur).strip():
            cells.append("".join(cur))
            cur = [line]
        else:
            cur.append(line)
    if cur:
        cells.append("".join(cur))
    return [c for c in cells if c.strip()]


def md_cell(lines: list[str]) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": as_source("".join(lines))}


def code_cell(code: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": as_source(code)}


def build(rel: str, title: str, shows: str, note: str) -> Path:
    raw = (ROOT / rel).read_text(errors="ignore")
    clean = strip_comments(raw)
    intro = [
        f"# {title}\n", "\n",
        f"**What this shows:** {shows}\n", "\n",
        f"**Honest note:** {NOT_CLAIM}\n", "\n",
        f"_Source: `{rel}` · {note}_\n",
    ]
    cells = [md_cell(intro)] + [code_cell(c) for c in split_cells(clean)]
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    name = Path(rel).stem
    if name.startswith("e2e_synthetic_"):
        name = name[len("e2e_synthetic_"):]
    out = HERE / f"{name}.ipynb"
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    return out


def main() -> None:
    written = [build(rel, *meta) for rel, meta in ORGANS.items()]
    for p in written:
        print("wrote", p.relative_to(ROOT))
    print(f"\n{len(written)} notebooks built.")


if __name__ == "__main__":
    main()
