# PinkSight — public clone-and-run Makefile (Team TestEin, OPSI 2026).
#
# Scope: the "clone-and-run guarantee" only — help / install / demo / reproduce.
# Uses plain python3 / pip (NO `uv` — a public cloner is not assumed to have uv installed).
# CI-parity targets (lint / format / ledger / test / audit) are intentionally NOT here.
#
# Quickstart:
#   git clone … && cd pinksight
#   pip install -e .          # base, zero extras — enough for `make demo` Tier 0/1-skip
#   make demo                 # zero-data, zero-network synthetic control-sentinel proof
# Unlock heavier demo tiers:
#   pip install -e '.[arms]'  # Tier 1: Track-B / Track-C (sklearn + lightgbm, no torch)
#   pip install -e '.[ml]'    # Tier 2: Track-A / fastMRI-NYU (torch + monai, heavy)

PYTHON ?= python3
PIP    ?= pip

.DEFAULT_GOAL := help
.PHONY: help install demo reproduce

help:  ## Show this help (all targets)
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## pip install -e . (base, zero extras — the zero-setup clone-and-run path)
	$(PIP) install -e .

demo:  ## Zero-data, zero-network 3-tier synthetic demo (SYNTHETIC — NOT A RESULT)
	$(PYTHON) scripts/run_demo.py

reproduce:  ## Real-data path: reproduce G3/G5/Track-C/dispatch rungs (needs data per data/README.md)
	$(PYTHON) scripts/run_reproduce.py
