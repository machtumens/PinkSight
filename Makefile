PYTHON ?= python3
PIP    ?= pip

.DEFAULT_GOAL := help
.PHONY: help install demo reproduce

help:
	@printf "  \033[36m%-12s\033[0m %s\n" \
		help       "Show this help (all targets)" \
		install    "pip install -e . (base, zero extras — the zero-setup clone-and-run path)" \
		demo       "Zero-data, zero-network 3-tier synthetic demo (SYNTHETIC — NOT A RESULT)" \
		reproduce  "Real-data path: reproduce G3/G5/Track-C/dispatch rungs (needs data per data/README.md)"

install:
	$(PIP) install -e .

demo:
	$(PYTHON) scripts/run_demo.py

reproduce:
	$(PYTHON) scripts/run_reproduce.py
