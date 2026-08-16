PYTHON ?= python3
PIP    ?= pip

.DEFAULT_GOAL := help
.PHONY: help install demo reproduce

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:
	$(PIP) install -e .

demo:
	$(PYTHON) scripts/run_demo.py

reproduce:
	$(PYTHON) scripts/run_reproduce.py
