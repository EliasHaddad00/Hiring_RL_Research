PY ?= .venv/Scripts/python.exe

.PHONY: test test-slow g1 figures paper-setup

test:
	$(PY) -m pytest experiments/tests

test-slow:
	$(PY) -m pytest experiments/tests -m slow

# Gate G1 checks; each writes to results/g1/
g1:
	$(PY) experiments/scripts/reproduce_rules.py
	$(PY) experiments/scripts/solve_small.py
	$(PY) experiments/scripts/measure_throughput.py
	$(PY) experiments/scripts/check_library.py
	$(PY) experiments/scripts/check_slot_view.py

# Download the official ICML 2026 style files into paper/ (they are not kept in the repository).
paper-setup:
	$(PY) experiments/scripts/paper_setup.py

# Every figure and table of the paper, from saved results only (one script each).
figures:
	$(PY) experiments/analysis/make_figures.py
