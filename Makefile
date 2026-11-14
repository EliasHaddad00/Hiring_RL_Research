PY ?= .venv/Scripts/python.exe

.PHONY: test test-slow g1 figures

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

# Stub: analysis scripts arrive in weeks 2 to 4.
figures:
	@echo "No figures yet (analysis/ is empty until week 2)."
