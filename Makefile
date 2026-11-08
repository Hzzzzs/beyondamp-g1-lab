PYTHON ?= .venv/bin/python

.PHONY: check test doctor
check:
	$(PYTHON) -m ruff check source scripts tests
	$(PYTHON) -m compileall -q source scripts

test:
	$(PYTHON) -m pytest -q

doctor:
	$(PYTHON) -m amp_lab.cli doctor
