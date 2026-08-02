# The real logic lives in tasks.py so that Windows, WSL2, Linux, and macOS all
# run the same code. This file is a dispatcher and nothing more.
#
#   make test LAB=1 NPROC=4
#
# is exactly
#
#   python tasks.py test LAB=1 NPROC=4

PY ?= python

# Forward supported variables as one quoted KEY=VALUE argument each. Quoting is
# important for expressions such as K="bucket and not slow".
ARGS = $(foreach v,LAB NPROC K SCRIPT VERBOSE FAST YES WRITE PROFILE,$(if $(strip $($(v))),"$(v)=$($(v))"))

.PHONY: help setup setup-cpu setup-gpu doctor test test-tier2 test-answers bench reconcile \
        gantt kill disk verify-clean clean nuke nuke-venv lint format freeze-check

help:
	@$(PY) tasks.py help

setup:
	@$(PY) tasks.py setup $(ARGS)

setup-cpu:
	@$(PY) tasks.py setup-cpu $(ARGS)

setup-gpu:
	@$(PY) tasks.py setup-gpu $(ARGS)

doctor:
	@$(PY) tasks.py doctor $(ARGS)

test:
	@$(PY) tasks.py test $(ARGS)

test-tier2:
	@$(PY) tasks.py test-tier2 $(ARGS)

test-answers:
	@$(PY) tasks.py test-answers $(ARGS)

bench:
	@$(PY) tasks.py bench $(ARGS)

reconcile:
	@$(PY) tasks.py reconcile $(ARGS)

gantt:
	@$(PY) tasks.py gantt $(ARGS)

kill:
	@$(PY) tasks.py kill $(ARGS)

disk:
	@$(PY) tasks.py disk $(ARGS)

verify-clean:
	@$(PY) tasks.py verify-clean $(ARGS)

clean:
	@$(PY) tasks.py clean $(ARGS)

nuke:
	@$(PY) tasks.py nuke $(ARGS)

nuke-venv:
	@$(PY) tasks.py nuke-venv $(ARGS)

lint:
	@$(PY) tasks.py lint $(ARGS)

format:
	@$(PY) tasks.py format $(ARGS)

freeze-check:
	@$(PY) tasks.py freeze-check $(ARGS)
