.PHONY: smoke core full figures clean install test

PYTHON ?= python3
PIP ?= pip

install:
	$(PIP) install -e ".[all,dev]"

smoke:
	$(PYTHON) -m sleeperbench.cli run --profile smoke

core:
	@echo "Usage: make core MODEL=gpt-4o-mini"
	@test -n "$(MODEL)" || (echo "ERROR: MODEL not set" && exit 1)
	$(PYTHON) -m sleeperbench.cli run --profile core --model "$(MODEL)"

full:
	@echo "Usage: make full MODEL=gpt-4o-mini"
	@test -n "$(MODEL)" || (echo "ERROR: MODEL not set" && exit 1)
	$(PYTHON) -m sleeperbench.cli run --profile full --model "$(MODEL)"

figures:
	$(PYTHON) scripts/reproduce_paper.sh

test:
	$(PYTHON) -m pytest tests/ -v

clean:
	rm -rf results/ dist/ build/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

report:
	@test -n "$(FILE)" || (echo "ERROR: FILE not set. Usage: make report FILE=results/smoke/results.jsonl" && exit 1)
	$(PYTHON) -m sleeperbench.cli report "$(FILE)"
