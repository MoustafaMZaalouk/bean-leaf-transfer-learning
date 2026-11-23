.PHONY: train eval test lint smoke clean-cache
train:
	PYTHONPATH=src python scripts/run_experiment.py

eval:
	PYTHONPATH=src python scripts/evaluate.py

test:
	python -m pytest -q

lint:
	python -m ruff check .
	python -m ruff format --check .

smoke:
	PYTHONPATH=src python scripts/run_experiment.py --smoke --output .smoke-reports
	PYTHONPATH=src python scripts/evaluate.py --reports .smoke-reports

clean-cache:
	rm -f .cache/features-*.npz
