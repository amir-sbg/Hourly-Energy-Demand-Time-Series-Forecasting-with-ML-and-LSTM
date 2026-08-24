.PHONY: test run clean

test:
	python -m pytest -q

run:
	python -m ts_forecasting.pipeline

clean:
	rm -rf artifacts reports data/processed .pytest_cache
