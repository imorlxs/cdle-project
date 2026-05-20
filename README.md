# cdle-project

This repository now contains a reproducible starter implementation for **Project proposal #1**. It provides:

- a configurable benchmark harness for dataframe-style operations
- an ML pipeline for predicting `fare_amount`
- markdown report generation with benchmark and model summary tables
- a commented notebook that mirrors the CLI workflow

## Repository layout

- `/project_proposal_1.py` – CLI and reusable functions for data generation, benchmarking, ML, and report generation
- `/configs/project_proposal_1.example.json` – example configuration for three datasets (smaller, NYC taxi reference, larger)
- `/notebooks/project_proposal_1.ipynb` – commented notebook walkthrough
- `/tests/test_project_proposal_1.py` – focused regression tests for the new workflow
- `/requirements.txt` – validated Python dependencies for the CPU baseline workflow

## Install

```bash
python -m pip install -r requirements.txt
```

Optional backends are intentionally loaded only when present:

- `modin` (set `MODIN_ENGINE=dask` to evaluate Dask+Modin)
- `pyspark` / `pyspark.pandas` for Koalas / pandas-on-Spark style runs
- `cudf` and `dask_cudf` for RAPIDS-enabled runs

## Quick start

Generate taxi-like sample datasets for a dry run:

```bash
python project_proposal_1.py generate-synthetic-data --output data/taxi_reference.csv --rows 500
python project_proposal_1.py generate-synthetic-data --output data/taxi_small.csv --rows 200
python project_proposal_1.py generate-synthetic-data --output data/taxi_large.csv --rows 800
```

Run the full workflow from a config file:

```bash
python project_proposal_1.py full-run --config configs/project_proposal_1.example.json
```

The command writes:

- `outputs/benchmark_results.csv`
- `outputs/benchmark_summary.md`
- `outputs/ml_results.json`
- `outputs/ml_summary.md`
- `outputs/project_proposal_1_report.md`

## Mapping to the assignment

### Experiment #1

Use the `benchmark` or `full-run` commands with the NYC taxi extract and compare operation timings across `pandas`, `dask`, `joblib`, and any optional backends you install (`modin`, `koalas`, `rapids`, `dask_rapids`).

### Experiment #2

To evaluate the requested combinations:

- **Dask + Modin**: install Modin and set `MODIN_ENGINE=dask`, then include `modin` in `benchmark_backends`
- **Dask + Rapids**: install RAPIDS and include `dask_rapids`
- **Dask + Modin + Rapids**: use a RAPIDS-capable environment with Modin configured for Dask and include both `modin` and `dask_rapids` in separate benchmark runs for comparison
- **Koalas**: install `pyspark` and include `koalas`

### Prediction

The ML pipeline:

- reads the taxi dataset
- performs cleaning and feature engineering
- tunes models with cross-validation using joblib-powered parallelism
- evaluates regression (`XGBRegressor` when available) and classification (`LogisticRegression` on discretized `fare_amount`)

Reported metrics include RMSE, MAE, R², accuracy, precision, recall, and weighted F1.

## Validation

Run the focused tests with:

```bash
python -m unittest tests/test_project_proposal_1.py
```
