# CDLE Project — Big Data ML Pipeline

Group project replicating and extending the benchmark from
[*Benchmark: Koalas (PySpark) and Dask*](https://www.databricks.com/blog/2021/04/27/benchmark-koalas-pyspark-and-dask.html).

---

## Isaac's scope

- Dataset preparation (NYC Taxi + 2 additional datasets)
- Dask benchmark implementation (all 15 operations × 3 scenarios)
- Koalas / pyspark.pandas benchmark implementation
- Syntactic differences documentation
- Initial notebook structure

---

## Repository layout

```
cdle-project/
├── config.py                  # paths, column names, filter params
├── requirements.txt           # Dask + ML dependencies
├── requirements-pyspark.txt   # adds PySpark + GCS
├── src/
│   ├── benchmark_utils.py     # BenchmarkTimer, results table
│   ├── operations_dask.py     # 15 operations — Dask
│   └── operations_koalas.py   # 15 operations — Koalas/pyspark.pandas
├── scripts/
│   └── download_data.py       # download TLC CSVs, normalise, save Parquet
├── notebooks/
│   ├── 00_dataset_preparation.ipynb   # download, EDA, Parquet conversion
│   ├── 01_benchmark_dask.ipynb        # Dask benchmark (3 scenarios)
│   ├── 02_benchmark_koalas.ipynb      # Koalas benchmark (3 scenarios)
│   └── 03_syntax_comparison.ipynb     # syntax table + combined results
├── gcp/
│   ├── setup_single_node.sh       # single VM (local execution)
│   ├── setup_dataproc_cluster.sh  # Dataproc cluster (distributed execution)
│   └── install_deps.sh            # install deps on a GCP VM
└── results/                       # output CSVs + PNGs (git-ignored)
```

---

## Quick start (local)

```bash
# 1. Install dependencies
python -m venv venv && source venv/bin/activate
pip install -r requirements-pyspark.txt

# 2. Download data (start with January 2009)
python scripts/download_data.py --year 2009 --months 1

# 3. Run notebooks
jupyter notebook notebooks/
```

## Quick start (GCP)

```bash
# One-time GCP setup
bash gcp/setup_single_node.sh          # creates VM + GCS bucket
# SSH into VM, then:
bash gcp/install_deps.sh               # clone repo + install deps

# For distributed execution:
bash gcp/setup_dataproc_cluster.sh     # creates Dataproc cluster
```

---

## Datasets

| Name | Rows (approx) | File | Role |
|---|---|---|---|
| California Housing | 20,640 | `california_housing.parquet` | Smaller dataset |
| NYC Yellow Taxi Jan 2009 | ~14 M | `yellow_taxi.parquet` | Primary benchmark |
| NYC Yellow Taxi 2009 full | ~170 M | `yellow_taxi_2009_full.parquet` | Larger dataset |

---

## Benchmark operations (from blog)

15 operations, 3 scenarios each (standard / filtered / filtered+cached):
`complex_arithmetic`, `count`, `count_index`, `groupby_stats`, `join`,
`join_count`, `mean`, `mean_complex_arithmetic`, `mean_series_add`,
`mean_series_mul`, `read_parquet`, `series_add`, `series_mul`, `std`,
`value_counts`
