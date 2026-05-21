#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


DEFAULT_TARGET = "fare_amount"
DEFAULT_OUTPUT_DIR = Path("outputs")
SYNTHETIC_DATA_START_DATE = datetime(2016, 1, 1, 0, 0, 0)
KNOWN_NUMERIC_COLUMNS = [
    "passenger_count",
    "trip_distance",
    "fare_amount",
    "tip_amount",
    "total_amount",
    "pickup_location_id",
    "dropoff_location_id",
]
KNOWN_CATEGORICAL_COLUMNS = ["vendor_id", "payment_type"]
KNOWN_DATETIME_COLUMNS = ["pickup_datetime", "dropoff_datetime"]


@dataclass
class BenchmarkResult:
    dataset: str
    backend: str
    operation: str
    seconds: float | None
    status: str
    details: str = ""


@dataclass
class ModelResult:
    """
    ahora los resultados de clasificación puedan 
    guardar también la matriz de confusión
    
    """
    task: str
    model: str
    best_params: dict[str, Any]
    metrics: dict[str, float]
    confusion_matrix: list[list[int]] | None = None
    notes: str = ""


def _load_module(module_name: str) -> Any:
    try:
        return __import__(module_name, fromlist=[module_name.rsplit(".", 1)[-1]])
    except Exception as exc:  # pragma: no cover - exercised by optional dependency paths
        raise RuntimeError(f"Optional dependency '{module_name}' is required: {exc}") from exc


def _load_backend_dependencies(backend: str) -> tuple[Any, Any]:
    if backend == "pandas":
        import pandas as pd

        return pd, pd
    if backend == "joblib":
        import joblib
        import numpy as np
        import pandas as pd

        return {"pd": pd, "np": np, "joblib": joblib}, pd
    if backend == "dask":
        import dask.dataframe as dd
        import pandas as pd

        return dd, pd
    if backend == "modin":
        modin_pd = _load_module("modin.pandas")
        import pandas as pd

        return modin_pd, pd
    if backend == "koalas":
        ps = _load_module("pyspark.pandas")
        import pandas as pd

        return ps, pd
    if backend == "rapids":
        cudf = _load_module("cudf")
        import pandas as pd

        return cudf, pd
    if backend == "dask_rapids":
        dask_cudf = _load_module("dask_cudf")
        import pandas as pd

        return dask_cudf, pd
    raise ValueError(f"Unsupported backend: {backend}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark dataframe backends and run the ML pipeline for Project Proposal #1."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate-synthetic-data", help="Create a taxi-like CSV dataset")
    generate.add_argument("--output", required=True, type=Path)
    generate.add_argument("--rows", type=int, default=2000)
    generate.add_argument("--seed", type=int, default=42)

    benchmark = subparsers.add_parser("benchmark", help="Run database-style benchmarks")
    benchmark.add_argument("--dataset", required=True, type=Path)
    benchmark.add_argument("--dataset-label", default="nyc_taxi")
    benchmark.add_argument(
        "--backends",
        nargs="+",
        default=["pandas", "joblib", "dask"],
        help="Backends to benchmark (pandas, joblib, dask, modin, koalas, rapids, dask_rapids)",
    )
    benchmark.add_argument("--output", required=True, type=Path)
    benchmark.add_argument("--summary", type=Path)

    ml = subparsers.add_parser("ml", help="Run the fare prediction ML pipeline")
    ml.add_argument("--dataset", required=True, type=Path)
    ml.add_argument("--output", required=True, type=Path)
    ml.add_argument("--summary", type=Path)

    run_all = subparsers.add_parser("full-run", help="Run benchmarks, ML, and generate a report from a JSON config")
    run_all.add_argument("--config", required=True, type=Path)

    report = subparsers.add_parser("report", help="Generate a markdown report from previously generated outputs")
    report.add_argument("--config", required=True, type=Path)

    return parser


def generate_synthetic_taxi_data(output: Path, rows: int = 2000, seed: int = 42) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    start = SYNTHETIC_DATA_START_DATE
    payment_types = ["cash", "card", "voucher", "unknown"]
    vendors = ["VTS", "CMT", "DDS"]
    fieldnames = [
        "pickup_datetime",
        "dropoff_datetime",
        "vendor_id",
        "passenger_count",
        "trip_distance",
        "pickup_location_id",
        "dropoff_location_id",
        "payment_type",
        DEFAULT_TARGET,
        "tip_amount",
        "total_amount",
    ]

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for _ in range(rows):
            pickup = start + timedelta(minutes=rng.randint(0, 60 * 24 * 150))
            trip_distance = max(0.4, round(rng.gammavariate(2.2, 2.8), 2))
            passenger_count = rng.randint(1, 6)
            duration_minutes = max(5, int(rng.gauss(12 + trip_distance * 2.3, 5)))
            dropoff = pickup + timedelta(minutes=duration_minutes)
            fare_amount = round(max(3.5, 2.5 + (trip_distance * 2.6) + (duration_minutes * 0.45) + rng.uniform(-3, 3)), 2)
            tip_amount = round(max(0.0, fare_amount * rng.uniform(0.0, 0.25)), 2)
            tolls = round(rng.uniform(0.0, 6.0), 2)
            writer.writerow(
                {
                    "pickup_datetime": pickup.isoformat(sep=" "),
                    "dropoff_datetime": dropoff.isoformat(sep=" "),
                    "vendor_id": rng.choice(vendors),
                    "passenger_count": passenger_count,
                    "trip_distance": trip_distance,
                    "pickup_location_id": rng.randint(1, 263),
                    "dropoff_location_id": rng.randint(1, 263),
                    "payment_type": rng.choice(payment_types),
                    DEFAULT_TARGET: fare_amount,
                    "tip_amount": tip_amount,
                    "total_amount": round(fare_amount + tip_amount + tolls, 2),
                }
            )
    return output


def _chunk_dataframe(df: Any, parts: int) -> list[Any]:
    import numpy as np

    indices = np.array_split(range(len(df)), max(1, min(parts, len(df))))
    return [df.iloc[list(index_chunk)].copy() for index_chunk in indices if len(index_chunk)]


def _load_dataframe(path: Path, backend: str) -> Any:
    module, _ = _load_backend_dependencies(backend)
    if backend == "joblib":
        return module["pd"].read_csv(path)
    if backend in {"pandas", "modin", "koalas"}:
        return module.read_csv(path)
    if backend == "dask":
        return module.read_csv(path, assume_missing=True, blocksize="16MB")
    if backend == "rapids":
        return module.read_csv(path)
    if backend == "dask_rapids":
        return module.read_csv(str(path), blocksize="16MB")
    raise ValueError(f"Unsupported backend: {backend}")


def _to_datetime(series: Any, backend: str) -> Any:
    module, pandas_module = _load_backend_dependencies(backend)
    if backend == "joblib":
        return module["pd"].to_datetime(series, errors="coerce")
    if backend in {"pandas", "modin"}:
        return module.to_datetime(series, errors="coerce")
    if backend == "dask":
        return module.to_datetime(series, errors="coerce")
    if backend == "koalas":
        return pandas_module.to_datetime(series.to_pandas(), errors="coerce")
    if backend == "rapids":
        return module.to_datetime(series)
    if backend == "dask_rapids":
        return series.astype("datetime64[ns]")
    raise ValueError(f"Unsupported backend: {backend}")


def _materialize(obj: Any, backend: str) -> Any:
    if backend in {"pandas", "joblib"}:
        return obj
    if backend == "dask":
        return obj.compute() if hasattr(obj, "compute") else obj
    if backend == "modin":
        return obj._to_pandas() if hasattr(obj, "_to_pandas") else obj
    if backend == "koalas":
        return obj.to_pandas() if hasattr(obj, "to_pandas") else obj
    if backend == "rapids":
        return obj.to_pandas() if hasattr(obj, "to_pandas") else obj
    if backend == "dask_rapids":
        computed = obj.compute() if hasattr(obj, "compute") else obj
        return computed.to_pandas() if hasattr(computed, "to_pandas") else computed
    raise ValueError(f"Unsupported backend: {backend}")


def _prepare_dataframe(df: Any, backend: str) -> Any:
    if backend == "joblib":
        pd = _load_backend_dependencies("joblib")[0]["pd"]
        prepared = df.copy()
        for column in KNOWN_NUMERIC_COLUMNS:
            if column in prepared.columns:
                prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
        for column in KNOWN_DATETIME_COLUMNS:
            if column in prepared.columns:
                prepared[column] = pd.to_datetime(prepared[column], errors="coerce")
        return prepared

    prepared = df.copy()
    if backend in {"pandas", "modin"}:
        module, _ = _load_backend_dependencies(backend)
        for column in KNOWN_NUMERIC_COLUMNS:
            if column in prepared.columns:
                prepared[column] = module.to_numeric(prepared[column], errors="coerce")
        for column in KNOWN_DATETIME_COLUMNS:
            if column in prepared.columns:
                prepared[column] = module.to_datetime(prepared[column], errors="coerce")
        return prepared

    if backend == "dask":
        module, _ = _load_backend_dependencies(backend)
        for column in KNOWN_NUMERIC_COLUMNS:
            if column in prepared.columns:
                prepared[column] = prepared[column].astype("float64")
        for column in KNOWN_DATETIME_COLUMNS:
            if column in prepared.columns:
                prepared[column] = module.to_datetime(prepared[column], errors="coerce")
        return prepared

    if backend == "koalas":
        pandas_module = _load_backend_dependencies("koalas")[1]
        pdf = prepared.to_pandas()
        for column in KNOWN_NUMERIC_COLUMNS:
            if column in pdf.columns:
                pdf[column] = pandas_module.to_numeric(pdf[column], errors="coerce")
        for column in KNOWN_DATETIME_COLUMNS:
            if column in pdf.columns:
                pdf[column] = pandas_module.to_datetime(pdf[column], errors="coerce")
        module, _ = _load_backend_dependencies("koalas")
        return module.from_pandas(pdf)

    if backend in {"rapids", "dask_rapids"}:
        for column in KNOWN_NUMERIC_COLUMNS:
            if column in prepared.columns:
                prepared[column] = prepared[column].astype("float64")
        return prepared

    raise ValueError(f"Unsupported backend: {backend}")


def _joblib_parallel_transform(df: Any, transform: Callable[[Any], Any]) -> Any:
    deps, _ = _load_backend_dependencies("joblib")
    pd = deps["pd"]
    joblib = deps["joblib"]
    parts = _chunk_dataframe(df, os.cpu_count() or 2)
    frames = joblib.Parallel(n_jobs=-1)(joblib.delayed(transform)(part) for part in parts)
    return pd.concat(frames, ignore_index=True)


def _run_benchmark_operations(df: Any, backend: str) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []

    def time_operation(name: str, func: Callable[[], Any]) -> None:
        started = time.perf_counter()
        materialized = func()
        if materialized is not None:
            _materialize(materialized, backend)
        elapsed = time.perf_counter() - started
        results.append(BenchmarkResult(dataset="", backend=backend, operation=name, seconds=round(elapsed, 6), status="ok"))

    if backend == "joblib":
        pd = _load_backend_dependencies("joblib")[0]["pd"]

        def with_features(frame: Any) -> Any:
            transformed = frame.copy()
            transformed = transformed[transformed[DEFAULT_TARGET] > 0].copy()
            transformed["pickup_hour"] = transformed["pickup_datetime"].dt.hour
            transformed["trip_duration_minutes"] = (
                transformed["dropoff_datetime"] - transformed["pickup_datetime"]
            ).dt.total_seconds().div(60).fillna(0)
            transformed["trip_speed_mph"] = transformed["trip_distance"].div(
                transformed["trip_duration_minutes"].replace(0, 1).div(60)
            )
            return transformed

        time_operation("filter_rows", lambda: _joblib_parallel_transform(df, lambda part: part[part[DEFAULT_TARGET] > 0]))
        featured = _joblib_parallel_transform(df, with_features)
        time_operation("feature_engineering", lambda: featured)
        time_operation(
            "groupby_aggregate",
            lambda: featured.groupby("payment_type", dropna=False)[[DEFAULT_TARGET, "trip_distance"]].mean().reset_index(),
        )
        lookup = pd.DataFrame(
            {
                "payment_type": ["cash", "card", "voucher", "unknown"],
                "payment_category": ["cash", "card", "prepaid", "other"],
            }
        )
        time_operation("join_lookup", lambda: featured.merge(lookup, on="payment_type", how="left"))
        time_operation("sort_values", lambda: featured.sort_values(by=["total_amount", DEFAULT_TARGET], ascending=False))
        return results

    time_operation("filter_rows", lambda: df[df[DEFAULT_TARGET] > 0])

    def build_features() -> Any:
        featured = df[df[DEFAULT_TARGET] > 0].copy()
        if backend == "koalas":
            pdf = featured.to_pandas()
            pdf["pickup_hour"] = pdf["pickup_datetime"].dt.hour
            pdf["trip_duration_minutes"] = (pdf["dropoff_datetime"] - pdf["pickup_datetime"]).dt.total_seconds().div(60).fillna(0)
            pdf["trip_speed_mph"] = pdf["trip_distance"].div(pdf["trip_duration_minutes"].replace(0, 1).div(60))
            module, _ = _load_backend_dependencies("koalas")
            return module.from_pandas(pdf)
        featured["pickup_hour"] = featured["pickup_datetime"].dt.hour
        featured["trip_duration_minutes"] = (
            featured["dropoff_datetime"] - featured["pickup_datetime"]
        ).dt.total_seconds().div(60).fillna(0)
        featured["trip_speed_mph"] = featured["trip_distance"].div(featured["trip_duration_minutes"].replace(0, 1).div(60))
        return featured

    featured = build_features()
    time_operation("feature_engineering", lambda: featured)
    time_operation(
        "groupby_aggregate",
        lambda: featured.groupby("payment_type", dropna=False)[[DEFAULT_TARGET, "trip_distance"]].mean().reset_index(),
    )

    if backend in {"rapids", "dask_rapids"}:
        module, _ = _load_backend_dependencies("rapids" if backend == "rapids" else "dask_rapids")
        lookup = module.DataFrame(
            {
                "payment_type": ["cash", "card", "voucher", "unknown"],
                "payment_category": ["cash", "card", "prepaid", "other"],
            }
        )
    else:
        pandas_module = _load_backend_dependencies(backend)[1]
        lookup = pandas_module.DataFrame(
            {
                "payment_type": ["cash", "card", "voucher", "unknown"],
                "payment_category": ["cash", "card", "prepaid", "other"],
            }
        )
        if backend == "koalas":
            module, _ = _load_backend_dependencies("koalas")
            lookup = module.from_pandas(lookup)
    time_operation("join_lookup", lambda: featured.merge(lookup, on="payment_type", how="left"))
    time_operation("sort_values", lambda: featured.sort_values(by=["total_amount", DEFAULT_TARGET], ascending=False))
    return results


def benchmark_dataset(dataset: Path, dataset_label: str, backends: Sequence[str]) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []
    for backend in backends:
        try:
            started = time.perf_counter()
            df = _prepare_dataframe(_load_dataframe(dataset, backend), backend)
            _materialize(df.head() if hasattr(df, "head") else df, backend)
            read_time = round(time.perf_counter() - started, 6)
            results.append(BenchmarkResult(dataset_label, backend, "read_data", read_time, "ok"))
            backend_results = _run_benchmark_operations(df, backend)
            for row in backend_results:
                row.dataset = dataset_label
            results.extend(backend_results)
        except Exception as exc:
            results.append(
                BenchmarkResult(
                    dataset=dataset_label,
                    backend=backend,
                    operation="read_data",
                    seconds=None,
                    status="skipped",
                    details=str(exc),
                )
            )
    return results


def _load_pandas_dataset(path: Path) -> Any:
    import pandas as pd

    df = pd.read_csv(path)
    for column in KNOWN_NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    for column in KNOWN_DATETIME_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")
    return df


def prepare_modeling_frame(df: Any, target_column: str = DEFAULT_TARGET) -> tuple[Any, Any, list[str], list[str]]:
    """Prepare a pandas modeling frame and return features, target, and inferred column groups.

    The function drops rows with missing target values, engineers pickup calendar fields and trip duration,
    excludes raw datetime columns from the feature matrix, and returns `(X, y, categorical_columns, numeric_columns)`.
    It assumes a taxi-like tabular input where `pickup_datetime`, `dropoff_datetime`, `vendor_id`, and
    `payment_type` may be present.
    """
    import pandas as pd
    from pandas.api.types import is_bool_dtype, is_object_dtype, is_string_dtype

    prepared = df.copy()
    prepared = prepared.dropna(subset=[target_column]).reset_index(drop=True)
    if "pickup_datetime" in prepared.columns:
        prepared["pickup_hour"] = prepared["pickup_datetime"].dt.hour
        prepared["pickup_dayofweek"] = prepared["pickup_datetime"].dt.dayofweek
        prepared["pickup_month"] = prepared["pickup_datetime"].dt.month
    if {"pickup_datetime", "dropoff_datetime"}.issubset(prepared.columns):
        prepared["trip_duration_minutes"] = (
            prepared["dropoff_datetime"] - prepared["pickup_datetime"]
        ).dt.total_seconds().div(60)
    for column in KNOWN_CATEGORICAL_COLUMNS:
        if column in prepared.columns:
            prepared[column] = prepared[column].fillna("unknown").astype(str)
    feature_columns = [column for column in prepared.columns if column != target_column and not column.endswith("datetime")]
    X = prepared[feature_columns].copy()
    y = prepared[target_column].astype(float)
    categorical_columns = [
        column
        for column in feature_columns
        if is_object_dtype(X[column])
        or is_string_dtype(X[column])
        or isinstance(X[column].dtype, pd.CategoricalDtype)
        or is_bool_dtype(X[column])
    ]
    numeric_columns = [column for column in feature_columns if column not in categorical_columns]
    X = X.replace([math.inf, -math.inf], float("nan"))
    if not len(X):
        raise ValueError("No rows available after preprocessing")
    return X, y, categorical_columns, numeric_columns


def _regression_search(random_state: int = 42) -> tuple[str, Any, dict[str, list[Any]], str]:
    """Build the regression model selection tuple `(model_name, estimator, grid, notes)`.

    `XGBRegressor` is preferred for the fare regression task. When xgboost is unavailable, the function
    falls back to `HistGradientBoostingRegressor` and returns a note explaining that fallback.
    """
    try:
        from xgboost import XGBRegressor

        estimator = XGBRegressor(
            objective="reg:squarederror",
            random_state=random_state,
            tree_method="hist",
            eval_metric="rmse",
            n_estimators=120,
        )
        grid = {"model__max_depth": [4, 6], "model__learning_rate": [0.05, 0.1]}
        return "XGBRegressor", estimator, grid, ""
    except Exception as exc:
        from sklearn.ensemble import HistGradientBoostingRegressor

        estimator = HistGradientBoostingRegressor(random_state=random_state)
        grid = {"model__max_depth": [4, 8], "model__learning_rate": [0.05, 0.1]}
        return "HistGradientBoostingRegressor", estimator, grid, f"xgboost unavailable, fallback used: {exc}"


def run_ml_pipeline(dataset: Path) -> list[ModelResult]:
    """Run the end-to-end fare prediction workflow for a taxi CSV dataset.

    The pipeline loads the dataset, performs preprocessing and feature engineering, then executes two
    cross-validated tasks: fare regression and discretized-fare classification. It returns a list of
    `ModelResult` records describing the selected model, best parameters, evaluation metrics, execution
    times, and the confusion matrix for the classification task.
    """
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,       # La matriz de confusión permite ver en qué clases falla el modelo
        f1_score,
        mean_absolute_error,
        mean_squared_error,
        precision_score,
        r2_score,
        recall_score,
    )
    from sklearn.model_selection import GridSearchCV, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    df = _load_pandas_dataset(dataset)
    X, y, categorical_columns, numeric_columns = prepare_modeling_frame(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_columns),
            ("cat", categorical_transformer, categorical_columns),
        ]
    )

    regression_name, regression_estimator, regression_grid, regression_note = _regression_search()

    regression_pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", regression_estimator),
        ]
    )

    regression_search = GridSearchCV(
        regression_pipeline,
        regression_grid,
        cv=3,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )

    """
    Entrenamos y medimos el tiempo 
    
    """
    regression_train_start = time.perf_counter()
    regression_search.fit(X_train, y_train)
    regression_training_time = time.perf_counter() - regression_train_start

    """ 
    Tiempos de prediccion 
    
    """
    regression_pred_start = time.perf_counter()
    regression_predictions = regression_search.predict(X_test)
    regression_prediction_time = time.perf_counter() - regression_pred_start

    rmse = math.sqrt(float(mean_squared_error(y_test, regression_predictions)))

    discretized_fare_bins = pd.qcut(
        y,
        q=min(4, y.nunique()),
        labels=False,
        duplicates="drop",
    )

    """ 
    Añadimos porporcionalidad entre 
    test/train usando stratify
    
    """
    X_class_train, X_class_test, y_class_train, y_class_test = train_test_split(
        X,
        discretized_fare_bins,
        test_size=0.2,
        random_state=42,
        stratify=discretized_fare_bins,
    )

    classification_preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
            (
                "cat",
                categorical_transformer,
                categorical_columns,
            ),
        ]
    )

    classification_pipeline = Pipeline(
        steps=[
            ("preprocess", classification_preprocessor),
            ("model", LogisticRegression(max_iter=2000)),
        ]
    )

    classification_search = GridSearchCV(
        classification_pipeline,
        {"model__C": [0.5, 1.0, 2.0]},
        cv=3,
        scoring="f1_weighted",
        n_jobs=-1,
    )
    """ 
    Medimos tiempos para entrenamietno
    de cladificacion
    
    """
    classification_train_start = time.perf_counter()
    classification_search.fit(X_class_train, y_class_train)
    classification_training_time = time.perf_counter() - classification_train_start

    """ 
    Medimos tiempos para prediccion
    de clasificacion
    
    """
    classification_pred_start = time.perf_counter()
    classification_predictions = classification_search.predict(X_class_test)
    classification_prediction_time = time.perf_counter() - classification_pred_start

    """ 
    Añadimos matriz de confusion
    para clasificacion
    
    """
    classification_confusion_matrix = confusion_matrix(
        y_class_test,
        classification_predictions,
    ).tolist()

    return [
        ModelResult(
            task="regression",
            model=regression_name,
            best_params=regression_search.best_params_,
            metrics={
                "rmse": round(rmse, 6),
                "mae": round(float(mean_absolute_error(y_test, regression_predictions)), 6),
                "r2": round(float(r2_score(y_test, regression_predictions)), 6),
                "training_time": round(regression_training_time, 6),
                "prediction_time": round(regression_prediction_time, 6),
            },
            confusion_matrix=None,
            notes=regression_note or "Cross-validation used GridSearchCV with joblib parallelism.",
        ),
        ModelResult(
            task="classification",
            model="LogisticRegression",
            best_params=classification_search.best_params_,
            metrics={
                "accuracy": round(float(accuracy_score(y_class_test, classification_predictions)), 6),
                "precision_weighted": round(
                    float(precision_score(y_class_test, classification_predictions, average="weighted", zero_division=0)),
                    6,
                ),
                "recall_weighted": round(
                    float(recall_score(y_class_test, classification_predictions, average="weighted", zero_division=0)),
                    6,
                ),
                "f1_weighted": round(
                    float(f1_score(y_class_test, classification_predictions, average="weighted", zero_division=0)),
                    6,
                ),
                "precision_macro": round(
                    float(precision_score(y_class_test, classification_predictions, average="macro", zero_division=0)),
                    6,
                ),
                "recall_macro": round(
                    float(recall_score(y_class_test, classification_predictions, average="macro", zero_division=0)),
                    6,
                ),
                "f1_macro": round(
                    float(f1_score(y_class_test, classification_predictions, average="macro", zero_division=0)),
                    6,
                ),
                "training_time": round(classification_training_time, 6),
                "prediction_time": round(classification_prediction_time, 6),
            },
            confusion_matrix=classification_confusion_matrix,
            notes=(
                "Target discretized with pandas.qcut into quantile-based fare classes before "
                "cross-validated LogisticRegression."
            ),
        ),
    ]

def write_benchmark_outputs(results: Sequence[BenchmarkResult], csv_path: Path, summary_path: Path | None = None) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "backend", "operation", "seconds", "status", "details"])
        writer.writeheader()
        for row in results:
            writer.writerow(asdict(row))
    if summary_path:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(render_benchmark_summary(results), encoding="utf-8")


def write_ml_outputs(results: Sequence[ModelResult], json_path: Path, summary_path: Path | None = None) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps([asdict(row) for row in results], indent=2), encoding="utf-8")
    if summary_path:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(render_ml_summary(results), encoding="utf-8")


def render_benchmark_summary(results: Sequence[BenchmarkResult]) -> str:
    grouped: dict[tuple[str, str], dict[str, str]] = {}
    backends = sorted({row.backend for row in results})
    for row in results:
        key = (row.dataset, row.operation)
        grouped.setdefault(key, {})
        grouped[key][row.backend] = (
            f"{row.seconds:.4f}s" if row.seconds is not None and row.status == "ok" else f"{row.status}: {row.details}"
        )
    header = ["Dataset", "Operation", *backends]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for (dataset, operation), values in sorted(grouped.items()):
        lines.append(
            "| "
            + " | ".join([dataset, operation, *[values.get(backend, "n/a") for backend in backends]])
            + " |"
        )
    return "\n".join(lines) + "\n"


def render_ml_summary(results: Sequence[ModelResult]) -> str:
    """ 
    ml_summary mas limpio con una 
    metrica por fila

    """
    lines = ["| Task | Model | Metric | Value | Notes |", "| --- | --- | --- | --- | --- |"]
    for row in results:
        for metric_name, metric_value in row.metrics.items():
            lines.append(
                f"| {row.task} | {row.model} | {metric_name} | {metric_value} | {row.notes} |"
            )

        if row.confusion_matrix is not None:
            lines.append(
                f"| {row.task} | {row.model} | confusion_matrix | {row.confusion_matrix} | Stored as nested class-count matrix. |"
            )

    return "\n".join(lines) + "\n"

def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def machine_snapshot() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "processor": platform.processor() or "unknown",
    }


def generate_report(config: dict[str, Any]) -> Path:
    output_dir = Path(config.get("output_dir", DEFAULT_OUTPUT_DIR))
    benchmark_summary = output_dir / "benchmark_summary.md"
    ml_summary = output_dir / "ml_summary.md"
    report_path = output_dir / "project_proposal_1_report.md"
    datasets = config.get("benchmark_datasets", [])
    machine = machine_snapshot()
    dataset_lines = [f"- **{item['name']}**: `{item['path']}`" for item in datasets] or [
        "- Configure datasets in `configs/project_proposal_1.example.json`."
    ]
    benchmark_text = benchmark_summary.read_text(encoding="utf-8").strip() if benchmark_summary.exists() else (
        "Run the benchmark command to populate this section."
    )
    ml_text = ml_summary.read_text(encoding="utf-8").strip() if ml_summary.exists() else (
        "Run the ML command to populate this section."
    )
    report_text = "\n".join(
        [
            "# Project proposal #1 report",
            "",
            "## Brief background on PySpark, Dask, Modin, Joblib, RAPIDS and Koalas",
            "- **PySpark / Koalas**: pandas-like APIs on top of Spark, suitable for cluster-scale distributed execution.",
            "- **Dask**: task-graph based parallel dataframe library for scaling pandas-style work across cores or clusters.",
            "- **Modin**: drop-in pandas acceleration layer that can target engines such as Dask.",
            "- **Joblib**: Python parallelism helper used here to parallelize cross-validation and a pandas-based benchmark baseline.",
            "- **RAPIDS**: GPU dataframe and ML stack (for example cudf / dask_cudf) for CUDA-enabled environments.",
            "",
            "## Materials and methods",
            "### Machines used and their characteristics",
            f"- Python: {machine['python']}",
            f"- Platform: {machine['platform']}",
            f"- CPU count: {machine['cpu_count']}",
            f"- Processor: {machine['processor']}",
            "",
            "### Datasets description",
            *dataset_lines,
            "",
            "## Experiment #1: repeat NYC taxi driver dataset study",
            benchmark_text,
            "",
            "## Experiment #2",
            "The benchmark harness is prepared to evaluate additional backends such as `modin`, `koalas`, `rapids`, and `dask_rapids` when the corresponding execution environments are available. "
            "`MODIN_ENGINE=dask` can be used to evaluate a Dask-backed Modin workflow. `dask_rapids` requires a CUDA-enabled RAPIDS environment. "
            "Results from these optional backends should be integrated into the same benchmark summary table to allow direct comparison against the CPU baselines.",
            "",
            "## Prediction",
            "### Methodology",
            "- The prediction workflow uses `fare_amount` as the target variable.",
            "- The pipeline cleans known taxi columns, engineers pickup calendar features and trip duration, and separates numerical and categorical predictors.",
            "- The regression task predicts the continuous fare amount and prefers `XGBRegressor` when available.",
            "- The classification task discretizes `fare_amount` into quantile-based fare classes using `pandas.qcut` and trains `LogisticRegression`.",
            "- Both tasks use `GridSearchCV` with 3-fold cross-validation and joblib-powered parallel execution.",
            "- Reported metrics include predictive quality metrics and execution-time measurements for training and prediction.",
            "",
            "### Results and Analysis",
            ml_text,
            "",
            "## Discussion and conclusions",
            "The benchmark results should be interpreted by considering both execution time and framework overhead. "
            "For smaller datasets, local pandas-based execution can remain competitive because distributed frameworks introduce scheduling and partition-management overhead. "
            "As dataset size increases, Dask and other distributed backends are expected to become more useful because they can split operations across partitions or workers. "
            "The ML results should be analyzed jointly in terms of predictive performance and computational cost: `XGBRegressor` is expected to provide a stronger nonlinear regression baseline, while `LogisticRegression` provides a simpler classification baseline after fare discretization. "
            "Optional GPU or Spark-based backends should only be discussed as executed results when the corresponding environment was actually available.",
        ]
    ) + "\n"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")
    return report_path


def run_full_config(config_path: Path) -> Path:
    config = load_json(config_path)
    output_dir = Path(config.get("output_dir", DEFAULT_OUTPUT_DIR))
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_results: list[BenchmarkResult] = []
    for dataset_entry in config.get("benchmark_datasets", []):
        benchmark_results.extend(
            benchmark_dataset(
                dataset=Path(dataset_entry["path"]),
                dataset_label=dataset_entry["name"],
                backends=config.get("benchmark_backends", ["pandas", "joblib", "dask"]),
            )
        )
    write_benchmark_outputs(
        benchmark_results,
        output_dir / "benchmark_results.csv",
        output_dir / "benchmark_summary.md",
    )

    ml_config = config.get("ml_dataset")
    if ml_config:
        ml_results = run_ml_pipeline(Path(ml_config["path"]))
        write_ml_outputs(ml_results, output_dir / "ml_results.json", output_dir / "ml_summary.md")

    return generate_report(config)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "generate-synthetic-data":
        output = generate_synthetic_taxi_data(args.output, rows=args.rows, seed=args.seed)
        print(output)
        return 0
    if args.command == "benchmark":
        results = benchmark_dataset(args.dataset, args.dataset_label, args.backends)
        write_benchmark_outputs(results, args.output, args.summary)
        print(args.output)
        return 0
    if args.command == "ml":
        results = run_ml_pipeline(args.dataset)
        write_ml_outputs(results, args.output, args.summary)
        print(args.output)
        return 0
    if args.command == "full-run":
        report_path = run_full_config(args.config)
        print(report_path)
        return 0
    if args.command == "report":
        config = load_json(args.config)
        report_path = generate_report(config)
        print(report_path)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
