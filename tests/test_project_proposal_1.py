from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import project_proposal_1 as proposal


class ProjectProposalTests(unittest.TestCase):
    def test_generate_synthetic_data_writes_expected_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "sample.csv"
            proposal.generate_synthetic_taxi_data(output, rows=25, seed=7)
            with output.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
            self.assertEqual(len(rows), 25)
            self.assertIn("fare_amount", rows[0])
            self.assertIn("pickup_datetime", rows[0])

    def test_benchmark_and_ml_commands_produce_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = root / "dataset.csv"
            proposal.generate_synthetic_taxi_data(dataset, rows=180, seed=11)

            benchmark_results = proposal.benchmark_dataset(dataset, "tiny", ["pandas", "joblib"])
            self.assertTrue(any(row.operation == "read_data" and row.status == "ok" for row in benchmark_results))

            benchmark_csv = root / "benchmark.csv"
            benchmark_md = root / "benchmark.md"
            proposal.write_benchmark_outputs(benchmark_results, benchmark_csv, benchmark_md)
            self.assertTrue(benchmark_csv.exists())
            self.assertIn("pandas", benchmark_md.read_text(encoding="utf-8"))

            ml_results = proposal.run_ml_pipeline(dataset)
            self.assertEqual({row.task for row in ml_results}, {"regression", "classification"})

            ml_json = root / "ml.json"
            ml_md = root / "ml.md"
            proposal.write_ml_outputs(ml_results, ml_json, ml_md)
            payload = json.loads(ml_json.read_text(encoding="utf-8"))
            self.assertEqual(len(payload), 2)
            self.assertIn("accuracy", payload[1]["metrics"])

    def test_full_run_creates_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_dir = root / "data"
            data_dir.mkdir()
            for name, rows in [("taxi_small.csv", 120), ("taxi_reference.csv", 240), ("taxi_large.csv", 360)]:
                proposal.generate_synthetic_taxi_data(data_dir / name, rows=rows, seed=rows)

            config = {
                "output_dir": str(root / "outputs"),
                "benchmark_backends": ["pandas", "joblib"],
                "benchmark_datasets": [
                    {"name": "small", "path": str(data_dir / "taxi_small.csv")},
                    {"name": "reference", "path": str(data_dir / "taxi_reference.csv")}
                ],
                "ml_dataset": {"name": "reference", "path": str(data_dir / "taxi_reference.csv")}
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            report_path = proposal.run_full_config(config_path)
            self.assertTrue(report_path.exists())
            report = report_path.read_text(encoding="utf-8")
            self.assertIn("Project proposal #1 report", report)
            self.assertIn("Experiment #1", report)


if __name__ == "__main__":
    unittest.main()
