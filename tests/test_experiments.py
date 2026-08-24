"""Tests for fair experiment execution and result processing."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from config import NetworkConfig
from experiments import ExperimentRunner, ResultProcessor


class ExperimentInfrastructureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = NetworkConfig(
            num_edge_nodes=2,
            num_iot_devices=3,
            num_rounds=1,
            task_generation_probability=1.0,
            queue_capacity=5,
            processing_capacity=1,
            num_experiment_runs=1,
            scale_device_counts=(3, 4),
        )

    def test_all_schedulers_receive_the_same_workload(self) -> None:
        records = ExperimentRunner(self.config).run("normal")
        workloads = [record["task_workload"] for record in records]
        self.assertEqual(len(records), 3)
        self.assertEqual(workloads[0], workloads[1])
        self.assertEqual(workloads[1], workloads[2])

    def test_runs_are_isolated_and_scale_is_applied(self) -> None:
        runner = ExperimentRunner(self.config)
        records = runner.run_scalability(seeds=(7,), device_counts=(4,))
        self.assertEqual({record["num_iot_devices"] for record in records}, {4})
        self.assertEqual({record["num_edge_nodes"] for record in records}, {2})
        self.assertEqual(len({id(record) for record in records}), 3)

    def test_same_seed_is_reproducible(self) -> None:
        first = ExperimentRunner(self.config).run("normal", seeds=(99,))[0]
        second = ExperimentRunner(self.config).run("normal", seeds=(99,))[0]
        self.assertEqual(first["task_workload"], second["task_workload"])
        self.assertEqual(first["total_generated"], second["total_generated"])

    def test_default_evaluation_supports_ten_reproducible_seeds(self) -> None:
        self.assertEqual(len(ExperimentRunner(NetworkConfig()).seeds), 10)
        self.assertEqual(ExperimentRunner(NetworkConfig()).seeds, tuple(range(42, 52)))

    def test_aggregation_mean_and_single_run_std(self) -> None:
        processor = ResultProcessor()
        records = [
            {"scheduler": "Baseline", "workload_mode": "normal", "num_iot_devices": 3,
             "num_edge_nodes": 2, "num_rounds": 1, "runtime_seconds": 1.0,
             "total_generated": 2, "energy_per_completed_task": None},
            {"scheduler": "Baseline", "workload_mode": "normal", "num_iot_devices": 3,
             "num_edge_nodes": 2, "num_rounds": 1, "runtime_seconds": 3.0,
             "total_generated": 4, "energy_per_completed_task": 2.0},
        ]
        # Populate optional metrics so the processor can aggregate a minimal fixture.
        for record in records:
            for metric in (
                "total_completed", "total_dropped", "completion_ratio", "drop_ratio", "throughput",
                "peak_congestion_ratio", "average_congestion_ratio", "maximum_queue_utilization",
                "average_queue_utilization", "congested_node_rounds", "total_energy_consumed",
                "average_remaining_energy", "minimum_remaining_energy", "maximum_remaining_energy",
                "average_engaged_nodes", "final_engaged_nodes", "average_queue", "maximum_queue",
                "minimum_queue", "queue_load_stddev", "redirected_tasks",
            ):
                record.setdefault(metric, 0.0)
        row = processor.aggregate(records)[0]
        self.assertEqual(row["runtime_seconds_mean"], 2.0)
        self.assertAlmostEqual(row["runtime_seconds_std"], 2**0.5)
        self.assertEqual(row["total_generated_mean"], 3)
        self.assertEqual(row["energy_per_completed_task_mean"], 2.0)

    def test_zero_denominator_and_improvement_are_safe(self) -> None:
        processor = ResultProcessor()
        self.assertIsNone(processor.improvement(0.0, 1.0, lower_is_better=True))
        record = ExperimentRunner(
            NetworkConfig(
                num_edge_nodes=1,
                num_iot_devices=1,
                num_rounds=1,
                task_generation_probability=0.0,
                num_experiment_runs=1,
            )
        ).run("normal")[0]
        self.assertEqual(record["completion_ratio"], 0.0)
        self.assertEqual(record["drop_ratio"], 0.0)
        self.assertIsNone(record["energy_per_completed_task"])

    def test_improvement_table_labels_regressions(self) -> None:
        processor = ResultProcessor()
        records = [
            {"scheduler": "Baseline", "workload_mode": "normal", "num_iot_devices": 1,
             "num_edge_nodes": 1, "num_rounds": 1, "total_dropped": 2,
             "completion_ratio": 0.5, "average_congestion_ratio": 0.5,
             "peak_congestion_ratio": 0.5, "energy_per_completed_task": 2.0,
             "total_energy_consumed": 10.0},
            {"scheduler": "Energy-Aware Congestion", "workload_mode": "normal", "num_iot_devices": 1,
             "num_edge_nodes": 1, "num_rounds": 1, "total_dropped": 3,
             "completion_ratio": 0.6, "average_congestion_ratio": 0.4,
             "peak_congestion_ratio": 0.5, "energy_per_completed_task": 3.0,
             "total_energy_consumed": 12.0},
        ]
        aggregated = processor.aggregate(records)
        rows = processor.improvement_table(aggregated)
        dropped = next(row for row in rows if row["metric"] == "total_dropped")
        self.assertEqual(dropped["interpretation"], "regression")

    def test_result_storage_contains_traceability_fields(self) -> None:
        records = ExperimentRunner(self.config).run("normal")
        with tempfile.TemporaryDirectory() as directory:
            paths = ResultProcessor(Path(directory)).write_results(records, "normal")
            self.assertTrue(paths["raw_json"].exists())
            self.assertTrue(paths["aggregated_json"].exists())
            self.assertTrue(paths["comparison_csv"].exists())
            loaded = ResultProcessor.load_results(paths["raw_json"])
            self.assertEqual(len(loaded), len(records))
            text = paths["raw_json"].read_text(encoding="utf-8")
            for field in ("scheduler", "workload_mode", "seed", "num_iot_devices", "config", "history"):
                self.assertIn(field, text)



if __name__ == "__main__":
    unittest.main()
