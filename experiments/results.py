"""Machine-readable storage, aggregation, tables, and plotting for experiments."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable


AGGREGATE_METRICS = (
    "runtime_seconds",
    "total_generated",
    "total_completed",
    "total_dropped",
    "queued_tasks",
    "dropped_queue_full",
    "dropped_inactive_node",
    "dropped_unavailable_node",
    "completion_ratio",
    "drop_ratio",
    "throughput",
    "peak_congestion_ratio",
    "peak_congested_nodes",
    "average_congestion_ratio",
    "maximum_queue_utilization",
    "average_queue_utilization",
    "average_queue_utilization_after",
    "congested_node_rounds",
    "total_energy_consumed",
    "energy_per_completed_task",
    "average_remaining_energy",
    "minimum_remaining_energy",
    "maximum_remaining_energy",
    "average_engaged_nodes",
    "final_engaged_nodes",
    "average_queue",
    "maximum_queue",
    "minimum_queue",
    "queue_load_stddev",
    "redirected_tasks",
)


class ResultProcessor:
    """Persist raw records and derive reproducible summaries without running simulations."""

    def __init__(self, output_directory: str | Path = "results/experiments") -> None:
        self.output_directory = Path(output_directory)
        self.results_root = self.output_directory.parent
        self.plots_directory = self.results_root / "plots"
        self.summaries_directory = self.results_root / "summaries"

    @staticmethod
    def load_results(path: str | Path) -> list[dict]:
        """Load raw JSON records without rerunning the simulation."""
        source = Path(path)
        if source.suffix.lower() != ".json":
            raise ValueError("result loading currently expects a JSON raw-results file")
        value = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
            raise ValueError("raw result JSON must contain a list of records")
        return value

    def write_results(self, records: list[dict], label: str | None = None) -> dict[str, Path]:
        """Write raw/aggregated JSON and CSV files and return their paths."""
        if not records:
            raise ValueError("cannot write an empty experiment result set")
        mode = label or str(records[0]["workload_mode"])
        directory = self.output_directory / mode
        directory.mkdir(parents=True, exist_ok=True)
        raw_json = directory / "raw_results.json"
        raw_csv = directory / "raw_results.csv"
        aggregate_json = directory / "aggregated_results.json"
        aggregate_csv = directory / "aggregated_results.csv"
        self._write_json(raw_json, records)
        self._write_csv(raw_csv, records)
        aggregated = self.aggregate(records)
        self._write_json(aggregate_json, aggregated)
        self._write_csv(aggregate_csv, aggregated)
        summary_directory = self.summaries_directory / mode
        summary_directory.mkdir(parents=True, exist_ok=True)
        comparison_csv = summary_directory / "comparison_table.csv"
        self._write_csv(comparison_csv, self.comparison_table(aggregated))
        improvements_csv = summary_directory / "improvements.csv"
        self._write_csv(improvements_csv, self.improvement_table(aggregated))
        paths = {
            "raw_json": raw_json,
            "raw_csv": raw_csv,
            "aggregated_json": aggregate_json,
            "aggregated_csv": aggregate_csv,
            "comparison_csv": comparison_csv,
            "improvements_csv": improvements_csv,
        }
        if mode == "scalability":
            scalability_csv = summary_directory / "scalability_table.csv"
            self._write_csv(
                scalability_csv,
                [
                    {
                        "iot_devices": row["num_iot_devices"],
                        "scheduler": row["scheduler"],
                        "completion_ratio_mean": row["completion_ratio_mean"],
                        "peak_congestion_ratio_mean": row["peak_congestion_ratio_mean"],
                        "total_energy_consumed_mean": row["total_energy_consumed_mean"],
                        "total_dropped_mean": row["total_dropped_mean"],
                        "runtime_seconds_mean": row["runtime_seconds_mean"],
                    }
                    for row in aggregated
                ],
            )
            paths["scalability_csv"] = scalability_csv
        return paths

    def improvement_table(
        self, aggregated: Iterable[dict], num_iot_devices: int | None = None
    ) -> list[dict]:
        """Calculate directional changes against Baseline without hiding regressions."""
        rows = list(aggregated)
        if num_iot_devices is not None:
            rows = [row for row in rows if row["num_iot_devices"] == num_iot_devices]
        baseline = next((row for row in rows if row["scheduler"] == "Baseline"), None)
        if baseline is None:
            return []
        metrics = (
            ("completion_ratio", False),
            ("total_dropped", True),
            ("average_congestion_ratio", True),
            ("peak_congestion_ratio", True),
            ("energy_per_completed_task", True),
            ("total_energy_consumed", True),
        )
        output: list[dict] = []
        for proposed in rows:
            if proposed["scheduler"] == "Baseline":
                continue
            for metric, lower_is_better in metrics:
                base_value = baseline.get(f"{metric}_mean")
                proposed_value = proposed.get(f"{metric}_mean")
                change = (
                    self.improvement(base_value, proposed_value, lower_is_better=lower_is_better)
                    if base_value is not None and proposed_value is not None
                    else None
                )
                output.append(
                    {
                        "scheduler": proposed["scheduler"],
                        "metric": metric,
                        "direction": "lower-is-better" if lower_is_better else "higher-is-better",
                        "baseline_mean": base_value,
                        "proposed_mean": proposed_value,
                        "change_percent": change,
                        "interpretation": (
                            "not available" if change is None
                            else "improvement" if change > 0
                            else "regression" if change < 0
                            else "unchanged"
                        ),
                    }
                )
        return output

    def aggregate(self, records: Iterable[dict]) -> list[dict]:
        """Aggregate each scheduler/workload/scale group using mean and sample std."""
        groups: dict[tuple, list[dict]] = defaultdict(list)
        for record in records:
            key = (
                record["scheduler"],
                record["workload_mode"],
                record["num_iot_devices"],
                record["num_edge_nodes"],
                record["num_rounds"],
            )
            groups[key].append(record)
        output: list[dict] = []
        for key, group in sorted(groups.items(), key=lambda item: item[0]):
            scheduler, workload_mode, devices, nodes, rounds = key
            row = {
                "scheduler": scheduler,
                "workload_mode": workload_mode,
                "num_iot_devices": devices,
                "num_edge_nodes": nodes,
                "num_rounds": rounds,
                "runs": len(group),
            }
            for metric in AGGREGATE_METRICS:
                values = [record.get(metric) for record in group if record.get(metric) is not None]
                row[f"{metric}_mean"] = mean(values) if values else None
                row[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0 if values else None
            output.append(row)
        return output

    def comparison_table(
        self, aggregated: Iterable[dict], num_iot_devices: int | None = None
    ) -> list[dict]:
        """Return one comparison row per metric with mean/std formatted separately."""
        rows = list(aggregated)
        if num_iot_devices is not None:
            rows = [row for row in rows if row["num_iot_devices"] == num_iot_devices]
        metrics = (
            ("Completion Ratio", "completion_ratio"),
            ("Total Generated", "total_generated"),
            ("Total Completed", "total_completed"),
            ("Total Dropped", "total_dropped"),
            ("Drop Ratio", "drop_ratio"),
            ("Average Congestion Ratio", "average_congestion_ratio"),
            ("Peak Congestion Ratio", "peak_congestion_ratio"),
            ("Maximum Queue Utilization", "maximum_queue_utilization"),
            ("Average Queue Utilization", "average_queue_utilization"),
            ("Total Congested Node-Rounds", "congested_node_rounds"),
            ("Peak Congested Nodes", "peak_congested_nodes"),
            ("Average Queue", "average_queue"),
            ("Maximum Queue", "maximum_queue"),
            ("Minimum Queue", "minimum_queue"),
            ("Queue Std Dev", "queue_load_stddev"),
            ("Total Energy", "total_energy_consumed"),
            ("Energy / Completed Task", "energy_per_completed_task"),
            ("Average Remaining Energy", "average_remaining_energy"),
            ("Minimum Remaining Energy", "minimum_remaining_energy"),
            ("Runtime (seconds)", "runtime_seconds"),
        )
        schedulers = _scheduler_order(rows)
        result: list[dict] = []
        for label, metric in metrics:
            row = {"metric": label}
            for scheduler in schedulers:
                match = next((item for item in rows if item["scheduler"] == scheduler), None)
                row[scheduler] = (
                    _format_mean_std(match.get(f"{metric}_mean"), match.get(f"{metric}_std"))
                    if match else "n/a"
                )
            result.append(row)
        return result

    def generate_graphs(self, records: Iterable[dict], label: str | None = None) -> list[Path]:
        """Generate the twelve requested standalone PNG figures."""
        rows = self.aggregate(records)
        if not rows:
            return []
        mode = label or rows[0]["workload_mode"]
        graph_dir = self.plots_directory / mode
        graph_dir.mkdir(parents=True, exist_ok=True)
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError("matplotlib is required to generate experiment graphs") from exc

        paths: list[Path] = []
        scheduler_metrics = (
            ("scheduler_completion_ratio.png", "Scheduler vs Completion Ratio", "completion_ratio_mean"),
            ("scheduler_dropped_tasks.png", "Scheduler vs Dropped Tasks", "total_dropped_mean"),
            ("scheduler_average_congestion_ratio.png", "Scheduler vs Average Congestion Ratio", "average_congestion_ratio_mean"),
            ("scheduler_peak_congestion_ratio.png", "Scheduler vs Peak Congestion Ratio", "peak_congestion_ratio_mean"),
            ("scheduler_maximum_queue_utilization.png", "Scheduler vs Maximum Queue Utilization", "maximum_queue_utilization_mean"),
            ("scheduler_average_queue_utilization.png", "Scheduler vs Average Queue Utilization", "average_queue_utilization_mean"),
            ("scheduler_energy_per_task.png", "Scheduler vs Energy Per Completed Task", "energy_per_completed_task_mean"),
            ("scheduler_total_energy.png", "Scheduler vs Total Energy Consumed", "total_energy_consumed_mean"),
        )
        first_scale = min(row["num_iot_devices"] for row in rows)
        scale_rows = [row for row in rows if row["num_iot_devices"] == first_scale]
        ordered_schedulers = _scheduler_order(scale_rows)
        for filename, title, metric in scheduler_metrics:
            fig, axis = plt.subplots(figsize=(7, 4))
            values = [
                next((row.get(metric) for row in scale_rows if row["scheduler"] == name), 0.0) or 0.0
                for name in ordered_schedulers
            ]
            errors = [
                next((row.get(metric.replace("_mean", "_std")) for row in scale_rows if row["scheduler"] == name), 0.0) or 0.0
                for name in ordered_schedulers
            ]
            axis.bar(
                ordered_schedulers,
                values,
                yerr=errors if any(errors) else None,
                capsize=4,
            )
            axis.set_title(title)
            axis.set_ylabel("Mean ± Std")
            axis.tick_params(axis="x", rotation=20)
            fig.tight_layout()
            path = graph_dir / filename
            fig.savefig(path, dpi=140)
            plt.close(fig)
            paths.append(path)

        scale_metrics = (
            ("devices_dropped_tasks.png", "IoT Devices vs Dropped Tasks", "total_dropped_mean"),
            ("devices_average_congestion_ratio.png", "IoT Devices vs Average Congestion Ratio", "average_congestion_ratio_mean"),
            ("devices_energy_per_task.png", "IoT Devices vs Energy Per Completed Task", "energy_per_completed_task_mean"),
            ("devices_runtime.png", "IoT Devices vs Simulation Runtime", "runtime_seconds_mean"),
        )
        for filename, title, metric in scale_metrics:
            fig, axis = plt.subplots(figsize=(7, 4))
            for scheduler in _scheduler_order(rows):
                points = sorted(
                    (row for row in rows if row["scheduler"] == scheduler),
                    key=lambda row: row["num_iot_devices"],
                )
                if not points:
                    continue
                x = [row["num_iot_devices"] for row in points]
                y = [row.get(metric) or 0.0 for row in points]
                error_key = metric.replace("_mean", "_std")
                error = [row.get(error_key) or 0.0 for row in points]
                axis.errorbar(
                    x,
                    y,
                    yerr=error if any(error) else None,
                    marker="o",
                    capsize=3,
                    label=scheduler,
                )
            axis.set_title(title)
            axis.set_xlabel("IoT devices")
            axis.set_ylabel("Mean ± Std")
            axis.legend()
            fig.tight_layout()
            path = graph_dir / filename
            fig.savefig(path, dpi=140)
            plt.close(fig)
            paths.append(path)
        return paths

    @staticmethod
    def improvement(baseline: float, proposed: float, *, lower_is_better: bool) -> float | None:
        """Return a directional percentage improvement, or None for zero baseline."""
        if baseline == 0:
            return None
        return (
            (baseline - proposed) / baseline * 100.0
            if lower_is_better
            else (proposed - baseline) / baseline * 100.0
        )

    @staticmethod
    def _write_json(path: Path, value: object) -> None:
        path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")

    @staticmethod
    def _write_csv(path: Path, rows: list[dict]) -> None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        fields = sorted({key for row in rows for key in row})
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def _scheduler_order(rows: Iterable[dict]) -> list[str]:
    preferred = ["Baseline", "Congestion-Aware", "Energy-Aware Congestion"]
    names = {row["scheduler"] for row in rows}
    return [name for name in preferred if name in names]


def _format_mean_std(value: float | None, spread: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4g} +/- {spread or 0.0:.4g}"
