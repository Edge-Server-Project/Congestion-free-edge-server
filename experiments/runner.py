"""Fair, reproducible experiment execution for the existing schedulers."""

from __future__ import annotations

from dataclasses import asdict, replace
from statistics import fmean
from time import monotonic
from collections import Counter
from typing import Iterable

from config import NetworkConfig
from services import (
    BaselineScheduler,
    CongestionAwareScheduler,
    EnergyAwareCongestionScheduler,
)
from simulation import Simulator


SCHEDULER_SPECS = (
    ("Baseline", BaselineScheduler),
    ("Congestion-Aware", CongestionAwareScheduler),
    ("Energy-Aware Congestion", EnergyAwareCongestionScheduler),
)


class ExperimentRunner:
    """Run independent scheduler replicas on identical deterministic workloads.

    A fresh ``Simulator`` is created for every scheduler/seed pair.  The
    simulator uses independent, seed-derived streams for topology, devices,
    and tasks, so scheduler execution cannot consume the workload RNG.  The
    runner additionally compares topology, device, and task signatures and
    raises if a future scheduler change violates that guarantee.
    """

    def __init__(self, config: NetworkConfig | None = None) -> None:
        self.config = config or NetworkConfig()

    @property
    def seeds(self) -> tuple[int, ...]:
        """Return the configured contiguous seed set."""
        return tuple(
            self.config.experiment_seed_start + offset
            for offset in range(self.config.num_experiment_runs)
        )

    def run(
        self,
        workload_mode: str = "normal",
        *,
        seeds: Iterable[int] | None = None,
        device_counts: Iterable[int] | None = None,
    ) -> list[dict]:
        """Run all schedulers for every requested seed and device count."""
        if workload_mode not in {"normal", "stress", "scalability"}:
            raise ValueError("workload_mode must be normal, stress, or scalability")
        seed_values = tuple(self.seeds if seeds is None else seeds)
        if not seed_values:
            raise ValueError("at least one experiment seed is required")
        counts = tuple(
            (self.config.scale_device_counts if workload_mode == "scalability" else (self.config.num_iot_devices,))
            if device_counts is None else device_counts
        )
        if not counts or any(count <= 0 for count in counts):
            raise ValueError("device_counts must contain positive integers")

        records: list[dict] = []
        for num_devices in counts:
            for seed in seed_values:
                base_config = self._configuration_for(workload_mode, seed, num_devices)
                reference_state: tuple | None = None
                reference_workload: tuple | None = None
                for scheduler_name, scheduler_class in SCHEDULER_SPECS:
                    # Each policy receives a new network, device collection,
                    # task objects, queues, and energy state.
                    simulator = Simulator(base_config)
                    signatures = self._signatures(simulator)
                    if reference_state is None:
                        reference_state = signatures
                    elif signatures != reference_state:
                        raise AssertionError(
                            "scheduler replicas do not share topology/device workload state"
                        )
                    started = monotonic()
                    simulator.run(scheduler_class)
                    elapsed = monotonic() - started
                    workload_signature = tuple(
                        (task.task_id, task.source_device_id, task.creation_round)
                        for task in simulator.all_tasks
                    )
                    if reference_workload is None:
                        reference_workload = workload_signature
                    elif workload_signature != reference_workload:
                        raise AssertionError(
                            "scheduler replicas generated different task workloads"
                        )
                    records.append(
                        self._record(
                            simulator,
                            scheduler_name=scheduler_name,
                            workload_mode=workload_mode,
                            seed=seed,
                            elapsed=elapsed,
                            config=base_config,
                            num_devices=num_devices,
                        )
                    )
        return records

    def run_scalability(
        self,
        *,
        seeds: Iterable[int] | None = None,
        device_counts: Iterable[int] | None = None,
    ) -> list[dict]:
        """Run the configurable device-count scalability matrix."""
        return self.run(
            "scalability",
            seeds=seeds,
            device_counts=device_counts,
        )

    def _configuration_for(
        self, workload_mode: str, seed: int, num_devices: int
    ) -> NetworkConfig:
        if workload_mode == "stress":
            stress = NetworkConfig.stress_test()
            return replace(stress, seed=seed, num_iot_devices=num_devices)
        return replace(self.config, seed=seed, num_iot_devices=num_devices)

    @staticmethod
    def _signatures(simulator: Simulator) -> tuple:
        topology = tuple(
            (node.node_id, node.x, node.y, tuple(node.neighbours))
            for node in simulator.network.nodes
        )
        devices = tuple(
            (device.device_id, device.x, device.y)
            for device in simulator.devices
        )
        return topology, devices, ()

    @staticmethod
    def _record(
        simulator: Simulator,
        *,
        scheduler_name: str,
        workload_mode: str,
        seed: int,
        elapsed: float,
        config: NetworkConfig,
        num_devices: int,
    ) -> dict:
        history = tuple(simulator.history)
        final = history[-1]
        generated = final.total_generated
        completed = final.total_completed
        dropped = final.total_dropped
        average_congestion = fmean(
            metric.congestion_ratio_before for metric in history
        ) if history else 0.0
        average_queue_utilization = fmean(
            metric.average_queue_utilization_before for metric in history
        ) if history else 0.0
        average_queue_utilization_after = fmean(
            metric.average_queue_utilization for metric in history
        ) if history else 0.0
        average_queue = fmean(metric.average_queue for metric in history) if history else 0.0
        average_engaged = fmean(metric.engaged_nodes_after for metric in history) if history else 0.0
        redirected = sum(metric.redirected_tasks for metric in history)
        accounting = simulator.task_accounting
        drop_reasons = Counter(
            task.drop_reason or "unspecified"
            for task in simulator.all_tasks
            if task.status.name == "DROPPED"
        )
        energy_per_task = (
            final.total_energy_consumed / completed if completed else None
        )
        return {
            "scheduler": scheduler_name,
            "workload_mode": workload_mode,
            "seed": seed,
            "num_iot_devices": num_devices,
            "num_edge_nodes": len(simulator.network.nodes),
            "num_rounds": config.num_rounds,
            "config": asdict(config),
            "runtime_seconds": elapsed,
            "total_generated": generated,
            "total_completed": completed,
            "total_dropped": dropped,
            "queued_tasks": accounting["queued"],
            "dropped_queue_full": drop_reasons.get("queue_full", 0),
            "dropped_inactive_node": drop_reasons.get("inactive_node", 0),
            "dropped_unavailable_node": drop_reasons.get("unavailable_node", 0),
            "completion_ratio": completed / generated if generated else 0.0,
            "drop_ratio": dropped / generated if generated else 0.0,
            "throughput": completed / config.num_rounds if config.num_rounds else 0.0,
            "peak_congestion_ratio": simulator.peak_congestion_ratio,
            "peak_congested_nodes": simulator.peak_congested_nodes,
            "average_congestion_ratio": average_congestion,
            "maximum_queue_utilization": simulator.peak_queue_utilization,
            "average_queue_utilization": average_queue_utilization,
            "average_queue_utilization_after": average_queue_utilization_after,
            "congested_node_rounds": simulator.total_congested_node_rounds,
            "total_energy_consumed": final.total_energy_consumed,
            "energy_per_completed_task": energy_per_task,
            "average_remaining_energy": final.average_node_energy,
            "minimum_remaining_energy": final.minimum_node_energy,
            "maximum_remaining_energy": final.maximum_node_energy,
            "average_engaged_nodes": average_engaged,
            "final_engaged_nodes": final.engaged_nodes_after,
            "average_queue": average_queue,
            "maximum_queue": max((metric.maximum_queue for metric in history), default=0),
            "minimum_queue": min((metric.minimum_queue for metric in history), default=0),
            "queue_load_stddev": fmean(metric.queue_load_stddev for metric in history)
            if history else 0.0,
            "redirected_tasks": redirected,
            "task_workload": [
                {
                    "task_id": task.task_id,
                    "source_device_id": task.source_device_id,
                    "creation_round": task.creation_round,
                }
                for task in simulator.all_tasks
            ],
            "history": [asdict(metric) for metric in history],
        }
