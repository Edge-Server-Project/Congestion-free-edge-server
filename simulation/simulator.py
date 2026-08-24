"""Deterministic task generation, allocation, queueing, and processing."""

from __future__ import annotations

import random
from dataclasses import dataclass
from statistics import pstdev

from config import NetworkConfig
from models import IoTDevice, Task, TaskStatus
from network import Network
from services import (
    BaselineScheduler,
    CongestionAnalyzer,
    CongestionReport,
    EnergyAwareCongestionScheduler,
    SchedulingDecision,
)


DEVICE_POSITION_SEED_OFFSET = 1
TASK_GENERATION_SEED_OFFSET = 2


@dataclass(frozen=True, slots=True)
class RoundMetrics:
    """Time-series measurements captured at the end of one simulation round."""

    round_number: int
    generated_this_round: int
    processed_this_round: int
    total_generated: int
    total_completed: int
    total_queued: int
    engaged_nodes_before: int
    engaged_nodes_after: int
    average_queue: float
    maximum_queue: int
    average_queue_utilization: float
    maximum_queue_utilization: float
    average_queue_utilization_before: float
    minimum_queue_utilization_before: float
    minimum_queue_utilization_after: float
    most_loaded_node: str | None
    congested_node_count: int
    congestion_ratio: float
    congested_node_count_before: int
    congestion_ratio_before: float
    maximum_queue_utilization_before: float
    maximum_queue_utilization_after: float
    dropped_this_round: int
    total_dropped: int
    redirected_tasks: int
    active_edge_nodes: int
    minimum_queue: int
    queue_load_stddev: float
    total_energy_remaining: float
    total_energy_consumed: float
    average_node_energy: float
    minimum_node_energy: float
    maximum_node_energy: float
    energy_consumed_this_round: float
    energy_consumed_per_completed_task: float | None


@dataclass(frozen=True, slots=True)
class RoundReport:
    """Queue observations, processing activity, and metrics for one round."""

    round_number: int
    generated_tasks: tuple[Task, ...]
    assignment_states: dict[str, str]
    scheduling_decisions: tuple[SchedulingDecision, ...]
    queued_tasks: tuple[Task, ...]
    processed_tasks: tuple[Task, ...]
    processed_tasks_by_node: dict[str, tuple[Task, ...]]
    dropped_tasks: tuple[Task, ...]
    queue_sizes_before_processing: dict[str, int]
    queue_sizes_after_processing: dict[str, int]
    processing_counts: dict[str, int]
    congestion_before_processing: CongestionReport
    congestion_after_processing: CongestionReport
    metrics: RoundMetrics

    @property
    def queue_sizes(self) -> dict[str, int]:
        """Return the post-processing queue sizes for backwards compatibility."""
        return self.queue_sizes_after_processing

    @property
    def engaged_node_count(self) -> int:
        """Return the number of nodes engaged after synchronous processing."""
        return self.metrics.engaged_nodes_after

    @property
    def total_generated(self) -> int:
        """Return the cumulative number of generated tasks."""
        return self.metrics.total_generated

    @property
    def total_completed(self) -> int:
        """Return the cumulative number of completed tasks."""
        return self.metrics.total_completed

    @property
    def queue_utilizations_before_processing(self) -> dict[str, float]:
        """Return pre-processing queue ratios derived by congestion analysis."""
        return {
            result.edge_node_id: result.queue_utilization
            for result in self.congestion_before_processing.node_results
        }

    @property
    def load_classifications_before_processing(self) -> dict[str, str]:
        """Return pre-processing load classes derived by congestion analysis."""
        return {
            result.edge_node_id: result.load_state.value
            for result in self.congestion_before_processing.node_results
        }


class Simulator:
    """Run the baseline IoT-to-edge task flow for a fixed number of rounds."""

    def __init__(self, config: NetworkConfig) -> None:
        """Create the edge network, IoT devices, and deterministic random streams."""
        self.config = config
        self.network = Network(config)
        self._device_rng = random.Random(config.seed + DEVICE_POSITION_SEED_OFFSET)
        self._task_rng = random.Random(config.seed + TASK_GENERATION_SEED_OFFSET)
        self._devices = self._create_iot_devices()
        self._task_sequence = 0
        self._all_tasks: list[Task] = []
        self._completed_tasks: list[Task] = []
        self._dropped_tasks: list[Task] = []
        self._history: list[RoundMetrics] = []
        self.congestion_analyzer = CongestionAnalyzer(
            config.low_load_threshold,
            config.congestion_threshold,
        )

    @property
    def devices(self) -> tuple[IoTDevice, ...]:
        """Return the configured IoT devices in deterministic creation order."""
        return tuple(self._devices.values())

    @property
    def all_tasks(self) -> tuple[Task, ...]:
        """Return every task generated during this simulation instance."""
        return tuple(self._all_tasks)

    @property
    def history(self) -> tuple[RoundMetrics, ...]:
        """Return one metrics record for every executed simulation round."""
        return tuple(self._history)

    @property
    def peak_congestion_ratio(self) -> float:
        """Return the highest pre-processing congestion ratio observed so far."""
        return max((metric.congestion_ratio_before for metric in self._history), default=0.0)

    @property
    def peak_queue_utilization(self) -> float:
        """Return the highest pre-processing utilization observed so far."""
        return max(
            (metric.maximum_queue_utilization_before for metric in self._history),
            default=0.0,
        )

    @property
    def total_congested_node_rounds(self) -> int:
        """Return the sum of congested-node observations across rounds."""
        return sum(metric.congested_node_count_before for metric in self._history)

    @property
    def maximum_congested_nodes(self) -> int:
        """Return the largest number of congested nodes in one round."""
        return max((metric.congested_node_count_before for metric in self._history), default=0)

    @property
    def peak_congested_nodes(self) -> int:
        """Return the peak number of congested nodes before processing."""
        return self.maximum_congested_nodes

    @property
    def task_accounting(self) -> dict[str, int]:
        """Return generated/completed/queued/dropped counts from task state."""
        counts = {status.name.lower(): 0 for status in TaskStatus}
        for task in self._all_tasks:
            counts[task.status.name.lower()] += 1
        return {
            "generated": len(self._all_tasks),
            "completed": counts[TaskStatus.COMPLETED.name.lower()],
            "queued": sum(len(node.queue) for node in self.network.nodes),
            "dropped": counts[TaskStatus.DROPPED.name.lower()],
            "assigned": counts[TaskStatus.ASSIGNED.name.lower()],
            "processing": counts[TaskStatus.PROCESSING.name.lower()],
        }

    def validate_task_accounting(self) -> None:
        """Ensure no generated task disappears between lifecycle states."""
        accounting = self.task_accounting
        if accounting["assigned"] or accounting["processing"]:
            raise AssertionError(f"unfinished task lifecycle states: {accounting}")
        accounted = accounting["completed"] + accounting["queued"] + accounting["dropped"]
        if accounting["generated"] != accounted:
            raise AssertionError(f"task accounting mismatch: {accounting}")

    def validate_energy_accounting(self) -> None:
        """Ensure node energy balances conserve the configured total."""
        initial = self.config.initial_energy * len(self.network.nodes)
        remaining = sum(node.remaining_energy for node in self.network.nodes)
        consumed = sum(node.consumed_energy for node in self.network.nodes)
        if any(node.remaining_energy < 0 for node in self.network.nodes):
            raise AssertionError("negative node energy detected")
        if abs(initial - (remaining + consumed)) > 1e-9:
            raise AssertionError(
                f"energy accounting mismatch: initial={initial}, remaining={remaining}, consumed={consumed}"
            )

    def run(self, scheduler: object | None = None) -> tuple[RoundReport, ...]:
        """Execute each configured round with the selected placement policy."""
        selected_scheduler = self._resolve_scheduler(scheduler)
        return tuple(
            self.run_round(round_number, selected_scheduler)
            for round_number in range(1, self.config.num_rounds + 1)
        )

    def run_round(self, round_number: int, scheduler: object | None = None) -> RoundReport:
        """Generate, assign, queue, process, and record one simulation round."""
        if round_number <= 0:
            raise ValueError("round_number must be greater than zero")

        selected_scheduler = self._resolve_scheduler(scheduler)
        generated_tasks = self._generate_tasks(round_number)
        queued_tasks, scheduling_decisions = self._assign_and_queue_tasks(
            generated_tasks,
            selected_scheduler,
        )
        dropped_tasks = tuple(task for task in generated_tasks if task.status is TaskStatus.DROPPED)
        self._dropped_tasks.extend(dropped_tasks)
        assignment_states = {task.task_id: task.status.name for task in generated_tasks}
        queue_sizes_before = self._queue_sizes()
        congestion_before = self.congestion_analyzer.analyze(self.network.nodes)
        self._assert_queue_consistency(queued_tasks)
        processing_counts, processed_tasks_by_node, processed_tasks = self._process_queued_tasks()
        queue_sizes_after = self._queue_sizes()
        congestion_after = self.congestion_analyzer.analyze(self.network.nodes)

        self._completed_tasks.extend(processed_tasks)
        metrics = self._build_metrics(
            round_number=round_number,
            generated_this_round=len(generated_tasks),
            processed_this_round=len(processed_tasks),
            dropped_this_round=len(dropped_tasks),
            redirected_tasks=sum(decision.redirected for decision in scheduling_decisions),
            queue_sizes_before=queue_sizes_before,
            queue_sizes_after=queue_sizes_after,
            congestion_before=congestion_before,
            congestion_after=congestion_after,
        )
        self._history.append(metrics)
        self.validate_task_accounting()
        self.validate_energy_accounting()
        return RoundReport(
            round_number=round_number,
            generated_tasks=generated_tasks,
            assignment_states=assignment_states,
            scheduling_decisions=scheduling_decisions,
            queued_tasks=queued_tasks,
            processed_tasks=processed_tasks,
            processed_tasks_by_node=processed_tasks_by_node,
            dropped_tasks=dropped_tasks,
            queue_sizes_before_processing=queue_sizes_before,
            queue_sizes_after_processing=queue_sizes_after,
            processing_counts=processing_counts,
            congestion_before_processing=congestion_before,
            congestion_after_processing=congestion_after,
            metrics=metrics,
        )

    def _create_iot_devices(self) -> dict[str, IoTDevice]:
        """Place the configured simulation devices inside the network area."""
        devices: dict[str, IoTDevice] = {}
        for index in range(1, self.config.num_iot_devices + 1):
            device = IoTDevice(
                device_id=f"D{index}",
                x=self._device_rng.uniform(0.0, self.config.area_width),
                y=self._device_rng.uniform(0.0, self.config.area_height),
                task_generation_probability=self.config.task_generation_probability,
            )
            if device.device_id in devices:
                raise ValueError(f"duplicate device ID: {device.device_id}")
            devices[device.device_id] = device
        return devices

    def _generate_tasks(self, round_number: int) -> tuple[Task, ...]:
        """Ask each device once to produce a task for the current round."""
        generated_tasks: list[Task] = []
        for device in self.devices:
            next_task_id = f"T{self._task_sequence + 1:03d}"
            task = device.maybe_generate_task(next_task_id, round_number, self._task_rng)
            if task is not None:
                self._task_sequence += 1
                self._all_tasks.append(task)
                generated_tasks.append(task)
        return tuple(generated_tasks)

    def _assign_and_queue_tasks(
        self,
        tasks: tuple[Task, ...],
        scheduler: object,
    ) -> tuple[tuple[Task, ...], tuple[SchedulingDecision, ...]]:
        """Place tasks through the selected scheduler and retain decision records."""
        queued_tasks: list[Task] = []
        decisions: list[SchedulingDecision] = []
        for task in tasks:
            source = self._devices[task.source_device_id]
            decision = scheduler.assign_task(task, source, self.network)
            decisions.append(decision)
            if task.status is TaskStatus.QUEUED:
                queued_tasks.append(task)
        return tuple(queued_tasks), tuple(decisions)

    def _resolve_scheduler(self, scheduler: object | None) -> object:
        """Normalize an instance, class, or omitted scheduler to a policy."""
        if scheduler is None:
            return BaselineScheduler()
        if isinstance(scheduler, type):
            if issubclass(scheduler, EnergyAwareCongestionScheduler):
                return scheduler(
                    congestion_threshold=self.config.congestion_threshold,
                    w_congestion=self.config.w_congestion,
                    w_energy=self.config.w_energy,
                    min_operational_energy=self.config.min_operational_energy,
                )
            if issubclass(scheduler, BaselineScheduler) and scheduler is not BaselineScheduler:
                return scheduler(self.config.congestion_threshold)
            return scheduler()
        if not hasattr(scheduler, "assign_task"):
            raise TypeError("scheduler must provide assign_task(task, source, network)")
        return scheduler

    def _process_queued_tasks(
        self,
    ) -> tuple[dict[str, int], dict[str, tuple[Task, ...]], tuple[Task, ...]]:
        """Process each edge node's FIFO queue after current-round assignment."""
        processing_counts: dict[str, int] = {}
        processed_by_node: dict[str, tuple[Task, ...]] = {}
        processed_tasks: list[Task] = []
        for edge_node in self.network.nodes:
            processed = edge_node.process_queued_tasks(self.config.processing_energy_per_task)
            processing_counts[edge_node.node_id] = len(processed)
            processed_by_node[edge_node.node_id] = processed
            processed_tasks.extend(processed)
            if not processed:
                edge_node.consume_idle_energy(self.config.idle_energy_per_round)
        return processing_counts, processed_by_node, tuple(processed_tasks)

    def _queue_sizes(self) -> dict[str, int]:
        """Capture actual queue lengths, using each queue as the source of truth."""
        return {node.node_id: len(node.queue) for node in self.network.nodes}

    def _assert_queue_consistency(self, queued_tasks: tuple[Task, ...]) -> None:
        """Assert each queued task is stored by the edge node named on the task."""
        for task in queued_tasks:
            if task.assigned_edge_node_id is None:
                raise AssertionError(f"queued task {task.task_id} has no assigned edge node")
            edge_node = self.network.get_node(task.assigned_edge_node_id)
            if task not in edge_node.queue:
                raise AssertionError(
                    f"queued task {task.task_id} is absent from {task.assigned_edge_node_id}'s queue"
                )
            if task.status is not TaskStatus.QUEUED:
                raise AssertionError(f"task {task.task_id} is not in the queued lifecycle state")

    def _build_metrics(
        self,
        *,
        round_number: int,
        generated_this_round: int,
        processed_this_round: int,
        dropped_this_round: int,
        redirected_tasks: int,
        queue_sizes_before: dict[str, int],
        queue_sizes_after: dict[str, int],
        congestion_before: CongestionReport,
        congestion_after: CongestionReport,
    ) -> RoundMetrics:
        """Build the persisted end-of-round metrics from actual queues."""
        queue_sizes = tuple(queue_sizes_after.values())
        energies = tuple(node.remaining_energy for node in self.network.nodes)
        total_energy_remaining = sum(energies)
        total_energy_consumed = sum(node.consumed_energy for node in self.network.nodes)
        previous_total_energy = (
            self.config.initial_energy * len(self.network.nodes)
            if round_number == 1
            else self._history[-1].total_energy_remaining
        )
        energy_consumed_this_round = previous_total_energy - total_energy_remaining
        return RoundMetrics(
            round_number=round_number,
            generated_this_round=generated_this_round,
            processed_this_round=processed_this_round,
            total_generated=len(self._all_tasks),
            total_completed=len(self._completed_tasks),
            total_queued=sum(queue_sizes),
            engaged_nodes_before=sum(size > 0 for size in queue_sizes_before.values()),
            engaged_nodes_after=sum(size > 0 for size in queue_sizes_after.values()),
            average_queue=sum(queue_sizes) / len(queue_sizes),
            maximum_queue=max(queue_sizes),
            average_queue_utilization=congestion_after.average_queue_utilization,
            maximum_queue_utilization=congestion_after.maximum_queue_utilization,
            average_queue_utilization_before=congestion_before.average_queue_utilization,
            minimum_queue_utilization_before=congestion_before.minimum_queue_utilization,
            minimum_queue_utilization_after=congestion_after.minimum_queue_utilization,
            most_loaded_node=(
                congestion_after.most_loaded_node.edge_node_id
                if congestion_after.most_loaded_node is not None
                else None
            ),
            congested_node_count=congestion_after.congested_node_count,
            congestion_ratio=congestion_after.congestion_ratio,
            congested_node_count_before=congestion_before.congested_node_count,
            congestion_ratio_before=congestion_before.congestion_ratio,
            maximum_queue_utilization_before=congestion_before.maximum_queue_utilization,
            maximum_queue_utilization_after=congestion_after.maximum_queue_utilization,
            dropped_this_round=dropped_this_round,
            total_dropped=len(self._dropped_tasks),
            redirected_tasks=redirected_tasks,
            active_edge_nodes=sum(size > 0 for size in queue_sizes),
            minimum_queue=min(queue_sizes),
            queue_load_stddev=pstdev(queue_sizes) if len(queue_sizes) > 1 else 0.0,
            total_energy_remaining=total_energy_remaining,
            total_energy_consumed=total_energy_consumed,
            average_node_energy=total_energy_remaining / len(energies),
            minimum_node_energy=min(energies),
            maximum_node_energy=max(energies),
            energy_consumed_this_round=energy_consumed_this_round,
            energy_consumed_per_completed_task=(
                energy_consumed_this_round / processed_this_round
                if processed_this_round
                else None
            ),
        )
