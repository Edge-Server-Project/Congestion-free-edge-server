"""Baseline and congestion-aware task placement policies."""

from __future__ import annotations

from dataclasses import dataclass

from models import EdgeNode, IoTDevice, Task
from network import Network


@dataclass(frozen=True, slots=True)
class SchedulingDecision:
    """Record how one generated task was placed."""

    task_id: str
    baseline_candidate_node_id: str
    actual_selected_node_id: str
    redirected: bool
    reason: str
    baseline_utilization: float = 0.0
    selected_utilization: float = 0.0
    alternative_node_ids: tuple[str, ...] = ()


class BaselineScheduler:
    """Preserve the original nearest-edge assignment behavior."""

    name = "Baseline"

    def assign_task(self, task: Task, source: IoTDevice, network: Network) -> SchedulingDecision:
        """Assign to the nearest edge node, even when that node later rejects a full queue."""
        candidate = self._baseline_candidate(source, network)
        task.baseline_candidate_node_id = candidate.node_id
        task.assign_to(candidate.node_id)
        candidate.enqueue_task(task)
        return SchedulingDecision(
            task_id=task.task_id,
            baseline_candidate_node_id=candidate.node_id,
            actual_selected_node_id=candidate.node_id,
            redirected=False,
            reason="baseline nearest-edge assignment",
            baseline_utilization=candidate.load_ratio,
            selected_utilization=candidate.load_ratio,
        )

    @staticmethod
    def _baseline_candidate(source: IoTDevice, network: Network) -> EdgeNode:
        """Select the same nearest-node candidate used by the original simulator."""
        if not network.nodes:
            raise ValueError("cannot assign a task without edge nodes")
        return min(
            network.nodes,
            key=lambda node: (source.distance_to(node), node.node_id),
        )


class CongestionAwareScheduler(BaselineScheduler):
    """Redirect only new tasks whose baseline candidate is congested."""

    name = "Congestion-Aware"

    def __init__(self, congestion_threshold: float = 0.80) -> None:
        """Create a scheduler using the configured congestion threshold."""
        if not 0.0 <= congestion_threshold <= 1.0:
            raise ValueError("congestion_threshold must be between zero and one")
        self.congestion_threshold = congestion_threshold

    def assign_task(self, task: Task, source: IoTDevice, network: Network) -> SchedulingDecision:
        """Keep a healthy baseline target; otherwise choose the least-loaded alternative."""
        baseline = self._baseline_candidate(source, network)
        task.baseline_candidate_node_id = baseline.node_id
        baseline_is_congested = baseline.load_ratio >= self.congestion_threshold
        baseline_unavailable = (
            not baseline.is_active or baseline.queue_length >= baseline.queue_capacity
        )

        selected = baseline
        reason = "baseline candidate below congestion threshold"
        alternative_node_ids: tuple[str, ...] = ()
        if baseline_is_congested or baseline_unavailable:
            alternatives = [
                node
                for node in network.nodes
                if node.is_active and node.node_id != baseline.node_id and node.queue_length < node.queue_capacity
            ]
            alternative_node_ids = tuple(node.node_id for node in alternatives)
            if alternatives:
                selected = min(
                    alternatives,
                    key=lambda node: (
                        node.load_ratio,
                        -node.queue_capacity + node.queue_length,
                        node.node_id,
                    ),
                )
                reason = (
                    "baseline candidate unavailable; selected lowest-utilization alternative"
                    if baseline_unavailable and not baseline_is_congested
                    else "baseline candidate congested; selected lowest-utilization alternative"
                )
            else:
                reason = (
                    "baseline candidate unavailable; no available alternative"
                    if baseline_unavailable and not baseline_is_congested
                    else "baseline candidate congested; no available alternative"
                )

        task.assign_to(selected.node_id)
        selected.enqueue_task(task)
        return SchedulingDecision(
            task_id=task.task_id,
            baseline_candidate_node_id=baseline.node_id,
            actual_selected_node_id=selected.node_id,
            redirected=selected.node_id != baseline.node_id,
            reason=reason,
            baseline_utilization=baseline.load_ratio,
            selected_utilization=selected.load_ratio,
            alternative_node_ids=alternative_node_ids,
        )


class EnergyAwareCongestionScheduler(CongestionAwareScheduler):
    """Place tasks using normalized congestion and energy-cost factors."""

    name = "Energy-Aware Congestion"

    def __init__(
        self,
        congestion_threshold: float = 0.80,
        w_congestion: float = 0.70,
        w_energy: float = 0.30,
        min_operational_energy: float = 1.0,
    ) -> None:
        """Create a transparent weighted-score placement policy."""
        super().__init__(congestion_threshold)
        if not 0.0 <= w_congestion <= 1.0 or not 0.0 <= w_energy <= 1.0:
            raise ValueError("energy-aware weights must be between zero and one")
        if abs((w_congestion + w_energy) - 1.0) > 1e-9:
            raise ValueError("w_congestion + w_energy must equal 1.0")
        if min_operational_energy < 0:
            raise ValueError("min_operational_energy must be non-negative")
        self.w_congestion = w_congestion
        self.w_energy = w_energy
        self.min_operational_energy = min_operational_energy

    def assign_task(self, task: Task, source: IoTDevice, network: Network) -> SchedulingDecision:
        """Select an eligible node using normalized queue and energy cost."""
        baseline = self._baseline_candidate(source, network)
        task.baseline_candidate_node_id = baseline.node_id
        eligible = [
            node for node in network.nodes
            if node.is_active
            and node.remaining_energy > self.min_operational_energy
            and node.queue_length < node.queue_capacity
        ]
        non_congested = [node for node in eligible if node.load_ratio < self.congestion_threshold]
        pool = non_congested or eligible
        if not pool:
            task.assign_to(baseline.node_id)
            baseline.enqueue_task(task)
            selected = baseline
            reason = "no eligible node with capacity; existing rejection behavior"
            alternatives: tuple[str, ...] = ()
        else:
            selected = min(
                pool,
                key=lambda node: (
                    self.score(node),
                    node.node_id,
                ),
            )
            task.assign_to(selected.node_id)
            selected.enqueue_task(task)
            reason = (
                "selected lowest normalized congestion-energy score"
                if selected.node_id != baseline.node_id
                else "baseline candidate selected by normalized congestion-energy score"
            )
            alternatives = tuple(node.node_id for node in pool if node.node_id != baseline.node_id)

        return SchedulingDecision(
            task_id=task.task_id,
            baseline_candidate_node_id=baseline.node_id,
            actual_selected_node_id=selected.node_id,
            redirected=selected.node_id != baseline.node_id,
            reason=reason,
            baseline_utilization=baseline.load_ratio,
            selected_utilization=selected.load_ratio,
            alternative_node_ids=alternatives,
        )

    def score(self, node: EdgeNode) -> float:
        """Return lower-is-better normalized congestion plus energy-cost score."""
        normalized_load = node.load_ratio
        normalized_energy_cost = node.energy_utilization
        return self.w_congestion * normalized_load + self.w_energy * normalized_energy_cost


EnergyAwareScheduler = EnergyAwareCongestionScheduler
