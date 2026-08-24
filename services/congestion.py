"""Queue-based congestion detection for the baseline edge simulation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from models import EdgeNode


class LoadState(str, Enum):
    """Queue utilization classifications used for congestion reporting."""

    LOW = "LOW"
    MODERATE = "MODERATE"
    CONGESTED = "CONGESTED"


@dataclass(frozen=True, slots=True)
class CongestionResult:
    """Measured queue state for one edge node at one observation point."""

    edge_node_id: str
    current_queue_size: int
    queue_capacity: int
    queue_utilization: float
    load_state: LoadState


@dataclass(frozen=True, slots=True)
class CongestionReport:
    """Aggregated congestion measurements for all supplied edge nodes."""

    node_results: tuple[CongestionResult, ...]
    congested_nodes: tuple[CongestionResult, ...]
    congestion_ratio: float
    average_queue_utilization: float
    maximum_queue_utilization: float
    minimum_queue_utilization: float
    most_loaded_node: CongestionResult | None

    @property
    def congested_node_count(self) -> int:
        """Return the number of nodes with congested queues."""
        return len(self.congested_nodes)


class CongestionAnalyzer:
    """Read and classify queue load without modifying edge-node behaviour."""

    def __init__(self, low_load_threshold: float, congestion_threshold: float) -> None:
        """Create an analyzer with validated, normalized classification thresholds."""
        self._validate_thresholds(low_load_threshold, congestion_threshold)
        self.low_load_threshold = low_load_threshold
        self.congestion_threshold = congestion_threshold

    def analyze(self, edge_nodes: Iterable[EdgeNode]) -> CongestionReport:
        """Return congestion measurements derived from each node's actual queue."""
        results = tuple(self.analyze_queue(node.node_id, len(node.queue), node.queue_capacity) for node in edge_nodes)
        if not results:
            return CongestionReport(
                node_results=(),
                congested_nodes=(),
                congestion_ratio=0.0,
                average_queue_utilization=0.0,
                maximum_queue_utilization=0.0,
                minimum_queue_utilization=0.0,
                most_loaded_node=None,
            )

        congested_nodes = tuple(result for result in results if result.load_state is LoadState.CONGESTED)
        utilizations = tuple(result.queue_utilization for result in results)
        most_loaded_node = min(
            results,
            key=lambda result: (-result.queue_utilization, result.edge_node_id),
        )
        return CongestionReport(
            node_results=results,
            congested_nodes=congested_nodes,
            congestion_ratio=len(congested_nodes) / len(results),
            average_queue_utilization=sum(utilizations) / len(utilizations),
            maximum_queue_utilization=max(utilizations),
            minimum_queue_utilization=min(utilizations),
            most_loaded_node=most_loaded_node,
        )

    def analyze_queue(
        self, edge_node_id: str, current_queue_size: int, queue_capacity: int
    ) -> CongestionResult:
        """Validate and classify one explicit queue-size observation."""
        if not edge_node_id.strip():
            raise ValueError("edge_node_id must not be empty")
        if isinstance(queue_capacity, bool) or not isinstance(queue_capacity, int) or queue_capacity <= 0:
            raise ValueError("queue_capacity must be a positive integer")
        if isinstance(current_queue_size, bool) or not isinstance(current_queue_size, int):
            raise ValueError("current_queue_size must be an integer")
        if not 0 <= current_queue_size <= queue_capacity:
            raise ValueError("current_queue_size must be between zero and queue_capacity")

        utilization = current_queue_size / queue_capacity
        return CongestionResult(
            edge_node_id=edge_node_id,
            current_queue_size=current_queue_size,
            queue_capacity=queue_capacity,
            queue_utilization=utilization,
            load_state=self._classify(utilization),
        )

    def _classify(self, utilization: float) -> LoadState:
        """Classify one normalized utilization without applying any policy action."""
        if utilization < self.low_load_threshold:
            return LoadState.LOW
        if utilization < self.congestion_threshold:
            return LoadState.MODERATE
        return LoadState.CONGESTED

    @staticmethod
    def _validate_thresholds(low_load_threshold: float, congestion_threshold: float) -> None:
        if not 0.0 <= low_load_threshold <= 1.0:
            raise ValueError("low_load_threshold must be between zero and one")
        if not 0.0 <= congestion_threshold <= 1.0:
            raise ValueError("congestion_threshold must be between zero and one")
        if low_load_threshold > congestion_threshold:
            raise ValueError("low_load_threshold must not exceed congestion_threshold")
