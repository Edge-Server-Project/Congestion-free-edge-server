"""The edge-node model used by the network foundation."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot, isfinite

from .task import Task, TaskStatus


@dataclass(slots=True)
class EdgeNode:
    """A compute-capable node placed in a two-dimensional edge network.

    The queue stores actual tasks and energy is tracked by a simple simulation
    accounting model. It is not a physical battery model.
    """

    node_id: str
    x: float
    y: float
    processing_capacity: int
    queue_capacity: int
    initial_energy: float = 1000.0
    remaining_energy: float | None = None
    consumed_energy: float = 0.0
    is_active: bool = True
    queue: list[Task] = field(default_factory=list, init=False, repr=False)
    received_tasks: int = 0
    processed_tasks: int = 0
    dropped_tasks: int = 0
    dropped_by_reason: dict[str, int] = field(default_factory=dict)
    current_processing_count: int = 0
    _neighbour_ids: set[str] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the intrinsic state of the edge node."""
        if not self.node_id.strip():
            raise ValueError("node_id must not be empty")
        if not isfinite(self.x) or not isfinite(self.y):
            raise ValueError("node position must contain finite coordinates")
        if (
            isinstance(self.processing_capacity, bool)
            or not isinstance(self.processing_capacity, int)
            or self.processing_capacity <= 0
        ):
            raise ValueError("processing_capacity must be a positive integer")
        if (
            isinstance(self.queue_capacity, bool)
            or not isinstance(self.queue_capacity, int)
            or self.queue_capacity <= 0
        ):
            raise ValueError("queue_capacity must be a positive integer")
        if not isfinite(self.initial_energy) or self.initial_energy <= 0:
            raise ValueError("initial_energy must be a positive finite number")
        if self.remaining_energy is None:
            self.remaining_energy = self.initial_energy
        if not isfinite(self.remaining_energy) or not 0.0 <= self.remaining_energy <= self.initial_energy:
            raise ValueError("remaining_energy must be between zero and initial_energy")
        if not isfinite(self.consumed_energy) or self.consumed_energy < 0:
            raise ValueError("consumed_energy must be non-negative")
        self.consumed_energy = self.initial_energy - self.remaining_energy
        if self.remaining_energy == 0.0:
            self.is_active = False

    @property
    def position(self) -> tuple[float, float]:
        """Return the node's two-dimensional simulation coordinate."""
        return (self.x, self.y)

    @property
    def neighbours(self) -> tuple[str, ...]:
        """Return neighbouring node IDs in stable order."""
        return tuple(sorted(self._neighbour_ids))

    @property
    def queue_length(self) -> int:
        """Return the number of actual tasks waiting at this edge node."""
        return len(self.queue)

    @property
    def load_ratio(self) -> float:
        """Return the queue fraction, rejecting a state above configured capacity."""
        if self.queue_length > self.queue_capacity:
            raise ValueError(f"queue for {self.node_id} exceeds its configured capacity")
        return self.queue_length / self.queue_capacity

    @property
    def energy_level(self) -> float:
        """Backward-compatible alias for remaining energy."""
        return self.remaining_energy

    @property
    def alive(self) -> bool:
        """Return whether the node can participate in the simulation."""
        return self.is_active

    @property
    def energy_ratio(self) -> float:
        """Return remaining energy normalized by the node's initial energy."""
        return self.remaining_energy / self.initial_energy

    @property
    def energy_utilization(self) -> float:
        """Return the normalized fraction of energy already consumed."""
        return min(1.0, max(0.0, 1.0 - self.energy_ratio))

    def consume_energy(self, amount: float) -> bool:
        """Consume energy if affordable, never allowing a negative balance."""
        if amount < 0 or not isfinite(amount):
            raise ValueError("energy amount must be a non-negative finite number")
        if not self.is_active or self.remaining_energy < amount:
            return False
        self.remaining_energy -= amount
        self.consumed_energy += amount
        if self.remaining_energy <= 0.0:
            self.remaining_energy = 0.0
            self.is_active = False
        return True

    def consume_idle_energy(self, amount: float) -> bool:
        """Consume configured idle energy for an active node."""
        return self.consume_energy(amount)

    def distance_to(self, other: EdgeNode) -> float:
        """Return the Euclidean distance to another edge node."""
        return hypot(self.x - other.x, self.y - other.y)

    def add_neighbour(self, neighbour_id: str) -> None:
        """Add a distinct neighbouring node ID.

        The set-backed representation prevents duplicate neighbours.
        """
        if not neighbour_id.strip():
            raise ValueError("neighbour_id must not be empty")
        if neighbour_id == self.node_id:
            raise ValueError("a node cannot be its own neighbour")
        self._neighbour_ids.add(neighbour_id)

    def clear_neighbours(self) -> None:
        """Remove all currently recorded network connections."""
        self._neighbour_ids.clear()

    def enqueue_task(self, task: Task) -> bool:
        """Add an assigned task to this node's FIFO queue when capacity allows."""
        if task.assigned_edge_node_id != self.node_id:
            raise ValueError(f"task {task.task_id} is not assigned to {self.node_id}")
        if not self.is_active:
            reason = "inactive_node"
            task.mark_dropped(reason)
            self.dropped_by_reason[reason] = self.dropped_by_reason.get(reason, 0) + 1
            self.dropped_tasks += 1
            return False
        if self.queue_length >= self.queue_capacity:
            reason = "queue_full"
            task.mark_dropped(reason)
            self.dropped_by_reason[reason] = self.dropped_by_reason.get(reason, 0) + 1
            self.dropped_tasks += 1
            return False
        task.mark_queued()
        self.queue.append(task)
        self.received_tasks += 1
        return True

    def process_queued_tasks(self, processing_energy_per_task: float = 0.0) -> tuple[Task, ...]:
        """Process affordable tasks up to capacity and charge each completed task."""
        if not self.is_active:
            return ()
        processed: list[Task] = []
        while self.queue and len(processed) < self.processing_capacity:
            if self.remaining_energy < processing_energy_per_task:
                break
            task = self.queue.pop(0)
            self.current_processing_count = 1
            task.mark_processing()
            if not self.consume_energy(processing_energy_per_task):
                self.queue.insert(0, task)
                task.status = TaskStatus.QUEUED
                self.current_processing_count = 0
                break
            task.mark_completed()
            processed.append(task)
        self.current_processing_count = 0
        self.processed_tasks += len(processed)
        return tuple(processed)
