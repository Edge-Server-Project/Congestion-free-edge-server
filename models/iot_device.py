"""Simulation-only IoT task sources."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from math import hypot, isfinite

from .edge_node import EdgeNode
from .task import Task


@dataclass(slots=True)
class IoTDevice:
    """An IoT device that probabilistically creates computational tasks."""

    device_id: str
    x: float
    y: float
    task_generation_probability: float
    generated_task_count: int = 0
    created_tasks: list[Task] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate the device identity, location, and producer configuration."""
        if not self.device_id.strip():
            raise ValueError("device_id must not be empty")
        if not isfinite(self.x) or not isfinite(self.y):
            raise ValueError("device position must contain finite coordinates")
        if not 0.0 <= self.task_generation_probability <= 1.0:
            raise ValueError("task_generation_probability must be between zero and one")

    @property
    def position(self) -> tuple[float, float]:
        """Return the device's two-dimensional simulation coordinate."""
        return (self.x, self.y)

    def maybe_generate_task(
        self, task_id: str, creation_round: int, rng: random.Random
    ) -> Task | None:
        """Create and retain a task when this round's deterministic draw succeeds."""
        if rng.random() >= self.task_generation_probability:
            return None

        task = Task(
            task_id=task_id,
            source_device_id=self.device_id,
            creation_round=creation_round,
        )
        self.created_tasks.append(task)
        self.generated_task_count += 1
        return task

    def distance_to(self, edge_node: EdgeNode) -> float:
        """Return the Euclidean distance to an edge node."""
        return hypot(self.x - edge_node.x, self.y - edge_node.y)
