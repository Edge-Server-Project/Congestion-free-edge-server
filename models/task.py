"""Computational task state for the edge simulation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TaskStatus(str, Enum):
    """Lifecycle states used by the baseline task flow."""

    GENERATED = "generated"
    ASSIGNED = "assigned"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    DROPPED = "dropped"


@dataclass(slots=True)
class Task:
    """A computational task created by one IoT device during one round."""

    task_id: str
    source_device_id: str
    creation_round: int
    status: TaskStatus = TaskStatus.GENERATED
    baseline_candidate_node_id: str | None = None
    assigned_edge_node_id: str | None = None
    drop_reason: str | None = None

    def __post_init__(self) -> None:
        """Validate immutable task identity and its initial lifecycle state."""
        if not self.task_id.strip():
            raise ValueError("task_id must not be empty")
        if not self.source_device_id.strip():
            raise ValueError("source_device_id must not be empty")
        if self.creation_round <= 0:
            raise ValueError("creation_round must be greater than zero")

    def assign_to(self, edge_node_id: str) -> None:
        """Associate this task with an edge node before it enters a queue."""
        if not edge_node_id.strip():
            raise ValueError("edge_node_id must not be empty")
        if self.status is not TaskStatus.GENERATED:
            raise ValueError(f"task {self.task_id} cannot be assigned from {self.status.value}")
        self.assigned_edge_node_id = edge_node_id
        self.status = TaskStatus.ASSIGNED

    def mark_queued(self) -> None:
        """Mark an assigned task as queued at its selected edge node."""
        if self.status is not TaskStatus.ASSIGNED:
            raise ValueError(f"task {self.task_id} must be assigned before queueing")
        self.status = TaskStatus.QUEUED

    def mark_processing(self) -> None:
        """Mark a queued task as currently being processed by its edge node."""
        if self.status is not TaskStatus.QUEUED:
            raise ValueError(f"task {self.task_id} must be queued before processing")
        self.status = TaskStatus.PROCESSING

    def mark_completed(self) -> None:
        """Mark a processing task as completed by an edge node."""
        if self.status is not TaskStatus.PROCESSING:
            raise ValueError(f"task {self.task_id} must be processing before completion")
        self.status = TaskStatus.COMPLETED

    def mark_dropped(self, reason: str = "queue_full") -> None:
        """Mark an assigned task as rejected and preserve the rejection reason."""
        if self.status is not TaskStatus.ASSIGNED:
            raise ValueError(f"task {self.task_id} must be assigned before dropping")
        if not reason.strip():
            raise ValueError("drop reason must not be empty")
        self.drop_reason = reason
        self.status = TaskStatus.DROPPED
