"""Central configuration for the initial edge-network simulation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


# Experiment defaults are module-level names as well as dataclass fields so
# scripts can inspect or override the matrix without duplicating values.
NUM_EXPERIMENT_RUNS = 10
EXPERIMENT_SEED_START = 42
SCALE_DEVICE_COUNTS = (20, 50, 100, 200)


@dataclass(frozen=True, slots=True)
class NetworkConfig:
    """Settings used to create a deterministic two-dimensional edge network."""

    seed: int = 42
    num_edge_nodes: int = 10
    area_width: float = 1000.0
    area_height: float = 1000.0
    communication_range: float = 180.0
    processing_capacity: int = 1
    queue_capacity: int = 50
    initial_energy: float = 1000.0
    initial_edge_energy: float | None = None
    processing_energy_per_task: float = 2.0
    idle_energy_per_round: float = 0.1
    min_operational_energy: float = 1.0
    w_congestion: float = 0.70
    w_energy: float = 0.30
    num_iot_devices: int = 20
    num_rounds: int = 3
    task_generation_probability: float = 0.45
    low_load_threshold: float = 0.50
    congestion_threshold: float = 0.80
    # Evaluation settings are kept in the same immutable configuration so an
    # experiment can be reproduced from one recorded object.
    num_experiment_runs: int = NUM_EXPERIMENT_RUNS
    experiment_seed_start: int = EXPERIMENT_SEED_START
    scale_device_counts: tuple[int, ...] = SCALE_DEVICE_COUNTS
    results_directory: str = "results/experiments"

    def __post_init__(self) -> None:
        """Reject invalid settings before a network is created."""
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if (
            isinstance(self.num_edge_nodes, bool)
            or not isinstance(self.num_edge_nodes, int)
            or self.num_edge_nodes <= 0
        ):
            raise ValueError("num_edge_nodes must be a positive integer")
        if (
            isinstance(self.queue_capacity, bool)
            or not isinstance(self.queue_capacity, int)
            or self.queue_capacity <= 0
        ):
            raise ValueError("queue_capacity must be a positive integer")
        if (
            isinstance(self.processing_capacity, bool)
            or not isinstance(self.processing_capacity, int)
            or self.processing_capacity <= 0
        ):
            raise ValueError("processing_capacity must be a positive integer")
        if (
            isinstance(self.num_iot_devices, bool)
            or not isinstance(self.num_iot_devices, int)
            or self.num_iot_devices <= 0
        ):
            raise ValueError("num_iot_devices must be a positive integer")
        if (
            isinstance(self.num_rounds, bool)
            or not isinstance(self.num_rounds, int)
            or self.num_rounds <= 0
        ):
            raise ValueError("num_rounds must be a positive integer")
        self._require_positive_finite("area_width", self.area_width)
        self._require_positive_finite("area_height", self.area_height)
        self._require_positive_finite("communication_range", self.communication_range)
        self._require_non_negative_finite("initial_energy", self.initial_energy)
        if self.initial_edge_energy is not None:
            self._require_positive_finite("initial_edge_energy", self.initial_edge_energy)
            object.__setattr__(self, "initial_energy", self.initial_edge_energy)
        self._require_positive_finite("initial_energy", self.initial_energy)
        self._require_non_negative_finite("processing_energy_per_task", self.processing_energy_per_task)
        self._require_non_negative_finite("idle_energy_per_round", self.idle_energy_per_round)
        self._require_non_negative_finite("min_operational_energy", self.min_operational_energy)
        if not 0.0 <= self.w_congestion <= 1.0 or not 0.0 <= self.w_energy <= 1.0:
            raise ValueError("energy-aware scheduler weights must be between zero and one")
        if abs((self.w_congestion + self.w_energy) - 1.0) > 1e-9:
            raise ValueError("w_congestion + w_energy must equal 1.0")
        if not 0.0 <= self.task_generation_probability <= 1.0:
            raise ValueError("task_generation_probability must be between zero and one")
        if not 0.0 <= self.low_load_threshold <= 1.0:
            raise ValueError("low_load_threshold must be between zero and one")
        if not 0.0 <= self.congestion_threshold <= 1.0:
            raise ValueError("congestion_threshold must be between zero and one")
        if self.low_load_threshold > self.congestion_threshold:
            raise ValueError("low_load_threshold must not exceed congestion_threshold")
        if (
            isinstance(self.num_experiment_runs, bool)
            or not isinstance(self.num_experiment_runs, int)
            or self.num_experiment_runs <= 0
        ):
            raise ValueError("num_experiment_runs must be a positive integer")
        if isinstance(self.experiment_seed_start, bool) or not isinstance(self.experiment_seed_start, int):
            raise ValueError("experiment_seed_start must be an integer")
        if not self.scale_device_counts or any(
            isinstance(count, bool) or not isinstance(count, int) or count <= 0
            for count in self.scale_device_counts
        ):
            raise ValueError("scale_device_counts must contain positive integers")
        if not isinstance(self.results_directory, str) or not self.results_directory.strip():
            raise ValueError("results_directory must be a non-empty string")

    @classmethod
    def stress_test(cls) -> "NetworkConfig":
        """Return a fixed, reproducible configuration that creates queue pressure."""
        return cls(
            seed=123,
            queue_capacity=10,
            processing_capacity=1,
            task_generation_probability=0.80,
            num_rounds=20,
        )

    @property
    def high_load_threshold(self) -> float:
        """Backward-compatible alias for the congestion threshold."""
        return self.congestion_threshold

    @staticmethod
    def _require_positive_finite(name: str, value: float) -> None:
        if not isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive finite number")

    @staticmethod
    def _require_non_negative_finite(name: str, value: float) -> None:
        if not isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a non-negative finite number")
