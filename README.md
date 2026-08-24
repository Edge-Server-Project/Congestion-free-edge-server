# Congestion-Free Edge Server

This repository provides a deterministic, baseline task flow for the project
**"Congestion Free Edge Architecture for Scalable Energy Efficient Edge
Computing"**.

## Current scope

- A configurable, two-dimensional edge network.
- Exactly 10 `EdgeNode` instances with the default configuration.
- Deterministic node placement through a seeded random generator.
- Undirected, distance-based connectivity between edge nodes.
- 20 IoT devices that automatically produce computational tasks.
- Nearest-edge baseline assignment, real FIFO edge queues, and processing.
- Interchangeable baseline and congestion-aware task-placement schedulers.
- Configurable simulation energy accounting and energy-aware congestion placement.
- Before/after queue observations, utilization measurements, load classes, and
  persisted per-round metrics for later congestion analysis.
- Basic validation and unit tests.

Run the network summary from the repository root:

```powershell
python main.py
```

Run the reproducible high-load congestion check without changing the normal
defaults:

```powershell
python main.py --stress
```

Run the tests:

```powershell
python -m unittest discover -s tests -v
```

Run the congestion-aware policy or compare both policies on the same stress
workload:

```powershell
python main.py --aware
python main.py --compare
```

Run the energy-aware scheduler directly:

```powershell
python main.py --energy
```

## Structure

- `config.py` holds the validated `NetworkConfig` values.
- `models/edge_node.py` defines the edge node and its real task queue.
- `models/iot_device.py` defines deterministic IoT task sources.
- `models/task.py` defines the task lifecycle.
- `network/network.py` creates nodes and constructs the topology.
- `services/congestion.py` classifies queue utilization and aggregates congestion.
- `services/scheduling.py` contains the nearest-edge baseline and congestion-aware placement policies.
- `models/edge_node.py` tracks remaining and consumed simulation energy.
- `simulation/simulator.py` executes generation, assignment, queueing,
  processing, queue snapshots, and metrics history.
- `experiments/runner.py` executes fair multi-seed scheduler and scalability
  matrices; `experiments/results.py` stores, aggregates, and plots results.
- `main.py` reports task lifecycle, queue changes, load status, and round summaries.
- `tests/` verifies network and simulation behaviour.

The configuration starts with the reference implementation's 1000 x 1000
network area, 180-unit communication range, and 0.45 task-production
probability. Devices use the nearest edge node as a deterministic baseline;
device-to-edge radio reachability is not modelled yet. The reference's
underwater, sink, packet, scheduler, optimization, and energy assumptions are
intentionally excluded.

Queue utilization is derived from actual queue contents (`len(queue) /
queue_capacity`). The default low/congestion classification thresholds are 50% and
80%; classification is observational only and never changes assignment. Since
processing is synchronous, an edge node is engaged before processing when its
queue is non-empty and after processing when work remains in its queue.

## Congestion detection

Congestion is measured as **queue utilization = current queue / queue
capacity**. A node is `LOW` below 50%, `MODERATE` from 50% to below 80%, and
`CONGESTED` at or above 80%. Detection runs after task assignment and before
processing to capture incoming workload pressure, then runs again after
processing to capture remaining backlog. The baseline nearest-edge assignment
policy does not react to these results yet.

The reported `peak_congestion_ratio` is specifically the maximum fraction of
edge nodes classified `CONGESTED` in any **pre-processing** round. It is not
maximum queue utilization and it is not the fraction of all node-round
observations. The evaluation reports these separately:

- `average_congestion_ratio`: mean per-round pre-processing congestion ratio.
- `peak_congestion_ratio`: maximum per-round pre-processing congestion ratio.
- `maximum_queue_utilization`: maximum node queue utilization observed before
  processing.
- `average_queue_utilization`: mean pre-processing queue utilization across
  rounds; `average_queue_utilization_after` is retained as a secondary metric.
- `congested_node_rounds`: sum of congested nodes across pre-processing rounds.
- `peak_congested_nodes`: largest number of congested nodes in one
  pre-processing round.

After-processing congestion is retained in the round history as secondary
information. Placement evaluation prioritizes the after-assignment,
pre-processing snapshot because it measures workload pressure before service
capacity removes tasks.

## Congestion-aware placement

The baseline scheduler selects the nearest edge node. The congestion-aware
scheduler computes that same baseline candidate first. If it is below the 80%
congestion threshold, the task remains there. If it is congested, active edge
nodes with available queue capacity are considered and the lowest-utilization
node is selected. Ties use higher remaining capacity, then lower node ID. This
is placement before queue insertion, not task migration; existing queued tasks
are never moved.

Comparison mode creates three fresh simulators with the same seed and
configuration, so topology, device positions, and task-generation sequence are
identical. Results are measurements only: a policy may reduce drops and
increase completion while still producing high congestion under sustained
overload.

The stress configuration uses seed `123`, 10-task queues, processing capacity
`1`, generation probability `0.80`, and 20 rounds. It reports peak queue
utilization, peak congestion ratio, maximum congested nodes, and total
congested node-rounds. Assignment output records the state at assignment time;
processing output separately identifies tasks that completed.

## Energy model

Each edge node starts with `1000.0` simulation energy units. A completed task
consumes `2.0` units. An active node that processes no task in a round consumes
`0.1` idle units. Processing is attempted before idle charging; energy is
charged exactly once per completed task and never becomes negative. A node at
zero energy becomes inactive. Queued tasks on an inactive node remain there
and are not migrated; the energy-aware scheduler does not assign work to
inactive or below-threshold nodes.

The energy-aware scheduler uses the normalized score:

`0.70 * queue_utilization + 0.30 * energy_utilization`

where energy utilization is `1 - remaining_energy / initial_energy`. Lower
scores are preferred. Nodes below `1.0` remaining energy unit are ineligible
for new energy-aware assignments. This is a transparent simulation model, not
a claim of physical battery accuracy.

## Not yet implemented

Task migration after assignment, dynamic load balancing, advanced energy
optimization, machine learning, cloud deployment, and a user
interface are future stages. Queue capacity is enforced: an assigned task is
explicitly marked `DROPPED` when its selected edge queue is full.

## Experimental evaluation and scalability

The evaluation runner compares the unchanged `Baseline`, `Congestion-Aware`,
and `Energy-Aware Congestion` schedulers. Every scheduler/seed pair gets a
fresh simulator with the same seeded topology, IoT-device positions, task
generation streams, queue/processing capacities, energy settings, and number
of rounds. The runner verifies the topology, device, and generated-task
signatures before recording a result, so mutable queues and task objects are
never shared between experiments.

Run the normal, stress, or device-scale matrix with:

```powershell
python main.py --experiment-normal
python main.py --experiment-stress
python main.py --scalability
python main.py --experiment-all
```

`NetworkConfig.num_experiment_runs`, `experiment_seed_start`, and
`scale_device_counts` control the matrix (defaults are ten seeds, 42 through
51, and 20/50/100/200 IoT devices). Each raw JSON/CSV record includes the
scheduler, workload mode, seed, scale, configuration, per-round history, and
final metrics. Aggregated JSON/CSV files contain mean and sample standard
deviation; a one-run group has standard deviation zero. Results are written
under `results/experiments/<mode>/`; comparison tables and directional
improvement tables are under `results/summaries/<mode>/`, and twelve separate
PNG figures are under `results/plots/<mode>/`.

Evaluation formulas use explicit zero-denominator handling: completion ratio
is completed/generated, drop ratio is dropped/generated, throughput is
completed/rounds, and energy per completed task is total energy/completed (or
`null` when no task completed). Runtime is measured with a monotonic timer.
Memory is not measured because it would add platform-specific complexity to
this small simulator.

The task accounting identity is checked after every round:

`generated = completed + currently queued + dropped`

The current model drops only when a selected queue is full or its node is
inactive. Drop reasons are stored explicitly; task migration is not performed.

## Final-project evaluation notes

### Problem and architecture

Edge devices generate work faster than a single edge node can always process.
Queues therefore provide the observable overload point:

```text
IoT Devices -> Edge Network -> Edge Nodes -> Task Queues -> Scheduling -> Processing
```

The three schedulers are deliberately simple baselines for comparison. The
project does not claim that any one strategy is universally optimal.

### Energy assumptions

Each node starts with 1000 simulation energy units. A completed task consumes
2 units and an active node that performs no processing consumes 0.1 idle units
per round. Energy is charged once, never becomes negative, and a zero-energy
node becomes inactive. The energy-aware score is:

`0.70 * queue_utilization + 0.30 * energy_utilization`

where `energy_utilization = 1 - remaining_energy / initial_energy`; lower is
preferred. This is a transparent simulation model, not a physical battery
claim.

### Limitations

The simulation does not model radio energy, transmission delay, packet loss,
dynamic topology, migration, node shutdown policies, memory usage, or physical
hardware calibration. Runtime measurements are machine-dependent. These
limitations are reported rather than hidden in the evaluation.
