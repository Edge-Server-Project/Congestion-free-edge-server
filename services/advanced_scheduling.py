"""Additional congestion-free task placement policies.

The optimization methods in this module are integration-ready heuristics. The
implementation guide does not identify source papers or prescribe equations,
so these classes must not be presented as paper-faithful reproductions.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Callable, Sequence

from models import EdgeNode, IoTDevice, Task
from network import Network
from .scheduling import SchedulingDecision


def _available(nodes: Sequence[EdgeNode]) -> list[EdgeNode]:
    return [n for n in nodes if n.is_active and n.queue_length < n.queue_capacity]


def _score(node: EdgeNode, virtual_queue: int = 0, *, delay: float = 1.0,
           congestion: float = 1.0, energy: float = 1.0, resource: float = 1.0) -> float:
    """Shared normalized lower-is-better queue/delay/energy objective."""
    q = node.queue_length + virtual_queue
    utilization = min(1.0, q / node.queue_capacity)
    return (delay * utilization + congestion * utilization
            + energy * node.energy_utilization + resource * (1.0 - node.energy_ratio))


class BatchScheduler:
    """Base class for policies that jointly allocate one round of tasks."""

    name = "Batch Scheduler"

    def assign_batch(self, tasks: tuple[Task, ...], sources: dict[str, IoTDevice],
                     network: Network) -> tuple[SchedulingDecision, ...]:
        nodes = network.nodes
        assignments = self.allocate(tasks, sources, nodes)
        decisions: list[SchedulingDecision] = []
        virtual: dict[str, int] = defaultdict(int)
        for task, node_id in zip(tasks, assignments):
            baseline = min(nodes, key=lambda n: (sources[task.source_device_id].distance_to(n), n.node_id))
            target = network.get_node(node_id) if node_id else baseline
            baseline_utilization = baseline.load_ratio
            selected_utilization = target.load_ratio
            task.baseline_candidate_node_id = baseline.node_id
            task.assign_to(target.node_id)
            target.enqueue_task(task)
            decisions.append(SchedulingDecision(
                task.task_id, baseline.node_id, target.node_id, target.node_id != baseline.node_id,
                self.name.lower() + " batch allocation" if node_id else "no feasible node; baseline rejection behavior",
                baseline_utilization,
                selected_utilization,
                tuple(n.node_id for n in nodes if n.node_id != baseline.node_id and n.is_active),
            ))
            if task.status.value == "queued":
                virtual[target.node_id] += 1
        return tuple(decisions)

    def allocate(self, tasks: tuple[Task, ...], sources: dict[str, IoTDevice],
                 nodes: tuple[EdgeNode, ...]) -> list[str | None]:
        raise NotImplementedError

    def assign_task(self, task: Task, source: IoTDevice, network: Network) -> SchedulingDecision:
        return self.assign_batch((task,), {source.device_id: source}, network)[0]


class LyapunovScheduler(BatchScheduler):
    """Queue-stability scheduler using one-step quadratic Lyapunov drift."""

    name = "Lyapunov"

    def __init__(self, congestion_weight: float = 1.0, energy_weight: float = 0.2,
                 delay_weight: float = 0.5) -> None:
        self.congestion_weight = congestion_weight
        self.energy_weight = energy_weight
        self.delay_weight = delay_weight

    @staticmethod
    def lyapunov_value(nodes: Sequence[EdgeNode], additions: dict[str, int] | None = None) -> float:
        additions = additions or {}
        return 0.5 * sum((n.queue_length + additions.get(n.node_id, 0)) ** 2 for n in nodes)

    def allocate(self, tasks, sources, nodes):
        virtual: dict[str, int] = defaultdict(int)
        result: list[str | None] = []
        baseline_value = self.lyapunov_value(nodes, virtual)
        for task in tasks:
            feasible = [n for n in nodes if n.is_active and n.queue_length + virtual[n.node_id] < n.queue_capacity]
            if not feasible:
                result.append(None)
                continue
            def cost(n: EdgeNode) -> tuple[float, str]:
                added = dict(virtual)
                added[n.node_id] = added.get(n.node_id, 0) + 1
                drift = self.lyapunov_value(nodes, added) - baseline_value
                return (drift + self.congestion_weight * n.load_ratio
                        + self.delay_weight * (n.queue_length + virtual[n.node_id]) / n.processing_capacity
                        + self.energy_weight * n.energy_utilization, n.node_id)
            selected = min(feasible, key=cost)
            result.append(selected.node_id)
            virtual[selected.node_id] += 1
        return result


class _PopulationScheduler(BatchScheduler):
    """Common finite, deterministic allocation-search support."""

    population_size = 16
    iterations = 20
    seed = 41

    def __init__(self, population_size: int | None = None, iterations: int | None = None,
                 seed: int | None = None) -> None:
        self.population_size = population_size or type(self).population_size
        self.iterations = iterations or type(self).iterations
        self.seed = type(self).seed if seed is None else seed
        if self.population_size < 2 or self.iterations <= 0:
            raise ValueError("population_size must be at least 2 and iterations must be positive")

    @staticmethod
    def _repair(vector: Sequence[int], tasks: tuple[Task, ...], nodes: tuple[EdgeNode, ...]) -> tuple[int, ...]:
        remaining = {n.node_id: n.queue_capacity - n.queue_length for n in nodes if n.is_active}
        valid = [n for n in nodes if n.is_active]
        repaired: list[int] = []
        for i, gene in enumerate(vector[:len(tasks)]):
            candidate = nodes[gene % len(nodes)] if nodes else None
            if candidate and remaining.get(candidate.node_id, 0) > 0:
                selected = candidate
            elif valid:
                selected = min((n for n in valid if remaining.get(n.node_id, 0) > 0),
                               key=lambda n: (n.queue_length + (len(repaired) / max(1, n.processing_capacity)), n.node_id),
                               default=None)
            else:
                selected = None
            if selected is None:
                repaired.append(-1)
            else:
                repaired.append(nodes.index(selected))
                remaining[selected.node_id] -= 1
        return tuple(repaired)

    @staticmethod
    def _fitness(vector: Sequence[int], tasks: tuple[Task, ...], nodes: tuple[EdgeNode, ...]) -> float:
        loads: dict[str, int] = defaultdict(int)
        score = 0.0
        for task, gene in zip(tasks, vector):
            if gene < 0 or not nodes:
                score += 10_000.0
                continue
            node = nodes[gene]
            loads[node.node_id] += 1
            score += _score(node, loads[node.node_id] - 1)
        return score

    def allocate(self, tasks, sources, nodes):
        if not tasks:
            return []
        rng = random.Random(self.seed)
        feasible = [i for i, n in enumerate(nodes) if n.is_active and n.queue_length < n.queue_capacity]
        if not feasible:
            return [None] * len(tasks)
        population = [self._repair([rng.choice(feasible) for _ in tasks], tasks, nodes)
                      for _ in range(self.population_size)]
        population = self.search(population, tasks, nodes, rng)
        best = min(population, key=lambda v: self._fitness(v, tasks, nodes))
        return [nodes[g].node_id if g >= 0 else None for g in best]

    def search(self, population, tasks, nodes, rng):
        raise NotImplementedError


class IDOScheduler(_PopulationScheduler):
    """Niche-based self-organizing allocation search (guide-level heuristic)."""

    name = "IDO/DNSLA"

    def search(self, population, tasks, nodes, rng):
        for _ in range(self.iterations):
            ranked = sorted(population, key=lambda v: self._fitness(v, tasks, nodes))
            niches: dict[tuple[int, ...], list[tuple[int, ...]]] = defaultdict(list)
            for vector in ranked:
                niches[tuple(sorted(set(vector)))].append(vector)
            next_population = ranked[:max(1, self.population_size // 4)]
            for members in niches.values():
                leader = members[0]
                for _ in range(max(1, self.population_size // max(1, len(niches)))):
                    child = list(leader)
                    for i in range(len(child)):
                        if rng.random() < 0.2:
                            child[i] = rng.randrange(len(nodes))
                    next_population.append(self._repair(child, tasks, nodes))
                    if len(next_population) >= self.population_size:
                        break
                if len(next_population) >= self.population_size:
                    break
            population = next_population[:self.population_size]
        return population


class DARC_DEScheduler(_PopulationScheduler):
    """Discrete differential evolution with bounded node-index repair."""

    name = "DARC-DE"

    def __init__(self, population_size=None, iterations=None, seed=None, scale: float = 0.6,
                 crossover_rate: float = 0.8) -> None:
        super().__init__(population_size, iterations, seed)
        if not 0 <= scale <= 2 or not 0 <= crossover_rate <= 1:
            raise ValueError("scale must be in [0, 2] and crossover_rate in [0, 1]")
        self.scale, self.crossover_rate = scale, crossover_rate

    def search(self, population, tasks, nodes, rng):
        for _ in range(self.iterations):
            updated = []
            for i, parent in enumerate(population):
                choices = [v for j, v in enumerate(population) if j != i]
                a, b, c = rng.sample(choices, 3) if len(choices) >= 3 else (parent, parent, parent)
                trial = list(parent)
                forced = rng.randrange(len(parent))
                for k in range(len(parent)):
                    if k == forced or rng.random() < self.crossover_rate:
                        trial[k] = round(a[k] + self.scale * (b[k] - c[k]))
                trial = self._repair(trial, tasks, nodes)
                updated.append(trial if self._fitness(trial, tasks, nodes) < self._fitness(parent, tasks, nodes) else parent)
            population = updated
        return population


class PSOScheduler(_PopulationScheduler):
    """Discrete particle swarm optimization over node-index allocation vectors."""

    name = "PSO"

    def search(self, population, tasks, nodes, rng):
        velocity = [[0.0] * len(tasks) for _ in population]
        personal = list(population)
        global_best = min(population, key=lambda v: self._fitness(v, tasks, nodes))
        for _ in range(self.iterations):
            for p, position in enumerate(population):
                position = list(position)
                for i in range(len(tasks)):
                    velocity[p][i] = (0.55 * velocity[p][i]
                                      + 1.3 * rng.random() * (personal[p][i] - position[i])
                                      + 1.3 * rng.random() * (global_best[i] - position[i]))
                    position[i] = round(position[i] + velocity[p][i])
                population[p] = self._repair(position, tasks, nodes)
                if self._fitness(population[p], tasks, nodes) < self._fitness(personal[p], tasks, nodes):
                    personal[p] = population[p]
            global_best = min(personal, key=lambda v: self._fitness(v, tasks, nodes))
        return personal


class GAScheduler(_PopulationScheduler):
    """Tournament-selected genetic search with uniform crossover and mutation."""

    name = "GA"

    def search(self, population, tasks, nodes, rng):
        for _ in range(self.iterations):
            ranked = sorted(population, key=lambda v: self._fitness(v, tasks, nodes))
            next_population = ranked[:2]
            while len(next_population) < self.population_size:
                pool = rng.sample(ranked[:max(2, len(ranked) // 2)], 2)
                a, b = pool
                child = [a[i] if rng.random() < 0.5 else b[i] for i in range(len(tasks))]
                for i in range(len(child)):
                    if rng.random() < 0.08:
                        child[i] = rng.randrange(len(nodes))
                next_population.append(self._repair(child, tasks, nodes))
            population = next_population
        return population


class PSGScheduler(BatchScheduler):
    """Priority-aware semi-greedy online placement with a restricted list."""

    name = "Priority-aware Semi-Greedy"

    def __init__(self, rcl_size: int = 3, seed: int = 41) -> None:
        if rcl_size <= 0:
            raise ValueError("rcl_size must be positive")
        self.rcl_size, self.seed = rcl_size, seed

    def allocate(self, tasks, sources, nodes):
        rng = random.Random(self.seed)
        virtual: dict[str, int] = defaultdict(int)
        result: list[tuple[str, str] | None] = []
        for task in sorted(tasks, key=lambda t: (-t.priority, t.deadline_round or 10**9, t.creation_round, t.task_id)):
            feasible = [n for n in nodes if n.is_active and n.queue_length + virtual[n.node_id] < n.queue_capacity]
            ranked = sorted(feasible, key=lambda n: (_score(n, virtual[n.node_id]), n.node_id))
            shortlist = ranked[:self.rcl_size]
            if not shortlist:
                result.append(None)
                continue
            choice = rng.choice(shortlist)
            virtual[choice.node_id] += 1
            result.append((task.task_id, choice.node_id))
        # BatchScheduler applies in input task order; key the choices back to task IDs.
        by_task = {task_id: node_id for item in result if item is not None for task_id, node_id in (item,)}
        return [by_task.get(t.task_id) for t in tasks]


class GaleShapleyScheduler(BatchScheduler):
    """Capacity-aware deferred acceptance using task-proposing preferences."""

    name = "Gale-Shapley"

    def allocate(self, tasks, sources, nodes):
        remaining = {n.node_id: max(0, n.queue_capacity - n.queue_length) if n.is_active else 0 for n in nodes}
        preferences = {}
        for task in tasks:
            source = sources[task.source_device_id]
            preferences[task.task_id] = [n.node_id for n in sorted(nodes, key=lambda n: (
                _score(n), source.distance_to(n), n.node_id)) if remaining[n.node_id] > 0]
        node_rank = {n.node_id: {t.task_id: rank for rank, t in enumerate(sorted(tasks, key=lambda t: (
            -t.priority, t.deadline_round or 10**9, t.task_id)))} for n in nodes}
        next_choice = {t.task_id: 0 for t in tasks}
        held: dict[str, list[Task]] = defaultdict(list)
        free = [t for t in tasks]
        while free:
            task = free.pop(0)
            i = next_choice[task.task_id]
            prefs = preferences[task.task_id]
            if i >= len(prefs):
                continue
            node_id = prefs[i]
            next_choice[task.task_id] += 1
            held[node_id].append(task)
            held[node_id].sort(key=lambda t: (node_rank[node_id].get(t.task_id, 10**9), t.task_id))
            if len(held[node_id]) > remaining[node_id]:
                rejected = held[node_id].pop()
                if next_choice[rejected.task_id] < len(preferences[rejected.task_id]):
                    free.append(rejected)
        matching = {t.task_id: n for n, assigned in held.items() for t in assigned}
        return [matching.get(t.task_id) for t in tasks]


class MLRLScheduler(BatchScheduler):
    """Hierarchical tabular Q-learning approximation with discretized queues.

    The policy is intentionally lightweight and learns during online episodes;
    it is not a reproduction of a specific multi-layer RL paper.
    """

    name = "Multi-Layer RL (tabular approximation)"

    def __init__(self, learning_rate: float = 0.15, discount: float = 0.9,
                 exploration: float = 0.1, seed: int = 41) -> None:
        if not 0 < learning_rate <= 1 or not 0 <= discount <= 1 or not 0 <= exploration <= 1:
            raise ValueError("invalid Q-learning parameters")
        self.learning_rate, self.discount = learning_rate, discount
        self.exploration, self.seed = exploration, seed
        self.q_values: dict[tuple[tuple[int, ...], int], float] = {}
        self._pending: tuple[tuple[int, ...], int] | None = None
        self._rng = random.Random(seed)

    @staticmethod
    def _state(nodes: Sequence[EdgeNode]) -> tuple[int, ...]:
        return tuple(min(4, int(n.load_ratio * 5)) for n in nodes)

    def allocate(self, tasks, sources, nodes):
        state = self._state(nodes)
        result = []
        virtual: dict[str, int] = defaultdict(int)
        for task in tasks:
            feasible = [i for i, n in enumerate(nodes)
                        if n.is_active and n.queue_length + virtual[n.node_id] < n.queue_capacity]
            if not feasible:
                result.append(None)
                continue
            # Upper layer narrows candidates to the least-loaded half; lower layer selects a node.
            ranked = sorted(feasible, key=lambda i: (nodes[i].load_ratio, -nodes[i].energy_ratio, nodes[i].node_id))
            shortlist = ranked[:max(1, (len(ranked) + 1) // 2)]
            action = (self._rng.choice(shortlist) if self._rng.random() < self.exploration else
                      min(shortlist, key=lambda i: (self.q_values.get((state, i), 0.0) - _score(nodes[i]), nodes[i].node_id)))
            result.append(nodes[action].node_id)
            virtual[nodes[action].node_id] += 1
            self._pending = (state, action)
        return result

    def observe_reward(self, reward: float, next_nodes: Sequence[EdgeNode]) -> None:
        """Update the most recent state/action estimate after a round observation."""
        if self._pending is None:
            return
        state, action = self._pending
        next_state = self._state(next_nodes)
        future = max((value for (s, _), value in self.q_values.items() if s == next_state), default=0.0)
        old = self.q_values.get((state, action), 0.0)
        self.q_values[(state, action)] = old + self.learning_rate * (reward + self.discount * future - old)
        self._pending = None
