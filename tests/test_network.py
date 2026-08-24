"""Behavioural tests for deterministic edge-network creation."""

from __future__ import annotations

from dataclasses import replace
import unittest

from config import NetworkConfig
from models import EdgeNode, Task, TaskStatus
from network import Network
from services import (
    BaselineScheduler,
    CongestionAnalyzer,
    CongestionAwareScheduler,
    EnergyAwareCongestionScheduler,
    LoadState,
)
from simulation import Simulator


class NetworkTests(unittest.TestCase):
    """Validate node creation, placement, and undirected connectivity."""

    def setUp(self) -> None:
        self.config = NetworkConfig()
        self.network = Network(self.config)

    def test_network_contains_exactly_ten_nodes(self) -> None:
        self.assertEqual(len(self.network.nodes), 10)

    def test_node_ids_are_unique(self) -> None:
        node_ids = [node.node_id for node in self.network.nodes]
        self.assertEqual(len(node_ids), len(set(node_ids)))

    def test_positions_are_within_network_area(self) -> None:
        for node in self.network.nodes:
            with self.subTest(node=node.node_id):
                self.assertGreaterEqual(node.x, 0.0)
                self.assertLessEqual(node.x, self.config.area_width)
                self.assertGreaterEqual(node.y, 0.0)
                self.assertLessEqual(node.y, self.config.area_height)

    def test_nodes_are_not_their_own_neighbours(self) -> None:
        for node in self.network.nodes:
            self.assertNotIn(node.node_id, node.neighbours)

    def test_neighbours_are_not_duplicated(self) -> None:
        for node in self.network.nodes:
            self.assertEqual(len(node.neighbours), len(set(node.neighbours)))

    def test_connectivity_is_symmetric(self) -> None:
        for node in self.network.nodes:
            for neighbour_id in node.neighbours:
                self.assertIn(node.node_id, self.network.get_node(neighbour_id).neighbours)

    def test_same_seed_reproduces_placement(self) -> None:
        duplicate = Network(self.config)
        self.assertEqual(
            [node.position for node in self.network.nodes],
            [node.position for node in duplicate.nodes],
        )

    def test_changing_seed_changes_placement(self) -> None:
        different_seed_network = Network(replace(self.config, seed=self.config.seed + 1))
        self.assertNotEqual(
            [node.position for node in self.network.nodes],
            [node.position for node in different_seed_network.nodes],
        )


class ConfigurationValidationTests(unittest.TestCase):
    """Validate failures for invalid foundational configuration."""

    def test_invalid_node_count_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            NetworkConfig(num_edge_nodes=0)

    def test_invalid_area_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            NetworkConfig(area_width=0.0)

    def test_invalid_communication_range_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            NetworkConfig(communication_range=-1.0)

    def test_energy_weights_must_sum_to_one(self) -> None:
        with self.assertRaises(ValueError):
            NetworkConfig(w_congestion=0.8, w_energy=0.8)

    def test_initial_edge_energy_alias_is_supported(self) -> None:
        config = NetworkConfig(initial_edge_energy=500.0)
        self.assertEqual(config.initial_energy, 500.0)


class SimulationTests(unittest.TestCase):
    """Validate baseline IoT task generation and edge queue processing."""

    def setUp(self) -> None:
        self.config = NetworkConfig(
            num_edge_nodes=1,
            num_iot_devices=2,
            num_rounds=1,
            task_generation_probability=1.0,
            processing_capacity=1,
            queue_capacity=10,
        )
        self.simulator = Simulator(self.config)
        self.report = self.simulator.run_round(1)

    def test_creates_exactly_twenty_iot_devices_by_default(self) -> None:
        self.assertEqual(len(Simulator(NetworkConfig()).devices), 20)

    def test_device_ids_are_unique(self) -> None:
        device_ids = [device.device_id for device in self.simulator.devices]
        self.assertEqual(len(device_ids), len(set(device_ids)))

    def test_tasks_are_generated_automatically(self) -> None:
        self.assertEqual(len(self.report.generated_tasks), self.config.num_iot_devices)

    def test_generated_tasks_have_valid_source_devices(self) -> None:
        device_ids = {device.device_id for device in self.simulator.devices}
        for task in self.report.generated_tasks:
            self.assertIn(task.source_device_id, device_ids)

    def test_assigned_edge_nodes_exist(self) -> None:
        edge_node_ids = {node.node_id for node in self.simulator.network.nodes}
        for task in self.report.generated_tasks:
            self.assertIn(task.assigned_edge_node_id, edge_node_ids)

    def test_tasks_enter_actual_edge_node_queues(self) -> None:
        edge_node = self.simulator.network.nodes[0]
        self.assertEqual(edge_node.received_tasks, self.config.num_iot_devices)
        self.assertTrue(all(isinstance(task, Task) for task in edge_node.queue))
        self.assertEqual(edge_node.queue_length, 1)

    def test_queue_count_matches_actual_queue_contents(self) -> None:
        edge_node = self.simulator.network.nodes[0]
        self.assertEqual(self.report.queue_sizes[edge_node.node_id], len(edge_node.queue))

    def test_processing_removes_tasks_and_updates_task_state(self) -> None:
        edge_node = self.simulator.network.nodes[0]
        self.assertEqual(len(self.report.processed_tasks), 1)
        self.assertEqual(edge_node.processed_tasks, 1)
        self.assertEqual(self.report.processed_tasks[0].status, TaskStatus.COMPLETED)
        self.assertEqual(edge_node.queue_length, self.config.num_iot_devices - 1)

    def test_assignment_snapshot_does_not_report_completion(self) -> None:
        queued_states = set(self.report.assignment_states.values())

        self.assertIn("QUEUED", queued_states)
        self.assertNotIn("COMPLETED", queued_states)
        self.assertTrue(all(task.status is TaskStatus.COMPLETED for task in self.report.processed_tasks))

    def test_task_lifecycle_transitions_match_queue_processing(self) -> None:
        task = Task(task_id="T-LIFECYCLE", source_device_id="D1", creation_round=1)
        node = EdgeNode(
            node_id="E1",
            x=0.0,
            y=0.0,
            processing_capacity=1,
            queue_capacity=1,
        )

        self.assertEqual(task.status, TaskStatus.GENERATED)
        task.assign_to(node.node_id)
        self.assertEqual(task.status, TaskStatus.ASSIGNED)
        self.assertTrue(node.enqueue_task(task))
        self.assertEqual(task.status, TaskStatus.QUEUED)
        self.assertEqual(node.process_queued_tasks(), (task,))
        self.assertEqual(task.status, TaskStatus.COMPLETED)

    def test_engaged_node_count_matches_queue_state(self) -> None:
        expected_count = sum(node.queue_length > 0 for node in self.simulator.network.nodes)
        self.assertEqual(self.report.engaged_node_count, expected_count)

    def test_same_seed_reproduces_task_generation_and_assignment(self) -> None:
        first = Simulator(NetworkConfig(task_generation_probability=1.0)).run_round(1)
        second = Simulator(NetworkConfig(task_generation_probability=1.0)).run_round(1)
        self.assertEqual(
            [(task.task_id, task.source_device_id, task.assigned_edge_node_id) for task in first.generated_tasks],
            [(task.task_id, task.source_device_id, task.assigned_edge_node_id) for task in second.generated_tasks],
        )


class ObservabilityTests(unittest.TestCase):
    """Validate queue snapshots, capacity boundaries, and round instrumentation."""

    def test_before_and_after_queue_observations_match_actual_processing(self) -> None:
        simulator = Simulator(
            NetworkConfig(
                num_edge_nodes=1,
                num_iot_devices=3,
                task_generation_probability=1.0,
                processing_capacity=1,
                queue_capacity=3,
            )
        )
        report = simulator.run_round(1)
        edge_node = simulator.network.nodes[0]

        self.assertEqual(report.queue_sizes_before_processing[edge_node.node_id], 3)
        self.assertEqual(report.processing_counts[edge_node.node_id], 1)
        self.assertEqual(report.queue_sizes_after_processing[edge_node.node_id], 2)
        self.assertEqual(report.queue_sizes_after_processing[edge_node.node_id], len(edge_node.queue))

    def test_processing_never_exceeds_node_capacity(self) -> None:
        simulator = Simulator(
            NetworkConfig(
                num_edge_nodes=1,
                num_iot_devices=5,
                task_generation_probability=1.0,
                processing_capacity=2,
                queue_capacity=5,
            )
        )
        report = simulator.run_round(1)
        edge_node = simulator.network.nodes[0]

        self.assertEqual(report.processing_counts[edge_node.node_id], edge_node.processing_capacity)
        self.assertLessEqual(len(report.processed_tasks), edge_node.processing_capacity)
        self.assertEqual(edge_node.current_processing_count, 0)

    def test_full_queue_drops_task_without_exceeding_capacity(self) -> None:
        simulator = Simulator(
            NetworkConfig(
                num_edge_nodes=1,
                num_iot_devices=3,
                task_generation_probability=1.0,
                processing_capacity=1,
                queue_capacity=2,
            )
        )
        report = simulator.run_round(1)
        edge_node = simulator.network.nodes[0]

        self.assertEqual(report.queue_sizes_before_processing[edge_node.node_id], 2)
        self.assertEqual(len(report.dropped_tasks), 1)
        self.assertEqual(report.dropped_tasks[0].status, TaskStatus.DROPPED)
        self.assertEqual(report.dropped_tasks[0].drop_reason, "queue_full")
        self.assertEqual(edge_node.dropped_tasks, 1)
        self.assertLessEqual(edge_node.queue_length, edge_node.queue_capacity)

    def test_remaining_queued_tasks_match_their_assigned_edge_node(self) -> None:
        simulator = Simulator(
            NetworkConfig(
                num_edge_nodes=1,
                num_iot_devices=3,
                task_generation_probability=1.0,
                processing_capacity=1,
                queue_capacity=3,
            )
        )
        simulator.run_round(1)
        edge_node = simulator.network.nodes[0]

        for task in edge_node.queue:
            self.assertEqual(task.assigned_edge_node_id, edge_node.node_id)
            self.assertEqual(task.status, TaskStatus.QUEUED)

    def test_utilization_and_load_summary_use_real_queue_lengths(self) -> None:
        simulator = Simulator(
            NetworkConfig(
                num_edge_nodes=1,
                num_iot_devices=3,
                task_generation_probability=1.0,
                processing_capacity=1,
                queue_capacity=3,
            )
        )
        report = simulator.run_round(1)
        edge_node = simulator.network.nodes[0]

        self.assertEqual(report.queue_utilizations_before_processing[edge_node.node_id], 1.0)
        self.assertEqual(report.load_classifications_before_processing[edge_node.node_id], "CONGESTED")
        self.assertEqual(report.metrics.maximum_queue, edge_node.queue_length)
        self.assertEqual(report.metrics.maximum_queue_utilization, edge_node.load_ratio)
        self.assertEqual(report.metrics.most_loaded_node, edge_node.node_id)

    def test_engaged_counts_match_pre_and_post_queue_snapshots(self) -> None:
        simulator = Simulator(
            NetworkConfig(
                num_edge_nodes=1,
                num_iot_devices=2,
                task_generation_probability=1.0,
                processing_capacity=1,
                queue_capacity=2,
            )
        )
        report = simulator.run_round(1)

        self.assertEqual(
            report.metrics.engaged_nodes_before,
            sum(size > 0 for size in report.queue_sizes_before_processing.values()),
        )
        self.assertEqual(
            report.metrics.engaged_nodes_after,
            sum(size > 0 for size in report.queue_sizes_after_processing.values()),
        )

    def test_history_records_every_round(self) -> None:
        simulator = Simulator(
            NetworkConfig(num_rounds=2, task_generation_probability=1.0)
        )
        reports = simulator.run()

        self.assertEqual(len(reports), 2)
        self.assertEqual([metric.round_number for metric in simulator.history], [1, 2])

    def test_task_accounting_identity_is_preserved(self) -> None:
        simulator = Simulator(
            NetworkConfig(
                num_edge_nodes=1,
                num_iot_devices=4,
                num_rounds=2,
                task_generation_probability=1.0,
                processing_capacity=1,
                queue_capacity=2,
            )
        )
        simulator.run()
        accounting = simulator.task_accounting
        self.assertEqual(
            accounting["generated"],
            accounting["completed"] + accounting["queued"] + accounting["dropped"],
        )
        self.assertEqual(accounting["assigned"], 0)
        self.assertEqual(accounting["processing"], 0)

    def test_energy_accounting_identity_is_preserved(self) -> None:
        simulator = Simulator(NetworkConfig(num_rounds=2, task_generation_probability=1.0))
        simulator.run()
        simulator.validate_energy_accounting()
        self.assertTrue(all(node.remaining_energy >= 0 for node in simulator.network.nodes))


class CongestionAnalysisTests(unittest.TestCase):
    """Validate queue-based congestion detection independently of allocation."""

    def setUp(self) -> None:
        self.analyzer = CongestionAnalyzer(0.50, 0.80)

    def test_low_load_is_classified_from_queue_utilization(self) -> None:
        result = self.analyzer.analyze_queue("E1", 5, 20)

        self.assertEqual(result.queue_utilization, 0.25)
        self.assertEqual(result.load_state, LoadState.LOW)

    def test_moderate_load_is_classified_from_queue_utilization(self) -> None:
        result = self.analyzer.analyze_queue("E1", 12, 20)

        self.assertEqual(result.queue_utilization, 0.60)
        self.assertEqual(result.load_state, LoadState.MODERATE)

    def test_congested_load_is_classified_at_the_threshold(self) -> None:
        result = self.analyzer.analyze_queue("E1", 16, 20)

        self.assertEqual(result.queue_utilization, 0.80)
        self.assertEqual(result.load_state, LoadState.CONGESTED)

    def test_full_queue_is_congested(self) -> None:
        result = self.analyzer.analyze_queue("E1", 20, 20)

        self.assertEqual(result.queue_utilization, 1.0)
        self.assertEqual(result.load_state, LoadState.CONGESTED)

    def test_invalid_queue_observations_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.analyzer.analyze_queue("E1", -1, 20)
        with self.assertRaises(ValueError):
            self.analyzer.analyze_queue("E1", 21, 20)
        with self.assertRaises(ValueError):
            self.analyzer.analyze_queue("E1", 0, 0)

    def test_congested_count_ratio_and_aggregate_utilization_are_correct(self) -> None:
        nodes = (
            self._node_with_queue("E1", 5),
            self._node_with_queue("E2", 12),
            self._node_with_queue("E3", 16),
        )
        report = self.analyzer.analyze(nodes)

        self.assertEqual(report.congested_node_count, 1)
        self.assertEqual(report.congestion_ratio, 1 / 3)
        self.assertEqual(report.most_loaded_node.edge_node_id, "E3")
        self.assertEqual(report.average_queue_utilization, (0.25 + 0.60 + 0.80) / 3)
        self.assertEqual(report.maximum_queue_utilization, 0.80)
        self.assertEqual(report.minimum_queue_utilization, 0.25)

    def test_processing_can_reduce_the_congestion_status(self) -> None:
        simulator = Simulator(
            NetworkConfig(
                num_edge_nodes=1,
                num_iot_devices=3,
                task_generation_probability=1.0,
                processing_capacity=2,
                queue_capacity=3,
            )
        )
        report = simulator.run_round(1)

        self.assertEqual(report.congestion_before_processing.congested_node_count, 1)
        self.assertEqual(report.congestion_after_processing.congested_node_count, 0)
        self.assertEqual(
            report.congestion_before_processing.most_loaded_node.load_state,
            LoadState.CONGESTED,
        )
        self.assertEqual(
            report.congestion_after_processing.most_loaded_node.load_state,
            LoadState.LOW,
        )

    def test_stress_configuration_is_valid_and_reproducible(self) -> None:
        config = NetworkConfig.stress_test()
        first = Simulator(config)
        second = Simulator(config)
        first_reports = first.run()
        second_reports = second.run()

        self.assertEqual(config.queue_capacity, 10)
        self.assertEqual(config.processing_capacity, 1)
        self.assertEqual(config.task_generation_probability, 0.80)
        self.assertEqual(config.num_rounds, 20)
        self.assertTrue(any(report.metrics.congested_node_count_before > 0 for report in first_reports))
        self.assertGreaterEqual(first.peak_queue_utilization, config.congestion_threshold)
        self.assertGreater(first.peak_congestion_ratio, 0.0)
        self.assertEqual(first.peak_queue_utilization, second.peak_queue_utilization)
        self.assertEqual(first.peak_congestion_ratio, second.peak_congestion_ratio)
        self.assertEqual(first.maximum_congested_nodes, second.maximum_congested_nodes)

    def test_peak_congestion_metrics_use_pre_processing_observations(self) -> None:
        simulator = Simulator(NetworkConfig.stress_test())
        simulator.run()
        self.assertEqual(
            simulator.peak_congestion_ratio,
            max(metric.congestion_ratio_before for metric in simulator.history),
        )
        self.assertEqual(
            simulator.peak_congested_nodes,
            max(metric.congested_node_count_before for metric in simulator.history),
        )
        self.assertEqual(
            simulator.total_congested_node_rounds,
            sum(metric.congested_node_count_before for metric in simulator.history),
        )

    @staticmethod
    def _node_with_queue(node_id: str, queue_size: int):
        node = EdgeNode(
            node_id=node_id,
            x=0.0,
            y=0.0,
            processing_capacity=1,
            queue_capacity=20,
        )
        node.queue.extend(
            Task(task_id=f"{node_id}T{index}", source_device_id="D1", creation_round=1)
            for index in range(queue_size)
        )
        return node


class SchedulerTests(unittest.TestCase):
    """Validate congestion-aware placement against controlled queue states."""

    def test_congested_baseline_is_redirected_to_least_loaded_alternative(self) -> None:
        config = NetworkConfig(num_edge_nodes=4, queue_capacity=10, processing_capacity=1)
        simulator = Simulator(config)
        self._position_nodes(simulator, (0.0, 100.0, 200.0, 300.0))
        self._fill_node(simulator.network.nodes[0], 10)
        self._fill_node(simulator.network.nodes[1], 2)
        self._fill_node(simulator.network.nodes[2], 5)

        task = Task(task_id="T-REDIRECT", source_device_id="D1", creation_round=1)
        source = simulator.devices[0]
        source.x = 0.0
        source.y = 0.0
        decision = CongestionAwareScheduler(config.congestion_threshold).assign_task(
            task, source, simulator.network
        )

        self.assertEqual(decision.baseline_candidate_node_id, "E1")
        self.assertEqual(decision.actual_selected_node_id, "E4")
        self.assertTrue(decision.redirected)
        self.assertEqual(task.assigned_edge_node_id, "E4")
        self.assertIn(task, simulator.network.get_node("E4").queue)
        self.assertLess(simulator.network.get_node("E4").queue_length, 10)

    def test_non_congested_baseline_is_not_unnecessarily_redirected(self) -> None:
        config = NetworkConfig(num_edge_nodes=3, queue_capacity=10)
        simulator = Simulator(config)
        self._position_nodes(simulator, (0.0, 100.0, 200.0))
        self._fill_node(simulator.network.nodes[0], 2)

        task = Task(task_id="T-KEEP", source_device_id="D1", creation_round=1)
        source = simulator.devices[0]
        source.x = 0.0
        source.y = 0.0
        decision = CongestionAwareScheduler(config.congestion_threshold).assign_task(
            task, source, simulator.network
        )

        self.assertEqual(decision.baseline_candidate_node_id, "E1")
        self.assertEqual(decision.actual_selected_node_id, "E1")
        self.assertFalse(decision.redirected)

    def test_all_full_nodes_drop_without_overflow(self) -> None:
        config = NetworkConfig(num_edge_nodes=3, queue_capacity=2)
        simulator = Simulator(config)
        self._position_nodes(simulator, (0.0, 100.0, 200.0))
        for node in simulator.network.nodes:
            self._fill_node(node, node.queue_capacity)

        task = Task(task_id="T-DROP", source_device_id="D1", creation_round=1)
        decision = CongestionAwareScheduler(config.congestion_threshold).assign_task(
            task, simulator.devices[0], simulator.network
        )

        self.assertFalse(decision.redirected)
        self.assertEqual(task.status, TaskStatus.DROPPED)
        self.assertTrue(all(node.queue_length <= node.queue_capacity for node in simulator.network.nodes))

    def test_congestion_aware_scheduler_skips_dead_baseline_node(self) -> None:
        config = NetworkConfig(num_edge_nodes=2, queue_capacity=5)
        simulator = Simulator(config)
        self._position_nodes(simulator, (0.0, 100.0))
        simulator.network.nodes[0].consume_energy(1000.0)
        task = Task(task_id="T-DEAD-REDIRECT", source_device_id="D1", creation_round=1)
        source = simulator.devices[0]
        source.x = source.y = 0.0
        decision = CongestionAwareScheduler(config.congestion_threshold).assign_task(
            task, source, simulator.network
        )
        self.assertEqual(decision.actual_selected_node_id, "E2")
        self.assertIn(task, simulator.network.nodes[1].queue)

    def test_redirected_tasks_are_distributed_by_updated_utilization(self) -> None:
        config = NetworkConfig(num_edge_nodes=3, queue_capacity=10)
        simulator = Simulator(config)
        self._position_nodes(simulator, (0.0, 100.0, 200.0))
        self._fill_node(simulator.network.nodes[0], 10)
        scheduler = CongestionAwareScheduler(config.congestion_threshold)
        source = simulator.devices[0]
        source.x = source.y = 0.0

        decisions = []
        for index in range(2):
            task = Task(task_id=f"T-DIST-{index}", source_device_id="D1", creation_round=1)
            decisions.append(scheduler.assign_task(task, source, simulator.network))

        self.assertEqual([decision.actual_selected_node_id for decision in decisions], ["E2", "E3"])

    def test_baseline_and_aware_receive_identical_workload(self) -> None:
        config = NetworkConfig.stress_test()
        baseline = Simulator(config)
        aware = Simulator(config)
        baseline_reports = baseline.run(BaselineScheduler)
        aware_reports = aware.run(CongestionAwareScheduler)

        self.assertEqual(
            [task.task_id for task in baseline.all_tasks],
            [task.task_id for task in aware.all_tasks],
        )
        self.assertEqual(
            [task.source_device_id for task in baseline.all_tasks],
            [task.source_device_id for task in aware.all_tasks],
        )
        self.assertEqual(len(baseline_reports), len(aware_reports))
        self.assertGreater(sum(metric.redirected_tasks for metric in aware.history), 0)

    @staticmethod
    def _position_nodes(simulator: Simulator, xs: tuple[float, ...]) -> None:
        for node, x in zip(simulator.network.nodes, xs):
            node.x = x
            node.y = 0.0

    @staticmethod
    def _fill_node(node: EdgeNode, count: int) -> None:
        for index in range(count):
            task = Task(
                task_id=f"{node.node_id}-FILL-{index}",
                source_device_id="D0",
                creation_round=1,
            )
            task.assign_to(node.node_id)
            if not node.enqueue_task(task):
                raise AssertionError("controlled fixture exceeded node capacity")


class EnergyModelTests(unittest.TestCase):
    """Validate processing, idle, failure, and energy metric accounting."""

    def test_processing_consumes_configured_energy_once(self) -> None:
        node = EdgeNode(
            node_id="E1", x=0.0, y=0.0, processing_capacity=1, queue_capacity=2, initial_energy=10.0
        )
        task = Task(task_id="T-ENERGY", source_device_id="D1", creation_round=1)
        task.assign_to(node.node_id)
        node.enqueue_task(task)

        self.assertEqual(len(node.process_queued_tasks(2.0)), 1)
        self.assertEqual(node.remaining_energy, 8.0)
        self.assertEqual(node.consumed_energy, 2.0)

    def test_idle_energy_is_charged_only_to_active_idle_node(self) -> None:
        node = EdgeNode(
            node_id="E1", x=0.0, y=0.0, processing_capacity=1, queue_capacity=2, initial_energy=10.0
        )

        self.assertTrue(node.consume_idle_energy(0.5))
        self.assertEqual(node.remaining_energy, 9.5)
        node.consume_energy(9.5)
        self.assertFalse(node.is_active)
        self.assertFalse(node.consume_idle_energy(0.5))
        self.assertEqual(node.remaining_energy, 0.0)

    def test_processing_cannot_make_energy_negative(self) -> None:
        node = EdgeNode(
            node_id="E1", x=0.0, y=0.0, processing_capacity=1, queue_capacity=2, initial_energy=1.0
        )
        task = Task(task_id="T-NO-ENERGY", source_device_id="D1", creation_round=1)
        task.assign_to(node.node_id)
        node.enqueue_task(task)

        self.assertEqual(node.process_queued_tasks(2.0), ())
        self.assertEqual(node.remaining_energy, 1.0)
        self.assertEqual(task.status, TaskStatus.QUEUED)

    def test_zero_energy_node_cannot_receive_new_task(self) -> None:
        node = EdgeNode(
            node_id="E1", x=0.0, y=0.0, processing_capacity=1, queue_capacity=2,
            initial_energy=10.0, remaining_energy=0.0,
        )
        task = Task(task_id="T-DEAD", source_device_id="D1", creation_round=1)
        task.assign_to(node.node_id)

        self.assertFalse(node.enqueue_task(task))
        self.assertEqual(task.status, TaskStatus.DROPPED)
        self.assertEqual(task.drop_reason, "inactive_node")
        self.assertFalse(node.is_active)

    def test_energy_utilization_is_normalized(self) -> None:
        node = EdgeNode(
            node_id="E1", x=0.0, y=0.0, processing_capacity=1, queue_capacity=2,
            initial_energy=100.0, remaining_energy=25.0,
        )

        self.assertEqual(node.energy_utilization, 0.75)

    def test_energy_metrics_and_zero_completed_division(self) -> None:
        config = NetworkConfig(
            num_edge_nodes=1,
            num_iot_devices=1,
            task_generation_probability=0.0,
            num_rounds=1,
            idle_energy_per_round=0.5,
            processing_energy_per_task=2.0,
        )
        simulator = Simulator(config)
        report = simulator.run_round(1)

        self.assertEqual(report.metrics.total_energy_consumed, 0.5)
        self.assertEqual(report.metrics.energy_consumed_this_round, 0.5)
        self.assertIsNone(report.metrics.energy_consumed_per_completed_task)

    def test_energy_aware_scheduler_uses_normalized_score(self) -> None:
        config = NetworkConfig(num_edge_nodes=2, queue_capacity=10)
        simulator = Simulator(config)
        SchedulerTests._position_nodes(simulator, (0.0, 100.0))
        source = simulator.devices[0]
        source.x = source.y = 0.0
        SchedulerTests._fill_node(simulator.network.nodes[0], 6)
        SchedulerTests._fill_node(simulator.network.nodes[1], 2)
        simulator.network.nodes[0].remaining_energy = 90.0
        simulator.network.nodes[1].remaining_energy = 30.0
        scheduler = EnergyAwareCongestionScheduler(
            congestion_threshold=0.80,
            w_congestion=0.70,
            w_energy=0.30,
        )
        task = Task(task_id="T-SCORE", source_device_id="D1", creation_round=1)
        decision = scheduler.assign_task(task, source, simulator.network)

        self.assertEqual(decision.actual_selected_node_id, "E2")
        self.assertIn(task, simulator.network.get_node("E2").queue)

    def test_energy_aware_scheduler_excludes_low_energy_nodes(self) -> None:
        config = NetworkConfig(num_edge_nodes=2, queue_capacity=10)
        simulator = Simulator(config)
        SchedulerTests._position_nodes(simulator, (0.0, 100.0))
        source = simulator.devices[0]
        source.x = source.y = 0.0
        simulator.network.nodes[0].remaining_energy = 0.5
        simulator.network.nodes[0].is_active = True
        task = Task(task_id="T-MIN-ENERGY", source_device_id="D1", creation_round=1)
        decision = EnergyAwareCongestionScheduler(min_operational_energy=1.0).assign_task(
            task, source, simulator.network
        )

        self.assertEqual(decision.actual_selected_node_id, "E2")

    def test_all_three_schedulers_use_the_same_generated_workload(self) -> None:
        config = NetworkConfig.stress_test()
        simulators = [Simulator(config) for _ in range(3)]
        simulators[0].run(BaselineScheduler)
        simulators[1].run(CongestionAwareScheduler)
        simulators[2].run(EnergyAwareCongestionScheduler)

        workload = [
            [(task.task_id, task.source_device_id, task.creation_round) for task in simulator.all_tasks]
            for simulator in simulators
        ]
        self.assertEqual(workload[0], workload[1])
        self.assertEqual(workload[1], workload[2])
