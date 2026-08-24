"""Run the normal simulation or the reproducible congestion stress test."""

import sys

from config import NetworkConfig
from experiments import ExperimentRunner, ResultProcessor
from services import CongestionAwareScheduler, CongestionReport, EnergyAwareCongestionScheduler
from simulation import RoundReport, Simulator


def main(mode: str | None = None) -> None:
    """Run normal rounds or the fixed high-load stress configuration."""
    arguments = set(sys.argv[1:])
    if "--experiment-all" in arguments:
        _run_experiment("all")
        return
    if "--experiment-normal" in arguments:
        _run_experiment("normal")
        return
    if "--experiment-stress" in arguments:
        _run_experiment("stress")
        return
    if "--scalability" in arguments:
        _run_experiment("scalability")
        return
    selected_mode = mode or (
        "compare" if "--compare" in arguments else
        "energy" if "--energy" in arguments else
        "aware" if "--aware" in arguments else
        "stress" if "--stress" in arguments else "normal"
    )
    if selected_mode == "compare":
        _run_comparison()
        return
    if selected_mode not in {"normal", "stress", "aware", "energy"}:
        raise ValueError("mode must be 'normal', 'stress', 'aware', 'energy', or 'compare'")
    config = NetworkConfig.stress_test() if selected_mode in {"stress", "aware", "energy"} else NetworkConfig()
    simulator = Simulator(config)

    title = (
        "ENERGY-AWARE CONGESTION SIMULATION" if selected_mode == "energy"
        else "CONGESTION-AWARE SIMULATION" if selected_mode == "aware"
        else "CONGESTION STRESS TEST" if selected_mode == "stress"
        else "NORMAL SIMULATION"
    )
    print(title)
    print("=" * len(title))
    print(f"Edge Nodes: {len(simulator.network.nodes)}")
    print(f"IoT Devices: {len(simulator.devices)}")
    print(f"Area: {config.area_width:g} x {config.area_height:g}")
    print(f"Simulation rounds: {config.num_rounds}")
    scheduler = (
        EnergyAwareCongestionScheduler(
            config.congestion_threshold,
            config.w_congestion,
            config.w_energy,
            config.min_operational_energy,
        )
        if selected_mode == "energy"
        else CongestionAwareScheduler(config.congestion_threshold)
        if selected_mode == "aware"
        else None
    )
    print(f"Scheduler: {scheduler.name if scheduler else 'Baseline'}")

    for report in simulator.run(scheduler):
        _print_round_report(
            report,
            len(simulator.network.nodes),
            simulator.network.nodes if selected_mode == "energy" else (),
        )

    if selected_mode in {"stress", "aware", "energy"}:
        print("\nStress-Test Results")
        print("------------------")
        print(f"Peak Queue Utilization: {simulator.peak_queue_utilization:.1%}")
        print(f"Peak Congestion Ratio: {simulator.peak_congestion_ratio:.1%}")
        print(f"Peak Congested Nodes: {simulator.peak_congested_nodes}")
        print(f"Total Congested Node-Rounds: {simulator.total_congested_node_rounds}")


def _print_round_report(report: RoundReport, edge_node_count: int, energy_nodes=()) -> None:
    """Print lifecycle, queue, load, processing, and metrics observations."""
    print(f"\n{'=' * 52}\nROUND {report.round_number}\n{'=' * 52}")
    print(f"Generated Tasks: {len(report.generated_tasks)}")
    print("\nTask Assignment:")
    if report.generated_tasks:
        for task in report.generated_tasks:
            print(
                f"{task.task_id} -> EdgeNode {task.assigned_edge_node_id} "
                f"[{report.assignment_states[task.task_id]}]"
            )
    else:
        print("No tasks generated")

    _print_queue_status("Queue Status (before processing):", report.queue_sizes_before_processing)

    _print_congestion_status("Congestion Status (before processing):", report.congestion_before_processing)

    print("\nProcessing:")
    processing_nodes = [
        (node_id, count)
        for node_id, count in report.processing_counts.items()
        if count > 0
    ]
    if processing_nodes:
        for node_id, count in processing_nodes:
            noun = "task" if count == 1 else "tasks"
            task_labels = ", ".join(
                f"{task.task_id} [COMPLETED]"
                for task in report.processed_tasks_by_node[node_id]
            )
            print(f"{node_id} -> {count} {noun}: {task_labels}")
    else:
        print("No tasks processed")

    _print_queue_status("Queue Status (after processing):", report.queue_sizes_after_processing)
    _print_congestion_status(
        "Congestion Status (after processing):", report.congestion_after_processing
    )

    metrics = report.metrics
    print("\nRound Summary")
    print("-------------")
    print(f"Generated This Round: {metrics.generated_this_round}")
    print(f"Processed This Round: {metrics.processed_this_round}")
    print(f"Total Generated: {metrics.total_generated}")
    print(f"Total Completed: {metrics.total_completed}")
    print(f"Currently Queued: {metrics.total_queued}")
    print(f"Engaged Before Processing: {metrics.engaged_nodes_before} / {edge_node_count}")
    print(f"Engaged After Processing: {metrics.engaged_nodes_after} / {edge_node_count}")
    print(f"Average Queue Utilization: {metrics.average_queue_utilization:.1%}")
    print(f"Maximum Queue Utilization: {metrics.maximum_queue_utilization:.1%}")
    print(f"Most Loaded Edge Node: {metrics.most_loaded_node}")
    print(f"Congested Nodes: {metrics.congested_node_count} / {edge_node_count}")
    print(f"Congestion Ratio: {metrics.congestion_ratio:.1%}")
    print(f"Energy Consumed This Round: {metrics.energy_consumed_this_round:.2f}")
    print(f"Total Energy Consumed: {metrics.total_energy_consumed:.2f}")
    print(f"Average Remaining Energy: {metrics.average_node_energy:.2f}")
    print(f"Minimum Remaining Energy: {metrics.minimum_node_energy:.2f}")
    if energy_nodes:
        print("\nEnergy Status:")
        for node in energy_nodes:
            state = "ACTIVE" if node.is_active else "DEAD"
            print(
                f"{node.node_id} -> {node.remaining_energy:.2f} / "
                f"{node.initial_energy:.2f} ({state})"
            )
    redirects = [decision for decision in report.scheduling_decisions if decision.redirected]
    if redirects:
        print("\nRedirections:")
        for decision in redirects:
            print(
                f"{decision.task_id}: {decision.baseline_candidate_node_id} -> "
                f"{decision.actual_selected_node_id} "
                f"({decision.baseline_utilization:.0%} -> {decision.selected_utilization:.0%}; "
                f"{decision.reason})"
            )


def _print_queue_status(title: str, queue_sizes: dict[str, int]) -> None:
    """Print real queue lengths from one observation point."""
    print(f"\n{title}")
    for node_id, queue_size in queue_sizes.items():
        noun = "task" if queue_size == 1 else "tasks"
        print(f"{node_id} -> {queue_size} {noun}")


def _print_congestion_status(title: str, congestion: CongestionReport) -> None:
    """Print measured node load and aggregate congestion without changing policy."""
    print(f"\n{title}")
    for result in congestion.node_results:
        print(
            f"{result.edge_node_id} -> {result.current_queue_size} / {result.queue_capacity} "
            f"tasks -> {result.queue_utilization:.1%} -> {result.load_state.value}"
        )

    most_loaded = congestion.most_loaded_node
    most_loaded_id = most_loaded.edge_node_id if most_loaded is not None else "none"
    print(f"Congested Nodes: {congestion.congested_node_count}")
    print(f"Congestion Ratio: {congestion.congestion_ratio:.1%}")
    print(f"Most Loaded Node: {most_loaded_id}")
    print(f"Maximum Utilization: {congestion.maximum_queue_utilization:.1%}")
    print(f"Average Utilization: {congestion.average_queue_utilization:.1%}")


def _run_comparison() -> None:
    """Run both policies on independent, identical deterministic workloads."""
    config = NetworkConfig.stress_test()
    baseline = Simulator(config)
    aware = Simulator(config)
    energy_aware = Simulator(config)
    baseline_reports = baseline.run()
    aware_reports = aware.run(CongestionAwareScheduler(config.congestion_threshold))
    energy_reports = energy_aware.run(
        EnergyAwareCongestionScheduler(
            config.congestion_threshold,
            config.w_congestion,
            config.w_energy,
            config.min_operational_energy,
        )
    )
    baseline_summary = _summary(baseline)
    aware_summary = _summary(aware)
    energy_summary = _summary(energy_aware)

    print("BASELINE VS CONGESTION-AWARE VS ENERGY-AWARE")
    print("============================================")
    print("Same stress configuration and seed used for all schedulers.")
    print("\nMetric                         Baseline    Congestion-Aware    Energy-Aware")
    print("-------------------------------------------------------------------------")
    for label in baseline_summary:
        print(
            f"{label:<30} {baseline_summary[label]:>9}    "
            f"{aware_summary[label]:>16}    {energy_summary[label]:>12}"
        )
    print(f"\nRounds compared: {len(baseline_reports)}")


def _summary(simulator: Simulator) -> dict[str, str]:
    """Format measured comparison metrics without claiming improvement."""
    final = simulator.history[-1]
    redirected = sum(metric.redirected_tasks for metric in simulator.history)
    return {
        "Generated": str(final.total_generated),
        "Completed": str(final.total_completed),
        "Dropped": str(final.total_dropped),
        "Peak Queue Utilization": f"{simulator.peak_queue_utilization:.1%}",
        "Peak Congestion Ratio": f"{simulator.peak_congestion_ratio:.1%}",
        "Average Congestion Ratio": f"{sum(m.congestion_ratio_before for m in simulator.history) / len(simulator.history):.1%}",
        "Average Queue Utilization": f"{sum(m.average_queue_utilization_before for m in simulator.history) / len(simulator.history):.1%}",
        "Peak Congested Nodes": str(simulator.peak_congested_nodes),
        "Redirected Tasks": str(redirected),
        "Redirection Rate": f"{redirected / final.total_generated:.1%}" if final.total_generated else "0.0%",
        "Total Energy Consumed": f"{final.total_energy_consumed:.2f}",
        "Energy / Completed Task": (
            f"{final.total_energy_consumed / final.total_completed:.2f}"
            if final.total_completed else "n/a"
        ),
        "Average Remaining Energy": f"{final.average_node_energy:.2f}",
        "Minimum Remaining Energy": f"{final.minimum_node_energy:.2f}",
    }


def _run_experiment(kind: str) -> None:
    """Run evaluation matrices, persist raw/aggregate results, and make plots."""
    if kind not in {"normal", "stress", "scalability", "all"}:
        raise ValueError("unknown experiment kind")
    config = NetworkConfig()
    runner = ExperimentRunner(config)
    processor = ResultProcessor(config.results_directory)
    modes = ("normal", "stress", "scalability") if kind == "all" else (kind,)
    for workload_mode in modes:
        records = (
            runner.run_scalability()
            if workload_mode == "scalability"
            else runner.run(workload_mode)
        )
        paths = processor.write_results(records, workload_mode)
        graphs = processor.generate_graphs(records, workload_mode)
        aggregated = processor.aggregate(records)
        print("\n" + "=" * 60)
        print(f"EXPERIMENT SUMMARY: {workload_mode.upper()}")
        print("=" * 60)
        print(f"Runs: {len({record['seed'] for record in records})}")
        print(f"Schedulers: {len({record['scheduler'] for record in records})}")
        print(f"Raw records: {len(records)}")
        for row in processor.comparison_table(aggregated):
            print(" | ".join(str(value) for value in row.values()))
        if workload_mode == "scalability":
            print("\nScalability (devices | scheduler | peak congestion | energy | dropped | runtime)")
            for row in aggregated:
                print(
                    f"{row['num_iot_devices']} | {row['scheduler']} | "
                    f"{row['peak_congestion_ratio_mean']:.3f} | "
                    f"{row['total_energy_consumed_mean']:.2f} | "
                    f"{row['total_dropped_mean']:.2f} | "
                    f"{row['runtime_seconds_mean']:.4f}s"
                )
        print(f"Results: {paths['raw_json'].parent}")
        print(f"Graphs generated: {len(graphs)}")


if __name__ == "__main__":
    main()
