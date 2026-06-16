"""Analyze trajectories from static_vs_learning_tube_mpc_25x2.npz and output a LaTeX table.

This script reads the .npz bundle and generates summary statistics for:
- Nominal MPC
- Static DT-MPC (NT-MPC)
- Learning DT-MPC (DT-MPC)

Output includes success rate, violation rate, trajectory length, and average time per step.
"""

import argparse
import os

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_bundle(data_path):
    """Load the experiment bundle from .npz file."""
    bundle = np.load(data_path, allow_pickle=True)
    return {
        "nominal_results": bundle["nominal_results"].tolist() if "nominal_results" in bundle else [],
        "static_results": bundle["static_results"].tolist() if "static_results" in bundle else [],
        "learning_results": bundle["learning_results"].tolist() if "learning_results" in bundle else [],
        "obstacles": np.asarray(bundle["obstacles"], dtype=float),
        "goal_state": np.asarray(bundle["goal_state"], dtype=float),
        "start_state": np.asarray(bundle["start_state"], dtype=float),
        "settings": bundle["settings"].item() if "settings" in bundle else {},
    }


def analyze_results(results, controller_name):
    """Compute statistics for a set of results."""
    if not results:
        return {
            "controller": controller_name,
            "num_runs": 0,
            "success_rate": 0.0,
            "violation_rate": 0.0,
            "avg_trajectory_length": 0.0,
            "avg_time_per_step": 0.0,
        }

    total_runs = len(results)
    success_count = sum(1 for res in results if res.get("status") == "GOAL_REACHED")
    violation_count = sum(1 for res in results if res.get("violated", False))

    # Trajectory length: number of steps taken
    trajectory_lengths = [res.get("steps_taken", 0) for res in results]
    avg_trajectory_length = float(np.mean(trajectory_lengths)) if trajectory_lengths else 0.0

    # Time per step: average across all runs
    step_times = []
    for res in results:
        step_time_array = res.get("step_time_s", [])
        if isinstance(step_time_array, np.ndarray) and len(step_time_array) > 0:
            step_times.append(float(np.mean(step_time_array)))
        elif hasattr(step_time_array, '__len__') and len(step_time_array) > 0:
            step_times.append(float(np.mean(step_time_array)))

    avg_time_per_step = float(np.mean(step_times)) if step_times else 0.0

    return {
        "controller": controller_name,
        "num_runs": total_runs,
        "success_rate": success_count / total_runs,
        "violation_rate": violation_count / total_runs,
        "avg_trajectory_length": avg_trajectory_length,
        "avg_time_per_step": avg_time_per_step,
    }


def format_latex_table(stats_list):
    """Generate a LaTeX table from statistics."""
    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"    \centering")
    lines.append(r"    \begin{tabular}{l c c c c}")
    lines.append(r"        \hline")
    lines.append(r"        Controller & Success & Violation & Traj. Length & Time/Step \\")
    lines.append(r"        & Rate & Rate & (steps) & (s) \\")
    lines.append(r"        \hline")

    for stats in stats_list:
        controller = stats["controller"]
        success = stats["success_rate"]
        violation = stats["violation_rate"]
        traj_len = stats["avg_trajectory_length"]
        time_per_step = stats["avg_time_per_step"]

        line = (
            f"        {controller} & "
            f"{success:.2%} & "
            f"{violation:.2%} & "
            f"{traj_len:.1f} & "
            f"{time_per_step:.4f} \\\\"
        )
        lines.append(line)

    lines.append(r"        \hline")
    lines.append(r"    \end{tabular}")
    lines.append(r"    \caption{Controller Performance Summary}")
    lines.append(r"    \label{tab:controller_performance}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze DT-MPC experiment data and generate LaTeX table"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=os.path.join(SCRIPT_DIR, "results", "static_vs_learning_tube_mpc_25x2.npz"),
        help="Path to the .npz bundle",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="File to save LaTeX table (if None, prints to stdout)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Data file not found: {args.data}")

    bundle = load_bundle(args.data)

    # Analyze each controller type
    nominal_stats = analyze_results(bundle["nominal_results"], "Nominal MPC")
    static_stats = analyze_results(bundle["static_results"], "NT-MPC")
    learning_stats = analyze_results(bundle["learning_results"], "DT-MPC")

    # Generate LaTeX table
    latex_table = format_latex_table([nominal_stats, static_stats, learning_stats])

    # Output
    if args.output:
        with open(args.output, "w") as f:
            f.write(latex_table)
        print(f"LaTeX table saved to {args.output}")
    else:
        print(latex_table)

    # Print summary to stdout
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    for stats in [nominal_stats, static_stats, learning_stats]:
        print(f"\n{stats['controller']}:")
        print(f"  Runs: {stats['num_runs']}")
        print(f"  Success Rate: {stats['success_rate']:.2%}")
        print(f"  Violation Rate: {stats['violation_rate']:.2%}")
        print(f"  Avg Trajectory Length: {stats['avg_trajectory_length']:.1f} steps")
        print(f"  Avg Time per Step: {stats['avg_time_per_step']:.4f} s")


if __name__ == "__main__":
    main()
