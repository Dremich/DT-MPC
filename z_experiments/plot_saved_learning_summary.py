"""Plot saved DT-MPC learning experiments from a serialized data bundle.

This script reads the .npz bundle written by
z_experiments/run_static_vs_learning_tube_mpc.py and produces:

- an overlay of selected learning trajectories
- a learning summary figure with alpha, gamma, gradients, and loss

Use ``--indices`` to choose specific learning runs or ``--max-runs`` to plot
the first few runs in the bundle.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from matplotlib.lines import Line2D


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")


def load_bundle(data_path):
    bundle = np.load(data_path, allow_pickle=True)
    return {
        "static_results": bundle["static_results"].tolist(),
        "learning_results": bundle["learning_results"].tolist(),
        "obstacles": np.asarray(bundle["obstacles"], dtype=float),
        "goal_state": np.asarray(bundle["goal_state"], dtype=float),
        "start_state": np.asarray(bundle["start_state"], dtype=float),
        "settings": bundle["settings"].item() if "settings" in bundle else {},
    }


def parse_indices(indices_text, max_runs, available_count):
    if indices_text:
        indices = [int(item.strip()) for item in indices_text.split(",") if item.strip()]
    else:
        indices = list(range(min(max_runs, available_count)))

    return [index for index in indices if 0 <= index < available_count]


def plot_trajectories(bundle, selected_indices, out_path):
    learning_results = bundle["learning_results"]
    obstacles = bundle["obstacles"]
    goal_state = bundle["goal_state"]
    start_state = bundle["start_state"]

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_aspect("equal")

    cmap = plt.get_cmap("tab10")
    for color_index, result_index in enumerate(selected_indices):
        result = learning_results[result_index]
        traj = np.asarray(result["states"], dtype=float)
        label = f"Run {result_index}"
        ax.plot(traj[:, 0], traj[:, 1], color=cmap(color_index % 10), linewidth=1.7, alpha=0.9, label=label)

        nominal = np.asarray(result.get("nominal_states", []), dtype=float)
        if nominal.size:
            ax.plot(
                nominal[:, 0],
                nominal[:, 1],
                color=cmap(color_index % 10),
                linestyle="--",
                linewidth=1.0,
                alpha=0.6,
            )

    for obs in obstacles:
        ax.add_patch(Circle((obs[0], obs[1]), obs[2], color="red", alpha=0.3))

    ax.plot(start_state[0], start_state[1], marker="o", color="black", markersize=7, label="Start")
    ax.plot(goal_state[0], goal_state[1], marker="*", color="green", markersize=12, label="Goal")
    ax.add_patch(Circle((goal_state[0], goal_state[1]), 0.5, color="green", alpha=0.18))

    ax.set_title("Selected Learning Trajectories")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_learning_summary(bundle, selected_indices, out_path):
    learning_results = bundle["learning_results"]

    fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=False)
    cmap = plt.get_cmap("tab10")

    value_ax = axes[0]
    value_gamma_ax = value_ax.twinx()
    grad_ax = axes[1]
    grad_gamma_ax = grad_ax.twinx()

    for color_index, result_index in enumerate(selected_indices):
        result = learning_results[result_index]
        color = cmap(color_index % 10)
        steps = np.arange(len(result["alpha"]))
        alpha = np.asarray(result["alpha"], dtype=float)
        gamma = np.asarray(result["gamma"], dtype=float)
        loss = np.asarray(result["loss"], dtype=float)
        grad_theta = np.asarray(result["grad_theta"], dtype=float)

        value_ax.plot(steps, alpha, color=color, linewidth=1.6, alpha=0.9)
        value_gamma_ax.plot(steps, gamma, color=color, linestyle="--", linewidth=1.2, alpha=0.9)

        grad_steps = np.arange(len(grad_theta))
        if grad_theta.size:
            grad_ax.plot(grad_steps, grad_theta[:, 0], color=color, linewidth=1.6, alpha=0.9)
            grad_gamma_ax.plot(grad_steps, grad_theta[:, 1], color=color, linestyle="--", linewidth=1.2, alpha=0.9)

        axes[2].plot(np.arange(len(loss)), loss, color=color, linewidth=1.6, alpha=0.9)

    value_ax.set_title("Barrier Parameters")
    value_ax.set_xlabel("Closed-loop step")
    value_ax.set_ylabel("alpha")
    value_gamma_ax.set_ylabel("gamma")
    value_ax.grid(True, alpha=0.25)

    grad_ax.set_title("Hypergradients")
    grad_ax.set_xlabel("Closed-loop step")
    grad_ax.set_ylabel(r"$\partial L / \partial \alpha$")
    grad_gamma_ax.set_ylabel(r"$\partial L / \partial \gamma$")
    grad_ax.grid(True, alpha=0.25)

    style_handles = [
        Line2D([0], [0], color="black", linewidth=2.0, linestyle="-", label="alpha (solid)"),
        Line2D([0], [0], color="black", linewidth=2.0, linestyle="--", label="gamma (dashed)"),
    ]
    fig.legend(handles=style_handles, loc="upper right", ncol=2, frameon=False)

    axes[2].set_title("Upper-Level Loss")
    axes[2].set_xlabel("Closed-loop step")
    axes[2].set_ylabel("Loss")
    axes[2].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot saved DT-MPC learning experiment data")
    parser.add_argument(
        "--data",
        type=str,
        default=os.path.join(SCRIPT_DIR, "results", "static_vs_learning_tube_mpc_25x2.npz"),
        help="Path to the .npz bundle written by the experiment runner",
    )
    parser.add_argument(
        "--indices",
        type=str,
        default="",
        help="Comma-separated learning run indices to plot, e.g. '0,2,4'",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=3,
        help="Plot at most this many learning runs when --indices is not supplied",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="static_vs_learning_tube_mpc_25x2",
        help="Prefix for saved figures",
    )
    args = parser.parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Data file not found: {args.data}")

    os.makedirs(FIGURES_DIR, exist_ok=True)
    bundle = load_bundle(args.data)
    selected_indices = parse_indices(args.indices, args.max_runs, len(bundle["learning_results"]))

    if not selected_indices:
        raise ValueError("No valid learning run indices were selected")

    trajectories_path = os.path.join(FIGURES_DIR, f"{args.output_prefix}_selected_learning_trajectories.png")
    summary_path = os.path.join(FIGURES_DIR, f"{args.output_prefix}_selected_learning_summary.png")

    plot_trajectories(bundle, selected_indices, trajectories_path)
    plot_learning_summary(bundle, selected_indices, summary_path)

    print(f"Loaded data from {args.data}")
    print(f"Plotted learning runs: {selected_indices}")
    print(f"Saved trajectory figure to {trajectories_path}")
    print(f"Saved summary figure to {summary_path}")


if __name__ == "__main__":
    main()