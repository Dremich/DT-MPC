"""Run static, learning, and nominal MPC rollouts over many disturbed trials.

This experiment mirrors the structure in z_experiments/run_experiments.py and
compares three controllers under the same disturbance model:

- Pure nominal MPC
- Static DT-MPC (learning_rate=0.0)
- Learning DT-MPC (learning_rate>0)

Each rollout uses true dynamics with per-step, per-state uniform disturbances in
[-0.05, 0.05], then all trajectories are overlaid in a single plot.
"""

import argparse
import os
from time import perf_counter

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from dynamics.dubins_car import DubinsCar
from dynamics.safety_embedded import SafetyEmbeddedDynamics
from learning.doc_engine import DifferentiableOptimalControl
from learning.dt_mpc_loop import DTMPCTrainer
from solvers.costs import QuadraticCost, TerminalCost
from solvers.ocp_interface import OCP
from solvers.optimal_control import DDPSolver


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


class UniformDisturbedDubinsCar(DubinsCar):
    """True dynamics model with additive uniform disturbance in continuous-time rates."""

    def __init__(self, wheelbase: float = 0.25, dmin: float = -0.05, dmax: float = 0.05, seed: int = 0):
        super().__init__(wheelbase=wheelbase)
        self.dmin = float(dmin)
        self.dmax = float(dmax)
        self.rng = np.random.default_rng(seed)

    def step(self, x, u, dt):
        disturbance = self.rng.uniform(self.dmin, self.dmax, size=self.state_dim)
        dx = np.array(self.dynamics(x, u), dtype=float) + disturbance
        return jnp.array(np.array(x, dtype=float) + dx * dt)


def cbf_factory(obs_array: np.ndarray):
    def cbf(x):
        dists = jnp.sqrt((x[0] - obs_array[:, 0]) ** 2 + (x[1] - obs_array[:, 1]) ** 2)
        return dists - obs_array[:, 2]

    return cbf


def make_nominal_ancillary(car, goal_state, horizon=50, dt=0.05):
    q_nom = jnp.diag(jnp.array([1.0, 1.0, 0.5, 100.0]))
    r_nom = jnp.diag(jnp.array([0.1, 0.1]))
    p_nom = jnp.diag(jnp.array([100.0, 100.0, 50.0, 0.0]))
    nominal_ocp = OCP(
        system=car,
        stage_cost=QuadraticCost(q_nom, r_nom, x_ref=goal_state),
        terminal_cost=TerminalCost(p_nom, x_ref=goal_state),
        horizon=horizon,
        dt=dt,
    )

    q_anc = jnp.diag(jnp.array([50.0, 50.0, 10.0, 100.0]))
    r_anc = jnp.diag(jnp.array([1.0, 1.0]))
    p_anc = jnp.diag(jnp.array([200.0, 200.0, 50.0, 0.0]))
    ancillary_ocp = OCP(
        system=car,
        stage_cost=QuadraticCost(q_anc, r_anc),
        terminal_cost=TerminalCost(p_anc),
        horizon=horizon,
        dt=dt,
    )
    return nominal_ocp, ancillary_ocp


def get_initial_state(car, cbf_fn, start_tuple):
    base = jnp.array([start_tuple[0], start_tuple[1], start_tuple[2]])
    b0 = float(jnp.sum(car.relaxed_barrier(cbf_fn(base), car.alpha)))
    return np.array([base[0], base[1], base[2], b0], dtype=float)


def run_episode(
    controller_type: str,
    obstacles: np.ndarray,
    goal_state: np.ndarray,
    start_state: tuple[float, float, float],
    steps: int,
    dt: float,
    horizon: int,
    learning_rate: float,
    seed: int,
):
    if controller_type not in {"static", "learning", "nominal"}:
        raise ValueError(f"Unsupported controller_type: {controller_type}")

    np.random.seed(seed)
    cbf = cbf_factory(obstacles)

    true_sys = UniformDisturbedDubinsCar(
        wheelbase=0.25,
        dmin=-2,
        dmax=2,
        seed=seed,
    )

    base_car = DubinsCar(wheelbase=0.25)
    car = SafetyEmbeddedDynamics(
        base_system=base_car,
        constraint_func=cbf,
        alpha=0.3,
        gamma=0.1,
        noise_std=0.0,
        true_base_system=true_sys,
    )

    nominal_ocp, ancillary_ocp = make_nominal_ancillary(car, goal_state, horizon=horizon, dt=dt)
    trainer = None
    if controller_type in {"static", "learning"}:
        trainer = DTMPCTrainer(
            car,
            nominal_ocp,
            ancillary_ocp,
            DDPSolver,
            DifferentiableOptimalControl(),
            learning_rate=learning_rate,
            horizon_H=steps,
        )

    current_state = get_initial_state(car, cbf, start_state)
    states = [current_state.copy()]
    nominal_states = []
    controls = []
    alpha_hist = [float(car.alpha)]
    gamma_hist = [float(car.gamma)]
    loss_hist = []
    grad_theta_hist = []
    step_time_hist = []
    violated = False
    status = "TIMEOUT"

    pure_nominal_state = None
    pure_nominal_prev_control = None

    t0 = perf_counter()
    for _ in range(steps):
        step_t0 = perf_counter()

        if controller_type == "nominal":
            if pure_nominal_state is None:
                pure_nominal_state = np.copy(current_state)

            nom_x, nom_u, _ = DDPSolver.run_ddp(
                nominal_ocp,
                pure_nominal_state,
                pure_nominal_prev_control,
            )
            pure_nominal_prev_control = np.roll(nom_u, shift=-1, axis=0)
            pure_nominal_prev_control[-1] = nom_u[-1]
            pure_nominal_state = np.asarray(nom_x[1], dtype=float)
            u = np.asarray(nom_u[0], dtype=float)
            diagnostics = {
                "alpha": float(car.alpha),
                "gamma": float(car.gamma),
                "loss": 0.0,
                "grad_theta": np.zeros(2, dtype=float),
            }
            nominal_state = pure_nominal_state.copy()
        else:
            trainer.learning_rate = learning_rate if controller_type == "learning" else 0.0
            u, diagnostics = trainer.train_step(current_state)
            nominal_state = np.asarray(trainer.current_nominal_state, dtype=float)

        current_state = car.step_sim(current_state, u, dt=dt)
        controls.append(np.asarray(u, dtype=float))
        nominal_states.append(np.asarray(nominal_state, dtype=float))
        states.append(current_state.copy())
        alpha_hist.append(float(diagnostics["alpha"]))
        gamma_hist.append(float(diagnostics["gamma"]))
        loss_hist.append(float(diagnostics["loss"]))
        grad_theta_hist.append(np.asarray(diagnostics["grad_theta"], dtype=float))
        step_time_hist.append(perf_counter() - step_t0)

        if np.any(np.array(cbf(current_state)) < 0.0):
            status = "COLLIDED"
            violated = True
            break

        if np.linalg.norm(current_state[0:2] - goal_state[0:2]) < 0.5:
            status = "GOAL_REACHED"
            break

    elapsed = perf_counter() - t0
    return {
        "status": status,
        "steps_taken": len(states) - 1,
        "time_s": elapsed,
        "states": np.array(states),
        "nominal_states": np.array(nominal_states),
        "controls": np.array(controls),
        "alpha": np.array(alpha_hist),
        "gamma": np.array(gamma_hist),
        "loss": np.array(loss_hist),
        "grad_theta": np.array(grad_theta_hist),
        "step_time_s": np.array(step_time_hist),
        "violated": violated,
    }


def summarize_controller_results(results):
    total_runs = len(results)
    if total_runs == 0:
        return {
            "num_runs": 0,
            "success_rate": 0.0,
            "violation_rate": 0.0,
            "avg_trajectory_length": 0.0,
            "avg_step_time_s": 0.0,
            "avg_total_time_s": 0.0,
        }

    success_count = sum(res["status"] == "GOAL_REACHED" for res in results)
    violation_count = sum(bool(res.get("violated", False)) for res in results)
    avg_trajectory_length = float(np.mean([res["steps_taken"] for res in results]))
    avg_step_time_s = float(
        np.mean([
            float(np.mean(res["step_time_s"])) if len(res["step_time_s"]) else 0.0
            for res in results
        ])
    )
    avg_total_time_s = float(np.mean([res["time_s"] for res in results]))

    return {
        "num_runs": total_runs,
        "success_rate": success_count / total_runs,
        "violation_rate": violation_count / total_runs,
        "avg_trajectory_length": avg_trajectory_length,
        "avg_step_time_s": avg_step_time_s,
        "avg_total_time_s": avg_total_time_s,
    }


def save_experiment_bundle(
    out_path,
    nominal_results,
    static_results,
    learning_results,
    summary_metrics,
    obstacles,
    goal_state,
    start_state,
    settings,
):
    np.savez_compressed(
        out_path,
        nominal_results=np.array(nominal_results, dtype=object),
        static_results=np.array(static_results, dtype=object),
        learning_results=np.array(learning_results, dtype=object),
        summary_metrics=np.array(summary_metrics, dtype=object),
        obstacles=np.array(obstacles, dtype=float),
        goal_state=np.array(goal_state, dtype=float),
        start_state=np.array(start_state, dtype=float),
        settings=np.array(settings, dtype=object),
    )


def plot_all_trajectories(
    nominal_results,
    static_results,
    learning_results,
    obstacles,
    goal_state,
    start_state,
    out_path,
):
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_aspect("equal")

    for i, res in enumerate(nominal_results):
        traj = res["states"]
        label = "Nominal MPC" if i == 0 else None
        ax.plot(traj[:, 0], traj[:, 1], color="#2ca02c", alpha=0.45, linewidth=1.3, label=label)

    for i, res in enumerate(static_results):
        traj = res["states"]
        label = "Static Tube-MPC" if i == 0 else None
        ax.plot(traj[:, 0], traj[:, 1], color="#1f77b4", alpha=0.45, linewidth=1.3, label=label)

    for i, res in enumerate(learning_results):
        traj = res["states"]
        label = "Learning Tube-MPC" if i == 0 else None
        ax.plot(traj[:, 0], traj[:, 1], color="#ff7f0e", alpha=0.45, linewidth=1.3, label=label)

    for obs in obstacles:
        ax.add_patch(Circle((obs[0], obs[1]), obs[2], color="red", alpha=0.35))

    ax.plot(start_state[0], start_state[1], marker="o", color="black", markersize=7, label="Start")
    ax.plot(goal_state[0], goal_state[1], marker="*", color="green", markersize=12, label="Goal")
    ax.add_patch(Circle((goal_state[0], goal_state[1]), 0.5, color="green", alpha=0.20))

    ax.set_title("Nominal vs Static vs Learning Trajectories")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Static vs learning tube-MPC multi-trajectory experiment")
    parser.add_argument("--num-runs", type=int, default=25, help="Number of runs per controller")
    parser.add_argument("--steps", type=int, default=60, help="Max closed-loop steps per run")
    parser.add_argument("--horizon", type=int, default=50, help="DDP horizon")
    parser.add_argument("--dt", type=float, default=0.05, help="Integration step")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Learning rate for DT-MPC")
    parser.add_argument("--seed", type=int, default=47, help="Base random seed")
    args = parser.parse_args()

    # centers = np.array(
    #     [
    #         [2.0, 4.0],
    #         [4.0, 2.0],
    #         [4.0, 8.0],
    #         [6.0, 6.0],
    #         [8.0, 4.0],
    #     ],
    #     dtype=float,
    # )
    
    
    centers = np.array([
        [5.0, 2.0, 1.0], [3.0, 6.0, 1.2], [7.0, 8.0, 1.5],
        [10.0, 4.0, 1.0], [12.0, 10.0, 1.5], [15.0, 7.0, 1.0],
        [8.0, 15.0, 1.8], [5.0, 12.0, 2.0], [12.0, 18.0, 2.2],
        [18.0, 12.0, 1.5], [20.0, 5.0, 1.5], [22.0, 15.0, 1.2],
        [16.0, 22.0, 2.5], [10.0, 25.0, 1.8], [25.0, 10.0, 3.2],
        [20.0, 25.0, 1.5], [25.0, 20.0, 1.2], [5.0, 20.0, 2.0]
    ], dtype=float)
    
    radii = np.full((centers.shape[0], 1), 1.0, dtype=float)
    obstacles = np.hstack([centers, radii])


    goal_state = np.array([28.0, 28.0, 0.0, 0.0], dtype=float)
    start_state = (0.0, 2.5, 0.0)

    # start_state = (0.0, 0.0, float(np.pi / 4.0))
    # goal_state = np.array([10.0, 10.0, 0.0, 0.0], dtype=float)

    nominal_results = []
    static_results = []
    learning_results = []

    print("Running nominal MPC rollouts...")
    for i in range(args.num_runs):
        res = run_episode(
            controller_type="nominal",
            obstacles=obstacles,
            goal_state=goal_state,
            start_state=start_state,
            steps=args.steps,
            dt=args.dt,
            horizon=args.horizon,
            learning_rate=0.0,
            seed=args.seed + i,
        )
        nominal_results.append(res)
        print(
            f"  Nominal {i + 1:02d}/{args.num_runs}: {res['status']:<12} "
            f"steps={res['steps_taken']:3d} step_time={np.mean(res['step_time_s']) if len(res['step_time_s']) else 0.0:.4f}s"
        )

    print("Running static tube-MPC rollouts...")
    for i in range(args.num_runs):
        res = run_episode(
            controller_type="static",
            obstacles=obstacles,
            goal_state=goal_state,
            start_state=start_state,
            steps=args.steps,
            dt=args.dt,
            horizon=args.horizon,
            learning_rate=0.0,
            seed=args.seed + i,
        )
        static_results.append(res)
        print(
            f"  Static {i + 1:02d}/{args.num_runs}: {res['status']:<12} "
            f"steps={res['steps_taken']:3d} step_time={np.mean(res['step_time_s']) if len(res['step_time_s']) else 0.0:.4f}s"
        )

    print("Running learning tube-MPC rollouts...")
    for i in range(args.num_runs):
        res = run_episode(
            controller_type="learning",
            obstacles=obstacles,
            goal_state=goal_state,
            start_state=start_state,
            steps=args.steps,
            dt=args.dt,
            horizon=args.horizon,
            learning_rate=args.learning_rate,
            seed=args.seed + 1000 + i,
        )
        learning_results.append(res)
        print(
            f"  Learning {i + 1:02d}/{args.num_runs}: {res['status']:<12} "
            f"steps={res['steps_taken']:3d} step_time={np.mean(res['step_time_s']) if len(res['step_time_s']) else 0.0:.4f}s"
        )

    nominal_summary = summarize_controller_results(nominal_results)
    static_summary = summarize_controller_results(static_results)
    learning_summary = summarize_controller_results(learning_results)

    def print_summary(name, summary):
        print(
            f"[{name}] success={summary['success_rate']:.2%} "
            f"violation={summary['violation_rate']:.2%} "
            f"avg_length={summary['avg_trajectory_length']:.1f} "
            f"avg_step_time={summary['avg_step_time_s']:.4f}s"
        )

    print_summary("Nominal", nominal_summary)
    print_summary("Static", static_summary)
    print_summary("Learning", learning_summary)

    out_path = os.path.join(FIGURES_DIR, "static_vs_learning_tube_mpc_25x2.png")
    plot_all_trajectories(
        nominal_results=nominal_results,
        static_results=static_results,
        learning_results=learning_results,
        obstacles=obstacles,
        goal_state=goal_state,
        start_state=start_state,
        out_path=out_path,
    )

    print(f"Saved trajectory comparison figure to {out_path}")

    data_path = os.path.join(RESULTS_DIR, "static_vs_learning_tube_mpc_25x2.npz")
    save_experiment_bundle(
        out_path=data_path,
        nominal_results=nominal_results,
        static_results=static_results,
        learning_results=learning_results,
        summary_metrics={
            "nominal": nominal_summary,
            "static": static_summary,
            "learning": learning_summary,
        },
        obstacles=obstacles,
        goal_state=goal_state,
        start_state=start_state,
        settings={
            "num_runs": args.num_runs,
            "steps": args.steps,
            "horizon": args.horizon,
            "dt": args.dt,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
        },
    )
    print(f"Saved experiment bundle to {data_path}")


if __name__ == "__main__":
    main()
