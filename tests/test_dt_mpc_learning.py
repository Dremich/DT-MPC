"""Closed-loop DT-MPC with online gradient-descent adaptation of theta = [alpha, gamma].

Runs the differentiable tube-based MPC (Algorithm 2 of arXiv:2308.08426) on the
"forest" Dubins-car course. Each step the shared barrier parameters [alpha, gamma]
are updated by a projected gradient-descent step using the hypergradient produced
by the DOC engine (Algorithm 1). The script:

  1. Validates the analytic DOC gradient against central finite differences at a
     deliberately near-obstacle state (paper-style check, Fig. 3).
  2. Runs the closed loop, printing alpha, gamma, nabla_theta L and the loss at
     every step so the parameter adaptation can be inspected numerically.
  3. Saves plots of the parameter history and the trajectory to ./figures/.

Environment overrides:
  DTMPC_STEPS  number of closed-loop steps (default 60)
  DTMPC_ETA    learning rate            (default 0.01)
  DTMPC_SHOW   "1" to pop up plots interactively, "0" to only save (default 1)
"""

import os
import time
import numpy as np
import jax.numpy as jnp

from dynamics.dubins_car import DubinsCar
from dynamics.safety_embedded import SafetyEmbeddedDynamics, SafetyEmbeddedVisualizer
from solvers.ocp_interface import OCP
from solvers.optimal_control import DDPSolver
from solvers.costs import QuadraticCost, TerminalCost
from learning.doc_engine import DifferentiableOptimalControl
from learning.dt_mpc_loop import DTMPCTrainer
import addcopyfighandler
# ==================================================
# Simulation parameters (forest course, matches test_tube_w_ddp_w_barrier.py)
# ==================================================
dt = 0.05
wheelbase = 0.25
horizon = 50
STEPS = int(os.environ.get("DTMPC_STEPS", "60"))
ETA = float(os.environ.get("DTMPC_ETA", "0.01"))
SHOW = os.environ.get("DTMPC_SHOW", "1") == "1"
NOISE_STD = float(os.environ.get("DTMPC_NOISE_STD", 5.0))

ALPHA_0 = float(os.environ.get("DTMPC_ALPHA0", "0.3"))
GAMMA_0 = float(os.environ.get("DTMPC_GAMMA0", "0.1"))

obstacles = np.array([
    [5.0, 2.0, 1.0], [3.0, 6.0, 1.2], [7.0, 8.0, 1.5],
    [10.0, 4.0, 1.0], [12.0, 10.0, 1.5], [15.0, 7.0, 1.0],
    [8.0, 15.0, 1.8], [5.0, 12.0, 2.0], [12.0, 18.0, 2.2],
    [18.0, 12.0, 1.5], [20.0, 5.0, 1.5], [22.0, 15.0, 1.2],
    [16.0, 22.0, 2.5], [10.0, 25.0, 1.8], [25.0, 10.0, 3.2],
    [20.0, 25.0, 1.5], [25.0, 20.0, 1.2], [5.0, 20.0, 2.0]
])
goal_state = np.array([28.0, 28.0, 0.0, 0.0])


def forest_cbf(x):
    """Distance to every obstacle (positive = safe). x is the base state [x, y, theta]."""
    dists = jnp.sqrt((x[0] - obstacles[:, 0]) ** 2 + (x[1] - obstacles[:, 1]) ** 2)
    return dists - obstacles[:, 2]


def make_setup(alpha=ALPHA_0, gamma=GAMMA_0, noise_std=NOISE_STD):
    """Builds a fresh car + nominal/ancillary OCPs (cost weights as in the tube test)."""
    base_car = DubinsCar(wheelbase=wheelbase)
    car = SafetyEmbeddedDynamics(base_system=base_car, constraint_func=forest_cbf,
                                 alpha=alpha, gamma=gamma, noise_std=noise_std)

    Q_nom = jnp.diag(jnp.array([1.0, 1.0, 0.5, 100.0]))
    R_nom = jnp.diag(jnp.array([0.1, 0.1]))
    P_nom = jnp.diag(jnp.array([100.0, 100.0, 50.0, 0.0]))
    nominal_ocp = OCP(system=car,
                      stage_cost=QuadraticCost(Q_nom, R_nom, x_ref=goal_state),
                      terminal_cost=TerminalCost(P_nom, x_ref=goal_state),
                      horizon=horizon, dt=dt)

    Q_anc = jnp.diag(jnp.array([50.0, 50.0, 10.0, 100.0]))
    R_anc = jnp.diag(jnp.array([1.0, 1.0]))
    P_anc = jnp.diag(jnp.array([200.0, 200.0, 50.0, 0.0]))
    ancillary_ocp = OCP(system=car,
                        stage_cost=QuadraticCost(Q_anc, R_anc),
                        terminal_cost=TerminalCost(P_anc),
                        horizon=horizon, dt=dt)
    return car, nominal_ocp, ancillary_ocp


def initial_state(car, base_xy=(0.0, 0.0), theta=0.0):
    """Builds [x, y, theta, b0] with the barrier state seeded from the CBF."""
    base = jnp.array([base_xy[0], base_xy[1], theta])
    b0 = float(jnp.sum(car.relaxed_barrier(forest_cbf(base), car.alpha)))
    return np.array([base_xy[0], base_xy[1], theta, b0])


# ==================================================
# 1. Finite-difference validation of the DOC hypergradient
# ==================================================
def finite_difference_check(eps=5e-3):
    """Compares analytic nabla_theta L (DOC) vs central differences of the loss.

    Performed at a near-obstacle state so the alpha-derivative is active. The
    nominal reference is held fixed; only the ancillary problem is re-solved for
    each perturbation, isolating exactly the gradient the DOC engine computes.
    """
    print("=" * 70)
    print("FINITE-DIFFERENCE GRADIENT CHECK (near-obstacle state)")
    print("=" * 70)

    car, nominal_ocp, ancillary_ocp = make_setup()
    doc = DifferentiableOptimalControl()
    trainer = DTMPCTrainer(car, nominal_ocp, ancillary_ocp, DDPSolver, doc,
                           learning_rate=ETA)

    # State near obstacle (5, 2, r=1): distance ~1.3 -> zeta ~0.3 < alpha=0.5.
    state = initial_state(car, base_xy=(3.7, 2.0), theta=np.pi / 8)

    # Fixed nominal reference for this check.
    nom_x, nom_u, _ = DDPSolver.run_ddp(nominal_ocp, state)
    ancillary_ocp.stage_cost.update_reference(nom_x, nom_u)
    ancillary_ocp.terminal_cost.update_reference(nom_x[-1])

    # Analytic gradient.
    anc_x, anc_u, _ = DDPSolver.run_ddp(ancillary_ocp, state)
    loss0, grad_x = trainer.compute_upper_level_loss(anc_x, nom_x)
    grad_theta = doc.compute_gradient(ancillary_ocp, anc_x, anc_u, grad_x)

    # Central finite differences (nominal reference held fixed).
    a0, g0 = car.alpha, car.gamma

    def loss_at(alpha, gamma):
        car.alpha, car.gamma = alpha, gamma
        ax, _, _ = DDPSolver.run_ddp(ancillary_ocp, state)
        L, _ = trainer.compute_upper_level_loss(ax, nom_x)
        return L

    fd_alpha = (loss_at(a0 + eps, g0) - loss_at(a0 - eps, g0)) / (2 * eps)
    fd_gamma = (loss_at(a0, g0 + eps) - loss_at(a0, g0 - eps)) / (2 * eps)
    car.alpha, car.gamma = a0, g0  # restore

    print(f"  loss at (alpha={a0}, gamma={g0}): {loss0:.6f}")
    print(f"  {'':12s}{'dL/d_alpha':>16s}{'dL/d_gamma':>16s}")
    print(f"  {'DOC (ours)':12s}{grad_theta[0]:16.6f}{grad_theta[1]:16.6f}")
    print(f"  {'finite diff':12s}{fd_alpha:16.6f}{fd_gamma:16.6f}")
    fd = np.array([fd_alpha, fd_gamma])
    denom = np.maximum(np.abs(fd), 1e-8)
    rel = np.abs(grad_theta - fd) / denom
    print(f"  {'rel. error':12s}{rel[0]:16.3e}{rel[1]:16.3e}")
    print()
    return grad_theta, fd


# ==================================================
# 2. Closed-loop run with theta adaptation
# ==================================================
def run_closed_loop():
    print("=" * 80)
    print(f"CLOSED-LOOP DT-MPC  (steps={STEPS}, eta={ETA}, "
          f"alpha0={ALPHA_0}, gamma0={GAMMA_0}, noise_std={NOISE_STD})")
    print("=" * 80)

    car, nominal_ocp, ancillary_ocp = make_setup()
    doc = DifferentiableOptimalControl()
    trainer = DTMPCTrainer(car, nominal_ocp, ancillary_ocp, DDPSolver, doc,
                           learning_rate=ETA, horizon_H=STEPS)

    current_state = initial_state(car)

    states = [current_state.copy()]
    nom_states = []
    alpha_hist = [car.alpha]
    gamma_hist = [car.gamma]
    grad_hist = []
    loss_hist = []

    t0 = time.perf_counter()
    for k in range(STEPS):
        u, diag = trainer.train_step(current_state)
        nom_states.append(trainer.current_nominal_state.copy())

        current_state = car.step_sim(current_state, u, dt)

        states.append(current_state.copy())
        alpha_hist.append(diag["alpha"])
        gamma_hist.append(diag["gamma"])
        grad_hist.append(diag["grad_theta"].copy())
        loss_hist.append(diag["loss"])

        dist = np.linalg.norm(current_state[0:2] - goal_state[0:2])
        g = diag["grad_theta"]
        print(f"step {k:3d} | dist {dist:6.2f} | b {current_state[3]:8.3f} | "
              f"alpha {diag['alpha']:.4f} | gamma {diag['gamma']:+.4f} | "
              f"grad=[{g[0]:+.3e}, {g[1]:+.3e}] | loss {diag['loss']:.3f}")

        if np.any(np.array(forest_cbf(current_state)) < 0.0):
            print(f"\nCollision detected at step {k}!")

        if dist < 0.5:
            print(f"\nGoal reached at step {k}!")
            break

    elapsed = time.perf_counter() - t0
    n = len(grad_hist)
    print("\n--- Summary ---")
    print(f"Steps executed:      {n}")
    print(f"Total time:          {elapsed:.2f} s  ({elapsed / max(n,1):.3f} s/step)")
    print(f"alpha: {ALPHA_0:.4f} -> {alpha_hist[-1]:.4f}  "
          f"(min {min(alpha_hist):.4f}, max {max(alpha_hist):.4f})")
    print(f"gamma: {GAMMA_0:+.4f} -> {gamma_hist[-1]:+.4f}  "
          f"(min {min(gamma_hist):+.4f}, max {max(gamma_hist):+.4f})")
    
    if np.any(np.array(forest_cbf(states[-1])) < 0.0):
        print("Final collision status: COLLIDED")
    else:
        print("Final collision status: SAFE")
    print(f"Final barrier value: {states[-1][3]:.3f}")

    return {
        "states": np.array(states),
        "nom_states": np.array(nom_states),
        "alpha": np.array(alpha_hist),
        "gamma": np.array(gamma_hist),
        "grad": np.array(grad_hist),
        "loss": np.array(loss_hist),
    }


def save_plots(result):
    import matplotlib
    if not SHOW:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs("figures", exist_ok=True)

    # Parameter / gradient / loss history.
    fig, axs = plt.subplots(2, 2, figsize=(11, 7))
    axs[0, 0].plot(result["alpha"], "b-o", ms=3)
    axs[0, 0].set_title("alpha (relaxation)"); axs[0, 0].set_xlabel("step"); axs[0, 0].grid(alpha=0.3)
    axs[0, 1].plot(result["gamma"], "r-o", ms=3)
    axs[0, 1].set_title("gamma (barrier feedback)"); axs[0, 1].set_xlabel("step"); axs[0, 1].grid(alpha=0.3)
    if len(result["grad"]) > 0:
        axs[1, 0].plot(result["grad"][:, 0], label=r"$\partial L/\partial\alpha$")
        axs[1, 0].plot(result["grad"][:, 1], label=r"$\partial L/\partial\gamma$")
    axs[1, 0].set_title("hypergradient"); axs[1, 0].set_xlabel("step"); axs[1, 0].legend(); axs[1, 0].grid(alpha=0.3)
    axs[1, 1].plot(result["loss"], "k-")
    axs[1, 1].set_title("upper-level loss"); axs[1, 1].set_xlabel("step"); axs[1, 1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("figures/dt_mpc_learning_params.png", dpi=120)
    print("Saved figures/dt_mpc_learning_params.png")

    if SHOW:
        SafetyEmbeddedVisualizer.visualize_trajectory(
            result["states"], obstacles, goal_state[0:2], 0.5,
            nominal_trajectory=result["nom_states"],
        )
    else:
        plt.close("all")


def main():
    finite_difference_check()
    result = run_closed_loop()
    save_plots(result)


if __name__ == "__main__":
    main()
