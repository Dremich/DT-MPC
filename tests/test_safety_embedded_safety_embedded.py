"""
Closed-loop Tube MPC demo using SafetyEmbeddedDynamics.

Nominal MPC plans a disturbance-free trajectory toward the goal.
Ancillary MPC tracks the nominal trajectory to reject process noise.
Closed-loop simulation uses step_sim() (noisy) while both planners
use the JAX-traceable step() internally.
"""

import jax.numpy as jnp
import numpy as np

from dynamics.safety_embedded import SafetyEmbeddedDynamics, SafetyEmbeddedVisualizer
from solvers.costs import QuadraticCost, TerminalCost
from solvers.ocp_interface import OCPFormulation
from solvers.optimal_control import DDPSolver
from solvers.tube_mpc import TubeMPC

# ---───────────────────────────────────────────────────

DT        = 0.1
N         = 20           # MPC horizon steps
SIM_STEPS = 60           # Closed-loop simulation steps

OBSTACLES = [
    [0.0, 4.0, 1.0],    # in the path between start and goal
    [2.0, 6.5, 0.8],
]

GOAL = np.array([0.0, 9.0, 0.0, 0.0])   # [x, y, theta, barrier_state]


# ---───────────────────────────────────────────────────────────

def build_ocps(car):
    nx, nu = car.state_dim, car.control_dim

    # Weight matrices (barrier_state column/row zeroed – planner ignores it)
    Q_nom = jnp.diag(jnp.array([1.0,  1.0,  0.1,  0.0]))
    R_nom = jnp.diag(jnp.array([0.1,  0.1]))
    P_nom = jnp.diag(jnp.array([100., 100., 10.,  0.0]))

    Q_anc = jnp.diag(jnp.array([10.,  10.,  1.0,  0.0]))
    R_anc = jnp.diag(jnp.array([0.01, 0.01]))
    P_anc = jnp.diag(jnp.array([200., 200., 20.,  0.0]))

    goal_jnp = jnp.array(GOAL)

    # Nominal OCP – penalises deviation from goal at every stage and terminal
    nominal_ocp = OCPFormulation(
        dynamics      = car,
        stage_cost    = QuadraticCost(Q_nom, R_nom, x_ref=goal_jnp),
        terminal_cost = TerminalCost(P_nom, x_ref=goal_jnp),
        horizon       = N,
        dt            = DT,
    )

    # Ancillary OCP – reference will be updated by TubeMPC each step
    ancillary_ocp = OCPFormulation(
        dynamics      = car,
        stage_cost    = QuadraticCost(Q_anc, R_anc),
        terminal_cost = TerminalCost(P_anc, x_ref=goal_jnp),
        horizon       = N,
        dt            = DT,
    )

    return nominal_ocp, ancillary_ocp


# ---─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    car = SafetyEmbeddedDynamics(wheelbase=0.5, obstacles=OBSTACLES, noise_std=0.05)
    nominal_ocp, ancillary_ocp = build_ocps(car)

    solver   = DDPSolver()
    tube_mpc = TubeMPC(nominal_ocp, ancillary_ocp, solver)

    # Initial state – compute barrier from CBF at starting position
    start_xy = np.array([-1.0, 0.0])
    cbf0 = float(car.CBF(jnp.array(start_xy)))
    current_state = np.array([-1.0, 0.0, np.pi / 2, 1.0 / cbf0 if cbf0 > 1e-6 else 1e6])

    states_history  = [current_state.copy()]
    nominal_history = []   # record last nominal trajectory for overlay

    print(f"Start: {current_state[:3]}")
    print(f"Goal : {GOAL[:3]}\n")

    for step in range(SIM_STEPS):
        # Get ancillary control from Tube MPC
        u = tube_mpc.tube_mpc(current_state)

        # Store the last nominal trajectory for visualisation
        solver.load_problem(nominal_ocp)  # re-loads nominal so we can peek
        nominal_history = tube_mpc._prev_nominal_control  # controls only needed for warm-start

        # Step the plant with noise (real world)
        current_state = car.step_sim(current_state, u, DT)
        states_history.append(current_state.copy())

        barrier = current_state[3]
        if barrier > 10.0:
            print(f"  Step {step:3d}: NEAR OBSTACLE  barrier={barrier:.2f}")

        dist = np.linalg.norm(current_state[:2] - GOAL[:2])
        if dist < 0.4:
            print(f"\nGoal reached at step {step}!")
            break

    states_history = np.array(states_history)
    print(f"\nFinal position : ({states_history[-1, 0]:.2f}, {states_history[-1, 1]:.2f})")
    print(f"Distance to goal: {np.linalg.norm(states_history[-1, :2] - GOAL[:2]):.3f} m")

    SafetyEmbeddedVisualizer.visualize_trajectory(
        trajectory=states_history,
        obstacles=OBSTACLES,
        goal=GOAL[:2],
        goal_radius=0.4,
    )
