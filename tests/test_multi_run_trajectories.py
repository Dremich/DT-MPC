import numpy as np
import jax.numpy as jnp

from dynamics.safety_embedded import SafetyEmbeddedDynamics, SafetyEmbeddedVisualizer
from solvers.ocp_interface import OCP
from solvers.optimal_control import DDPSolver
from solvers.tube_mpc import TubeMPC
from solvers.costs import QuadraticCost, TerminalCost

# ==================================================
# Simulation parameters (identical to single-run test)
# ==================================================
dt = 0.1
wheelbase = 0.25
horizon = 50
steps = 300
num_runs = 10

obstacles = np.array([
    [5.0, 5.0, 1.0],
    [3.0, 7.0, 1.0],
    [7.0, 3.0, 1.0],
    [1.0, 9.0, 1.0],
])
goal_state = np.array([10.0, 10.0, 0.0, 0.0])

car = SafetyEmbeddedDynamics(wheelbase, obstacles)

init_cbf = float(car.CBF(jnp.array([0.0, 0.0])))
initial_barrier = 1.0 / init_cbf if init_cbf > 1e-6 else 1e6


def make_controller():
    """Creates a fresh TubeMPC controller with all new cost/OCP instances."""
    Q_nom = jnp.diag(jnp.array([1.0, 1.0, 0.5, 100.0]))
    R_nom = jnp.diag(jnp.array([0.1, 0.1]))
    P_nom = jnp.diag(jnp.array([100.0, 100.0, 50.0, 100.0]))

    Q_anc = jnp.diag(jnp.array([50.0, 50.0, 10.0, 0.0]))
    R_anc = jnp.diag(jnp.array([1.0, 1.0]))
    P_anc = jnp.diag(jnp.array([200.0, 200.0, 50.0, 0.0]))

    nominal_ocp = OCP(
        system=car,
        stage_cost=QuadraticCost(Q_nom, R_nom, x_ref=goal_state),
        terminal_cost=TerminalCost(P_nom, x_ref=goal_state),
        horizon=horizon,
        dt=dt,
    )
    ancillary_ocp = OCP(
        system=car,
        stage_cost=QuadraticCost(Q_anc, R_anc),
        terminal_cost=TerminalCost(P_anc),
        horizon=horizon,
        dt=dt,
    )
    return TubeMPC(nominal_ocp, ancillary_ocp, DDPSolver)


def run_simulation():
    """Runs one closed-loop simulation and returns (actual states, nominal states)."""
    controller = make_controller()
    current_state = np.array([0.0, 0.0, 0.0, initial_barrier])
    states = [current_state.copy()]
    nom_states = []

    for k in range(steps):
        u = controller.step_tube(current_state)
        nom_states.append(controller.current_nominal_state.copy())
        current_state = car.step_sim(current_state, u, dt)
        states.append(current_state.copy())

        if np.linalg.norm(current_state[0:2] - goal_state[0:2]) < 0.5:
            print(f"  Run finished at step {k + 1} (goal reached).")
            break

    return np.array(states), np.array(nom_states)


# ==================================================
# Run multiple simulations
# ==================================================
all_trajectories = []
all_nominal_trajectories = []
for i in range(num_runs):
    print(f"Run {i + 1}/{num_runs}...")
    traj, nom_traj = run_simulation()
    all_trajectories.append(traj)
    all_nominal_trajectories.append(nom_traj)

# ==================================================
# Plot all trajectories on a single figure
# ==================================================
SafetyEmbeddedVisualizer.visualize_multiple_trajectories(
    all_trajectories,
    obstacles,
    goal=goal_state[0:2],
    goal_radius=0.5,
    nominal_trajectories=all_nominal_trajectories,
)
