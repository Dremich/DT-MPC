import numpy as np
import jax.numpy as jnp

from dynamics.safety_embedded import SafetyEmbeddedDynamics, SafetyEmbeddedVisualizer
from solvers.ocp_interface import OCP
from solvers.optimal_control import DDPSolver
from solvers.tube_mpc import TubeMPC
from solvers.costs import QuadraticCost, TerminalCost

# ==================================================
# Specific to Dubins Car (Safety-Embedded Dynamics)
# ==================================================

# Simulation parameters
dt = 0.1
wheelbase = 0.25
horizon = 50
steps = 300 # Increased for larger course

# Establish environment 
# --- Original Course (Commented Out) ---
# obstacles = np.array([
#     [5.0, 5.0, 1.0],
#     [3.0, 7.0, 1.0],
#     [7.0, 3.0, 1.0],
#     # Top-left obstacle
#     [1.0, 9.0, 1.0],])
# goal_state = np.array([10.0, 10.0, 0.0, 0.0]) 

# --- Challenging "Forest" Course ---
obstacles = np.array([
    [5.0, 2.0, 1.0], [3.0, 6.0, 1.2], [7.0, 8.0, 1.5],
    [10.0, 4.0, 1.0], [12.0, 10.0, 1.5], [15.0, 7.0, 1.0],
    [8.0, 15.0, 1.8], [5.0, 12.0, 1.0], [12.0, 18.0, 1.2],
    [18.0, 12.0, 1.5], [20.0, 5.0, 1.5], [22.0, 15.0, 1.2],
    [16.0, 22.0, 1.5], [10.0, 25.0, 1.8], [25.0, 10.0, 1.2],
    [20.0, 25.0, 1.5], [25.0, 20.0, 1.2], [5.0, 20.0, 1.0]
])
goal_state = np.array([28.0, 28.0, 0.0, 0.0]) # x, y, theta, barrier_state
# ------------------------------------

car = SafetyEmbeddedDynamics(wheelbase, obstacles)

# Initialize correct starting barrier state[cite: 13]
init_cbf = float(car.CBF(jnp.array([0.0, 0.0])))
initial_barrier = 1.0 / init_cbf if init_cbf > 1e-6 else 1e6
current_state = np.array([0.0, 0.0, 0.0, initial_barrier]) # x, y, theta, barrier_state

# Nominal MPC (Goal-Seeking & Obstacle Avoidance)
# The 4th diagonal element penalizes the barrier state
Q_nom = jnp.diag(jnp.array([1.0, 1.0, 0.5, 50.0]))
R_nom = jnp.diag(jnp.array([0.1, 0.1]))
P_nom = jnp.diag(jnp.array([100.0, 100.0, 50.0, 100.0]))

nom_stage_cost = QuadraticCost(Q_nom, R_nom, x_ref=goal_state)
nom_term_cost = TerminalCost(P_nom, x_ref=goal_state)

nominal_ocp = OCP(system=car, stage_cost=nom_stage_cost, terminal_cost=nom_term_cost, horizon=horizon, dt=dt)

# Ancillary MPC (Error-Tracking)
# Heavily penalizes deviation from the nominal trajectory (elements 0-2). Barrier penalty is 0 here.
Q_anc = jnp.diag(jnp.array([50.0, 50.0, 10.0, 0.0]))
R_anc = jnp.diag(jnp.array([1.0, 1.0]))
P_anc = jnp.diag(jnp.array([200.0, 200.0, 50.0, 0.0]))

# No x_ref or u_ref passed initially; TubeMPC updates them dynamically
anc_stage_cost = QuadraticCost(Q_anc, R_anc) 
anc_term_cost = TerminalCost(P_anc)

ancillary_ocp = OCP(system=car, stage_cost=anc_stage_cost, terminal_cost=anc_term_cost, horizon=horizon, dt=dt)

# Establish Tube MPC
controller = TubeMPC(nominal_ocp, ancillary_ocp, DDPSolver)

# Simulation loop
states = [current_state.copy()] # Track states for visualization
nom_states = [] # Track nominal states for dual visualization[cite: 13]

for k in range(steps):
    print(f"Step {k}: Computing OCP...")

    # Obtain control from Tube MPC
    u = controller.step_tube(current_state)
    
    # Store nominal state[cite: 17]
    nom_states.append(controller.current_nominal_state.copy())

    # Progress physics using the noisy simulator step[cite: 13]
    current_state = car.step_sim(current_state, u, dt)
    states.append(current_state.copy())

    # Early stopping if goal is reached
    if np.linalg.norm(current_state[0:2] - goal_state[0:2]) < 0.5:
        print(f"Goal reached at step {k}!")
        break

# Visualize results
states = np.array(states)
nom_states = np.array(nom_states)

SafetyEmbeddedVisualizer.visualize_trajectory(
    states,
    obstacles, 
    goal_state[0:2], 
    0.5,
    nominal_trajectory=nom_states # Pass nominal trajectory overlay[cite: 13]
)