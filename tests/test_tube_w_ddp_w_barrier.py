import time
import numpy as np
import jax.numpy as jnp

from dynamics.dubins_car import DubinsCar # <-- Assuming this is your base system
from dynamics.safety_embedded import SafetyEmbeddedDynamics, SafetyEmbeddedVisualizer
from solvers.ocp_interface import OCP
from solvers.optimal_control import DDPSolver
from solvers.tube_mpc import TubeMPC
from solvers.costs import QuadraticCost, TerminalCost

# ==================================================
# Simulation Parameters
# ==================================================
dt = 0.1
wheelbase = 0.25
horizon = 50
steps = 75 

# --- Challenging "Forest" Course ---
obstacles = np.array([
    [5.0, 2.0, 1.0], [3.0, 6.0, 1.2], [7.0, 8.0, 1.5],
    [10.0, 4.0, 1.0], [12.0, 10.0, 1.5], [15.0, 7.0, 1.0],
    [8.0, 15.0, 1.8], [5.0, 12.0, 2.0], [12.0, 18.0, 2.2],
    [18.0, 12.0, 1.5], [20.0, 5.0, 1.5], [22.0, 15.0, 1.2],
    [16.0, 22.0, 2.5], [10.0, 25.0, 1.8], [25.0, 10.0, 3.2],
    [20.0, 25.0, 1.5], [25.0, 20.0, 1.2], [5.0, 20.0, 2.0]
])
goal_state = np.array([28.0, 28.0, 0.0, 0.0]) 

# 1. Define the abstract constraint function
def forest_cbf(x):
    """Calculates distance to all obstacles. x is the base state [x, y, theta]"""
    dists = jnp.sqrt((x[0] - obstacles[:, 0])**2 + (x[1] - obstacles[:, 1])**2)
    return dists - obstacles[:, 2]

# 2. Instantiate the Base System and the Safety Wrapper
base_car = DubinsCar(wheelbase=wheelbase)
car = SafetyEmbeddedDynamics(base_system=base_car, constraint_func=forest_cbf, alpha=0.5, gamma=0.1)

# 3. Initialize the correct starting barrier state dynamically
init_base_state = jnp.array([0.0, 0.0, 0.0])
H_init = forest_cbf(init_base_state)
initial_barrier = float(jnp.sum(car.relaxed_barrier(H_init, car.alpha)))

current_state = np.array([0.0, 0.0, 0.0, initial_barrier]) 

# ==================================================
# Controller Setup
# ==================================================
Q_nom = jnp.diag(jnp.array([1.0, 1.0, 0.5, 100.0]))
R_nom = jnp.diag(jnp.array([0.1, 0.1]))
P_nom = jnp.diag(jnp.array([100.0, 100.0, 50.0, 100.0]))

nom_stage_cost = QuadraticCost(Q_nom, R_nom, x_ref=goal_state)
nom_term_cost = TerminalCost(P_nom, x_ref=goal_state)
nominal_ocp = OCP(system=car, stage_cost=nom_stage_cost, terminal_cost=nom_term_cost, horizon=horizon, dt=dt)

Q_anc = jnp.diag(jnp.array([50.0, 50.0, 10.0, 0.0]))
R_anc = jnp.diag(jnp.array([1.0, 1.0]))
P_anc = jnp.diag(jnp.array([200.0, 200.0, 50.0, 0.0]))

anc_stage_cost = QuadraticCost(Q_anc, R_anc) 
anc_term_cost = TerminalCost(P_anc)
ancillary_ocp = OCP(system=car, stage_cost=anc_stage_cost, terminal_cost=anc_term_cost, horizon=horizon, dt=dt)

controller = TubeMPC(nominal_ocp, ancillary_ocp, DDPSolver)

# ==================================================
# Simulation Loop
# ==================================================
states = [current_state.copy()] 
controls = [] 
nom_states = [] 

start_time = time.perf_counter()
controller_time = 0.0
sim_time = 0.0
step_times = []

for k in range(steps):
    print(f"Step {k}: Computing OCP...")
    step_start = time.perf_counter()

    t0 = time.perf_counter()
    u = controller.step_tube(current_state)
    t1 = time.perf_counter()
    controller_time += (t1 - t0)
    
    nom_states.append(controller.current_nominal_state.copy())

    t2 = time.perf_counter()
    current_state = car.step_sim(current_state, u, dt)
    t3 = time.perf_counter()
    sim_time += (t3 - t2)

    step_end = time.perf_counter()
    step_times.append(step_end - step_start)

    states.append(current_state.copy())
    controls.append(u.copy())

    if np.linalg.norm(current_state[0:2] - goal_state[0:2]) < 0.5:
        print(f"Goal reached at step {k}!")
        break
    else:
        print(f"Distance from goal: {np.linalg.norm(current_state[0:2] - goal_state[0:2]):.2f}")
        print(f"Safety Barrier Value: {current_state[3]:.2f}")

# ==================================================
# Visualization
# ==================================================
states = np.array(states)
nom_states = np.array(nom_states)

SafetyEmbeddedVisualizer.visualize_trajectory(
    states,
    obstacles, 
    goal_state[0:2], 
    0.5,
    nominal_trajectory=nom_states 
)

# Print timing summary
total_elapsed = time.perf_counter() - start_time
n_steps = len(controls)
print("\n--- Timing Summary ---")
print(f"Total elapsed time: {total_elapsed:.6f} s")
if n_steps > 0:
    print(f"Steps executed: {n_steps}")
    print(f"Average time per step: {total_elapsed / n_steps:.6f} s")
    print(f"Average controller time per step: {controller_time / n_steps:.6f} s")
    print(f"Average simulator time per step: {sim_time / n_steps:.6f} s")
    print(f"Average loop overhead per step: {(sum(step_times) - controller_time - sim_time) / n_steps:.6f} s")
