"""
Open-loop simulation of SafetyEmbeddedDynamics.

Drives the car with a proportional heading controller and monitors the
barrier state (= 1/CBF) to detect obstacle proximity.  Uses step_sim()
for realistic noisy simulation while keeping the JAX-based CBF and
step() available for DDP planning.
"""

import jax.numpy as jnp
import numpy as np

from dynamics.safety_embedded import SafetyEmbeddedDynamics, SafetyEmbeddedVisualizer

OBSTACLES = np.array([
    [5.0, 5.0, 1.0],
    [3.0, 7.0, 1.0],
    [7.0, 3.0, 1.0],
    [1.0, 9.0, 1.0],
])

DT        = 0.1
WHEELBASE = 0.25
HORIZON   = 50


def run_open_loop_simulation():
    car = SafetyEmbeddedDynamics(wheelbase=WHEELBASE, obstacles=OBSTACLES, noise_std=0.05)

    # Initialise barrier state from CBF at starting position
    start_xy = np.array([0.0, 0.0])
    cbf0     = float(car.CBF(jnp.array(start_xy)))
    current_state = np.array([0.0, 0.0, 0.0, 1.0 / cbf0 if cbf0 > 1e-6 else 1e6])

    states_history = [current_state.copy()]

    for k in range(HORIZON):
        # Proportional heading controller – steer toward pi/4 (north-east)
        heading_error = current_state[2] - np.pi / 4
        u_k = np.array([2.0, -heading_error])

        # Noisy simulation step
        current_state = car.step_sim(current_state, u_k, DT)

        barrier = current_state[3]
        if barrier > 10.0:
            print(f"  Step {k:3d}: NEAR OBSTACLE  barrier={barrier:.2f}")
        elif barrier < 0.0:
            print(f"  Step {k:3d}: OUTSIDE SAFE REGION  barrier={barrier:.2f}")

        states_history.append(current_state.copy())

    states_history = np.array(states_history)

    # Report final barrier and CBF statistics
    cbf_vals = 1.0 / np.where(
        states_history[:, 3] > 1e-6, states_history[:, 3], np.nan
    )
    print(f"\nSimulation complete: {len(states_history)} steps")
    print(f"Min CBF along trajectory: {np.nanmin(cbf_vals):.3f}")
    print(f"Final position: ({states_history[-1, 0]:.2f}, {states_history[-1, 1]:.2f})")

    SafetyEmbeddedVisualizer.visualize_trajectory(
        trajectory=states_history,
        obstacles=OBSTACLES,
        goal=np.array([10.0, 10.0]),
        goal_radius=0.25,
    )


if __name__ == "__main__":
    run_open_loop_simulation()
