"""Entry point for running DT-MPC training (online adaptation of theta=[alpha,gamma]).

A compact, self-contained demo of Algorithm 2: it runs the differentiable
tube-based MPC on a small obstacle course and prints how the shared barrier
parameters [alpha, gamma] adapt via gradient descent. For the full forest course,
finite-difference gradient validation and plots, see
``tests/test_dt_mpc_learning.py``.
"""

import numpy as np
import jax.numpy as jnp

from dynamics.dubins_car import DubinsCar
from dynamics.safety_embedded import SafetyEmbeddedDynamics
from solvers.ocp_interface import OCP
from solvers.optimal_control import DDPSolver
from solvers.costs import QuadraticCost, TerminalCost
from learning.doc_engine import DifferentiableOptimalControl
from learning.dt_mpc_loop import DTMPCTrainer


def main(steps: int = 30, eta: float = 0.01) -> None:
    dt, horizon = 0.1, 50
    obstacles = np.array([[5.0, 5.0, 1.0], [3.0, 7.0, 1.0], [7.0, 3.0, 1.0]])
    goal_state = np.array([10.0, 10.0, 0.0, 0.0])

    def cbf(x):
        d = jnp.sqrt((x[0] - obstacles[:, 0]) ** 2 + (x[1] - obstacles[:, 1]) ** 2)
        return d - obstacles[:, 2]

    car = SafetyEmbeddedDynamics(DubinsCar(wheelbase=0.25), cbf, alpha=0.5, gamma=0.1)

    Q_nom = jnp.diag(jnp.array([1.0, 1.0, 0.5, 100.0]))
    R_nom = jnp.diag(jnp.array([0.1, 0.1]))
    P_nom = jnp.diag(jnp.array([100.0, 100.0, 50.0, 100.0]))
    nominal_ocp = OCP(car, QuadraticCost(Q_nom, R_nom, x_ref=goal_state),
                      TerminalCost(P_nom, x_ref=goal_state), horizon, dt)

    Q_anc = jnp.diag(jnp.array([50.0, 50.0, 10.0, 0.0]))
    R_anc = jnp.diag(jnp.array([1.0, 1.0]))
    P_anc = jnp.diag(jnp.array([200.0, 200.0, 50.0, 0.0]))
    ancillary_ocp = OCP(car, QuadraticCost(Q_anc, R_anc), TerminalCost(P_anc), horizon, dt)

    doc = DifferentiableOptimalControl()
    trainer = DTMPCTrainer(car, nominal_ocp, ancillary_ocp, DDPSolver, doc,
                           learning_rate=eta, horizon_H=steps)

    b0 = float(jnp.sum(car.relaxed_barrier(cbf(jnp.array([0.0, 0.0, 0.0])), car.alpha)))
    state = np.array([0.0, 0.0, 0.0, b0])

    print(f"Initial theta: alpha={car.alpha:.4f}, gamma={car.gamma:+.4f}")
    for k in range(steps):
        u, diag = trainer.train_step(state)
        state = car.step_sim(state, u, dt)
        dist = np.linalg.norm(state[0:2] - goal_state[0:2])
        g = diag["grad_theta"]
        print(f"step {k:3d} | dist {dist:5.2f} | alpha {diag['alpha']:.4f} | "
              f"gamma {diag['gamma']:+.4f} | grad=[{g[0]:+.2e}, {g[1]:+.2e}] | "
              f"loss {diag['loss']:.2f}")
        if dist < 0.5:
            print(f"Goal reached at step {k}.")
            break
    print(f"Final theta:   alpha={car.alpha:.4f}, gamma={car.gamma:+.4f}")


if __name__ == "__main__":
    main()
