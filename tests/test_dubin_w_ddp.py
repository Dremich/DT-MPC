import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import addcopyfighandler

from dynamics.dubins_car import DubinsCar, DubinsCarVisualizer
from solvers.costs import QuadraticCost, TerminalCost
from solvers.ocp_interface import OCP
from solvers.optimal_control import DDPSolver

DT = 0.1
N = 50  # Horizon
NX = 3  # State dim [x, y, theta]
NU = 2  # Control dim [v, omega]

# =====================================================================
# 2. TEST IT
# =====================================================================
if __name__ == "__main__":

    # Build problem
    car = DubinsCar(wheelbase=0.5)
    cost = QuadraticCost(Q=jnp.diag(jnp.array([1.0, 1.0, 0.5])), R=jnp.diag(jnp.array([0.1, 0.1])))
    terminal_cost = TerminalCost(P=jnp.diag(jnp.array([100.0, 100.0, 50.0])), x_ref=jnp.array([0.0, 0.0, 0.0]))
    ocp = OCP(system=car, stage_cost=cost, terminal_cost=terminal_cost, horizon=N, dt=DT)

    # Start at [x=-5, y=5, theta=-pi/2]
    start_state = np.array([-5.0, 5.0, -np.pi/2])
    
    # Initial guess: Do nothing (zero control)
    guess_u = np.zeros((N, NU))
    
    # Run DDP!
    optimal_X, optimal_U = DDPSolver.run_ddp(ocp, start_state, guess_u, max_iters=30, cost_threshold=0.5)
    
    # Plotting
    plt.plot(optimal_X[:, 0], optimal_X[:, 1], '-o', label='Car Path')
    plt.plot(0, 0, 'rx', markersize=10, label='Goal')
    plt.title("DDP Solver Output")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.grid()
    plt.show()
