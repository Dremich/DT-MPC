import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional, Tuple

from dynamics.base_system import DynamicalSystem
jax.config.update("jax_enable_x64", True)


class SafetyEmbeddedDynamics(DynamicalSystem):
    """
    Safety-Embedded Dynamics class: augments a base system with barrier states.
    """
    def __init__(self, base_system, constraint_func, alpha: float = 0.5, gamma: float = 0.1, rho: float = 10.0, noise_std: float = 0.25):
        self.base_system = base_system
        self.constraint_func = constraint_func
        self.alpha = alpha
        self.gamma = gamma
        self.rho = rho
        self.noise_std = noise_std
        
        self._state_dim   = base_system.state_dim + 1   
        self._control_dim = base_system.control_dim     

        # Cache JAX transformations 
        self._step_jit = jax.jit(self.step)
        
        # Standard Jacobians for the DDP solver (A_d, B_d)
        self._jac_A_c = jax.jit(jax.jacobian(self.dynamics, argnums=0))
        self._jac_B_c = jax.jit(jax.jacobian(self.dynamics, argnums=1))
        self._jac_A_d = jax.jit(jax.jacobian(self.step, argnums=0))
        self._jac_B_d = jax.jit(jax.jacobian(self.step, argnums=1))
        
        # Learning Jacobian for Algorithm 1 (Derivative w.r.t alpha)
        self._jac_alpha = jax.jit(jax.jacobian(self.step_for_learning, argnums=3))

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def control_dim(self) -> int:
        return self._control_dim

    def relaxed_barrier(self, zeta: jnp.ndarray, alpha: float) -> jnp.ndarray:
        safe_b = 1.0 / jnp.maximum(zeta, 1e-8) 
        diff = zeta - alpha
        relax_b = (1.0 / alpha) - (diff / (alpha**2)) + ((diff**2) / (alpha**3))
        return jnp.where(zeta >= alpha, safe_b, relax_b)

    def dynamics(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        base_x = x[:self.base_system.state_dim]
        base_dx = self.base_system.dynamics(base_x, u) 
        return jnp.concatenate([base_dx, jnp.array([0.0])])

    def step(self, x: jnp.ndarray, u: jnp.ndarray, dt: float, params: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Standard step for the DDP/MPC solver."""
        if params is None:
            alpha = self.alpha
            rho = self.rho
        else:
            alpha = params[0]  # Assuming alpha is the first parameter in the array
            rho = params[1]    # Assuming rho is the second parameter in the array

        return self.step_for_learning(x, u, dt, alpha, rho)
        
    def barrier_aggregate_sum(self, H_k: jnp.ndarray, H_next: jnp.ndarray, rho: float) -> jnp.ndarray:
        B_k = jnp.sum(self.relaxed_barrier(H_k, self.alpha))
        B_next = jnp.sum(self.relaxed_barrier(H_next, self.alpha))
        return B_total_k, B_total_next
    
    def barrier_aggregate_logsumexp(self, H_k: jnp.ndarray, H_next: jnp.ndarray, alpha: float, rho: float) -> jnp.ndarray:
        # logsumexp computes the max, so we negate H to find the soft-minimum distance
        min_H_k = -jax.scipy.special.logsumexp(-rho * H_k) / rho
        min_H_next = -jax.scipy.special.logsumexp(-rho * H_next) / rho

        # Now pass only the SINGLE closest distance through the relaxed barrier
        B_total_k = self.relaxed_barrier(min_H_k, alpha)
        B_total_next = self.relaxed_barrier(min_H_next, alpha)
        
        return B_total_k, B_total_next

    def step_for_learning(self, x: jnp.ndarray, u: jnp.ndarray, dt: float, alpha: float, rho: float) -> jnp.ndarray:
        """Exposes alpha explicitly so JAX can differentiate through it."""
        base_x = x[:self.base_system.state_dim]
        b_k = x[-1]

        base_x_next = self.base_system.step(base_x, u, dt)
        
        H_k = self.constraint_func(base_x)
        H_next = self.constraint_func(base_x_next)

        B_total_k, B_total_next = self.barrier_aggregate_logsumexp(H_k, H_next, alpha, rho)

        b_next = B_total_next - self.gamma * (B_total_k - b_k)
        return jnp.concatenate([base_x_next, jnp.array([b_next])])

    def step_sim(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """Noisy Euler step for closed-loop simulation."""
        base_x = x[:self.base_system.state_dim]
        b_k = x[-1]
        
        # Add noise to the continuous base dynamics
        noise = np.random.normal(0.0, self.noise_std, self.base_system.state_dim)
        base_dx = np.array(self.base_system.dynamics(base_x, u)) + noise
        base_x_next = base_x + base_dx * dt
        
        # Calculate the deterministic barrier update based on the noisy state
        H_k = self.constraint_func(jnp.array(base_x))
        H_next = self.constraint_func(jnp.array(base_x_next))
        
        B_total_k = float(jnp.sum(self.relaxed_barrier(H_k, self.alpha)))
        B_total_next = float(jnp.sum(self.relaxed_barrier(H_next, self.alpha)))
        
        b_next = B_total_next - self.gamma * (B_total_k - b_k)
        return np.concatenate([base_x_next, np.array([b_next])])

    def continuous_jacobians(self, x: jnp.ndarray, u: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return self._jac_A_c(x, u), self._jac_B_c(x, u)

    def discrete_jacobians(self, x: jnp.ndarray, u: jnp.ndarray, dt: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return self._jac_A_d(x, u, dt), self._jac_B_d(x, u, dt)
        
    def jacobian_wrt_alpha(self, x: jnp.ndarray, u: jnp.ndarray, dt: float) -> jnp.ndarray:
        """Returns nabla_alpha f for the learning phase."""
        return self._jac_alpha(x, u, dt, self.alpha)

class SafetyEmbeddedVisualizer:
    """Utility class to visualise trajectories, obstacles, and goal regions."""

    @staticmethod
    def visualize_trajectory(
        trajectory,
        obstacles,
        goal=None,
        goal_radius: float = 0.25,
        figsize=(8, 8),
        nominal_trajectory=None,
    ) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        traj = np.asarray(trajectory)
        if traj.size == 0:
            raise ValueError("Empty trajectory given to visualize_trajectory.")

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect("equal")

        # Actual trajectory
        ax.plot(traj[:, 0], traj[:, 1], "b-",  linewidth=2, label="Actual trajectory")
        ax.plot(traj[0,  0], traj[0,  1], "go", markersize=8, label="Start")
        ax.plot(traj[-1, 0], traj[-1, 1], "rs", markersize=8, label="End")

        # Optional nominal trajectory overlay
        if nominal_trajectory is not None:
            nom = np.asarray(nominal_trajectory)
            ax.plot(nom[:, 0], nom[:, 1], "k--", linewidth=1.5, label="Nominal trajectory")

        # Obstacles
        for obs in np.asarray(obstacles):
            ax.add_patch(Circle((obs[0], obs[1]), obs[2], color="red", alpha=0.4))

        # Goal region
        if goal is not None:
            ax.add_patch(
                Circle((goal[0], goal[1]), goal_radius, color="green", alpha=0.4)
            )
            ax.plot(goal[0], goal[1], "g*", markersize=12, label="Goal")

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title("Safety-Embedded Dubins Car Trajectory")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def visualize_multiple_trajectories(
        trajectories,
        obstacles,
        goal=None,
        goal_radius: float = 0.25,
        figsize=(8, 8),
        nominal_trajectories=None,
    ) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        from matplotlib.cm import get_cmap

        _, ax = plt.subplots(figsize=figsize)
        ax.set_aspect("equal")

        cmap = get_cmap("tab10")
        n = len(trajectories)

        for i, traj in enumerate(trajectories):
            traj = np.asarray(traj)
            if traj.size == 0:
                continue
            color = cmap(i % 10)
            ax.plot(traj[:, 0], traj[:, 1], "-", color=color, linewidth=1.5, alpha=0.8, label=f"Run {i + 1}")
            ax.plot(traj[0, 0], traj[0, 1], "o", color=color, markersize=6)
            ax.plot(traj[-1, 0], traj[-1, 1], "s", color=color, markersize=6)

            if nominal_trajectories is not None and i < len(nominal_trajectories):
                nom = np.asarray(nominal_trajectories[i])
                if nom.size > 0:
                    ax.plot(nom[:, 0], nom[:, 1], "--", color=color, linewidth=1.0, alpha=0.4)

        for obs in np.asarray(obstacles):
            ax.add_patch(Circle((obs[0], obs[1]), obs[2], color="red", alpha=0.4))

        if goal is not None:
            ax.add_patch(Circle((goal[0], goal[1]), goal_radius, color="green", alpha=0.4))
            ax.plot(goal[0], goal[1], "g*", markersize=12, label="Goal")

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title(f"Safety-Embedded Dubins Car — {n} Runs")
        plt.tight_layout()
        plt.show()
