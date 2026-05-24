import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple

from dynamics.base_system import DynamicalSystem

jax.config.update("jax_enable_x64", True)


class SafetyEmbeddedDynamics(DynamicalSystem):
    """
    Dubins Car with an embedded safety (barrier) state.

    State : [x, y, theta, barrier_state]   where barrier_state = 1 / CBF(x,y)
    Control: [v, omega]  – linear and angular velocity

    CBF(x,y) = min over obstacles of  dist(x,y, obs_centre) - obs_radius

    step()      – JAX-traceable, deterministic; used by the DDP planner.
    step_sim()  – NumPy, adds Gaussian process noise; used for closed-loop simulation.
    """

    def __init__(self, wheelbase: float = 1.0, obstacles=None, noise_std: float = 0.25):
        self._state_dim   = 4   # [x, y, theta, barrier_state]
        self._control_dim = 2   # [v, omega]
        self.L            = wheelbase if wheelbase is not None else 0.25
        self.noise_std    = noise_std
        self.obstacles    = obstacles

        # Pre-convert obstacle list to a JAX array for vectorised CBF computation
        if obstacles is not None:
            self._obs = jnp.array(obstacles, dtype=jnp.float64)   # (M, 3)
        else:
            self._obs = None

        # Cache JAX transformations for the discrete step
        self._step_jit = jax.jit(self.step)
        self._jac_A = jax.jit(jax.jacobian(self.step, argnums=0))
        self._jac_B = jax.jit(jax.jacobian(self.step, argnums=1))

    # ---───────────────────────────────────────────────────────

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def control_dim(self) -> int:
        return self._control_dim

    # ---───────────────────────────────────────────────────────────

    def CBF(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Control Barrier Function value.
        Returns the minimum signed distance to any obstacle surface.
        JAX-traceable (no Python-level data-dependent branching).
        """
        if self._obs is None:
            return jnp.array(1.0)
        dist = jnp.sqrt((x[0] - self._obs[:, 0]) ** 2 + (x[1] - self._obs[:, 1]) ** 2) \
               - self._obs[:, 2]
        return jnp.min(dist)

    # ---─────────────────────────────────────────────────────────

    def dynamics(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """
        Deterministic continuous-time derivative  dx/dt = f(x, u).
        JAX-traceable.  barrier_state is handled discretely in step().
        """
        v, omega, theta = u[0], u[1], x[2]
        return jnp.array([
            v * jnp.cos(theta),
            v * jnp.sin(theta),
            (v / self.L) * jnp.tan(omega),
            0.0,   # barrier_state_dot updated after Euler step
        ])

    def step(self, x: jnp.ndarray, u: jnp.ndarray, dt: float) -> jnp.ndarray:
        """
        JAX-traceable deterministic Euler step.
        Used by DDPSolver for planning and for computing Jacobians via jax.jacobian.
        """
        x_new = x + self.dynamics(x, u) * dt
        cbf   = self.CBF(x_new)
        # Guard against zero / negative CBF (obstacle penetration)
        barrier = jnp.where(cbf > 1e-6, 1.0 / cbf, 1e6)
        x_new   = x_new.at[3].set(barrier)
        return x_new

    def step_sim(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """
        Noisy Euler step for closed-loop simulation.
        Adds Gaussian process noise to the positional/heading states.
        """
        v, omega, theta = float(u[0]), float(u[1]), float(x[2])
        noise = np.random.normal(0.0, self.noise_std, 3)
        x_dot = np.array([
            v * np.cos(theta) + noise[0],
            v * np.sin(theta) + noise[1],
            (v / self.L) * np.tan(omega) + noise[2],
            0.0,
        ])
        x_new       = np.array(x, dtype=np.float64) + x_dot * dt
        cbf         = float(self.CBF(jnp.array(x_new[:2])))
        x_new[3]    = 1.0 / cbf if cbf > 1e-6 else 1e6
        return x_new

    # ---────────────────────────────────────────────────────────

    def continuous_jacobians(
        self, x: jnp.ndarray, u: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Analytical continuous Jacobians A_c = df/dx, B_c = df/du."""
        v, omega, theta = u[0], u[1], x[2]

        A = jnp.zeros((self._state_dim, self._state_dim))
        A = A.at[0, 2].set(-v * jnp.sin(theta))
        A = A.at[1, 2].set( v * jnp.cos(theta))

        B = jnp.zeros((self._state_dim, self._control_dim))
        B = B.at[0, 0].set(jnp.cos(theta))
        B = B.at[1, 0].set(jnp.sin(theta))
        B = B.at[2, 0].set((1.0 / self.L) * jnp.tan(omega))
        B = B.at[2, 1].set((v / self.L) / jnp.cos(omega) ** 2)
        return A, B

    def discrete_jacobians(
        self, x: jnp.ndarray, u: jnp.ndarray, dt: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Exact discrete Jacobians via JAX autodiff on step().
        Captures the barrier-state update that the analytic continuous
        Jacobians miss.
        """
        A_d = self._jac_A(x, u, dt)
        B_d = self._jac_B(x, u, dt)
        return A_d, B_d


# ---───────────────────────────────────────────────────────────

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
