import numpy as np
from typing import Tuple
from dynamics.base_system import DynamicalSystem

class SafetyEmbeddedDynamics(DynamicalSystem):
    """
    Dubins Car kinematic model.
    
    State: [x, y, theta, barrier_state] - position and orientation with barrier state for safety embedding
    barrier_state defined as 1/CBF
    CBF is defined as h(x) = min{d(x, obstacles) - safety_radius}
    Control: [v, omega] - linear and angular velocity
    """

    def __init__(self, wheelbase=1.0, obstacles=None):
        # State parameters
        self._state_dim = 4                     # [x, y, theta, barrier_state]
        self._control_dim = 2                   # [v, omega]
        self.obstacles = obstacles              # Embedded barrier states depend on the obstacles in the environment 
        self.limit_v = 1.0                        # Maximum linear velocity
        self.limit_omega = np.pi / 4             # Maximum angular velocity (radians

        # TODO: Update abstract class to allow for passing system parameters in constructor
        if wheelbase is None:
            self.L = 0.25                       # wheelbase length (for more complex dynamics)
        else:
            self.L = wheelbase  # wheelbase length (for more complex dynamics)

    def CBF(self, x: np.ndarray) -> float:
        """Control Barrier Function (CBF) for safety embedding."""
        # Compute the minimum distance to obstacles and subtract safety radius
        if self.obstacles is None:
            return 1.0  # No obstacles, so CBF is always positive
        
        x_pos, y_pos = x[0], x[1]
        
        min_distance = np.inf
        for obs in self.obstacles:
            obs_x, obs_y, obs_radius = obs
            distance = np.sqrt((x_pos - obs_x)**2 + (y_pos - obs_y)**2) - (obs_radius)
            min_distance = min(min_distance, distance)
        
        return min_distance

    @property
    def state_dim(self) -> int:
        """Dimension of the state vector (n)."""
        return self._state_dim

    @property
    def control_dim(self) -> int:
        """Dimension of the control vector (m)."""
        return self._control_dim

    def dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Computes the continuous-time derivative dx/dt = f(x, u).
        """
        v = u[0]
        omega = u[1]
        theta = x[2]

        # Add i.i.d Gaussian noise to dynamics.
        x_dot = v * np.cos(theta) + np.random.normal(0, 0.1)  # Add small noise for realism
        y_dot = v * np.sin(theta) + np.random.normal(0, 0.1)  # Add small noise for realism
        theta_dot = v / self.L * np.tan(omega) + np.random.normal(0, 0.1)  # Add small noise for realism

        # Compute the CBF for the barrier state
        barrier_state_dot = 0  # Placeholder - update based on CBF dynamics

        return np.array([x_dot, y_dot, theta_dot, barrier_state_dot])  
    
    def step(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """Discrete-time step using Euler integration."""
        x_dot = self.dynamics(x, u)
        x_new = x + x_dot * dt
        x_new[3] = 1.0 / self.CBF(x_new[0:2])  # Update the barrier state
        return x_new

    def continuous_jacobians(self, x: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        v = u[0]
        omega = u[1]
        theta = x[2]

        # Jacobian w.r.t state x
        A = np.zeros((self.state_dim, self.state_dim))
        A[0, 2] = -v * np.sin(theta)  # df1/dtheta
        A[1, 2] = v * np.cos(theta)   # df2/dtheta

        # Jacobian w.r.t control u
        B = np.zeros((self.state_dim, self.control_dim))
        B[0, 0] = np.cos(theta)       # df1/dv
        B[1, 0] = np.sin(theta)       # df2/dv
        B[2, 0] = (1 / self.L) * np.tan(omega)  # df3/dv
        B[2, 1] = (v / self.L) * (1 / (np.cos(omega)**2))  # df3/domega

        return A, B

class SafetyEmbeddedVisualizer:
    """Utility class to visualize trajectories, obstacles, and goal regions.

    All plotting is done via the static `visualize_trajectory` helper so it can
    be used independently of any plant instance.
    """

    @staticmethod
    def visualize_trajectory(
        trajectory,
        obstacles: np.ndarray,
        goal: np.ndarray | None = None,
        goal_radius: float = 0.25,
        figsize=(8, 8),
    ) -> None:
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Circle
        except ImportError:
            raise ImportError("matplotlib is required for visualization")

        traj = np.asarray(trajectory)

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect('equal')

        # Trajectory
        if traj.size == 0:
            raise ValueError("Empty trajectory given to visualize_trajectory")
        ax.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=2, label='Trajectory')
        ax.plot(traj[0, 0], traj[0, 1], 'go', markersize=8, label='Start')
        ax.plot(traj[-1, 0], traj[-1, 1], 'rs', markersize=8, label='End')

        # Obstacles
        for obs in np.asarray(obstacles):
            c = Circle((obs[0], obs[1]), obs[2], color='red', alpha=0.5)
            ax.add_patch(c)

        # Goal
        if goal is not None:
            goal_c = Circle((goal[0], goal[1]), goal_radius, color='green', alpha=0.5)
            ax.add_patch(goal_c)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title('Dubins Car Trajectory')
        plt.axis('equal')
        plt.show()
