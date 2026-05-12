import numpy as np
from typing import Tuple
from dynamics.base_system import DynamicalSystem

class DubinsCar(DynamicalSystem):
    """
    Dubins Car kinematic model.
    
    State: [x, y, theta] - position and orientation
    Control: [v, omega] - linear and angular velocity
    """
    
    def __init__(self, wheelbase=1.0):
        # State parameters
        self._state_dim = 3                     # [x, y, theta]
        self._control_dim = 2                   # [v, omega]
        
        # TODO: Update abstract class to allow for passing system parameters in constructor
        if wheelbase is None:
            self.L = 0.25                       # wheelbase length (for more complex dynamics)
        else:
            self.L = wheelbase  # wheelbase length (for more complex dynamics)


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

        x_dot = v * np.cos(theta)
        y_dot = v * np.sin(theta)
        theta_dot = v / self.L * np.tan(omega)

        return np.array([x_dot, y_dot, theta_dot])
    
    def continuous_jacobians(self, x: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the continuous-time Jacobians A_c = df_c/dx and B_c = df_c/du.
        
        The continuous dynamics f_c(x, u) are defined as:
            dx/dt = v * cos(theta)
            dy/dt = v * sin(theta)
            dtheta/dt = (v / L) * tan(omega)
            
        State Jacobian A_c (3x3):
        How the state rates change with respect to the current state [x, y, theta].
                [  0,  0,  -v * sin(theta) ]
        A_c =   [  0,  0,   v * cos(theta) ]
                [  0,  0,         0        ]

        Control Jacobian B_c (3x2):
        How the state rates change with respect to the controls [v, omega].
                [     cos(theta),                  0              ]
        B_c =   [     sin(theta),                  0              ]
                [ (1/L) * tan(omega),    (v/L) * (1/cos^2(omega)) ]
        """

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



class DubinsCarVisualizer:
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
