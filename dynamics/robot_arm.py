"""3-link robot arm dynamics (SM5.3 / Appendix J.C of arXiv:2308.08426).

The arm is modeled as a 6-DOF double integrator in joint space (page 20:
"6-dimensional double integrator ... a linear system"). Each of the three links
has a pitch and a yaw angle, giving 6 generalized coordinates q and 6 angular
velocities, for 12 states; the 6 controls are joint torques (q_ddot = u, unit
inertia). The nonlinearity of the task lives in the *forward kinematics* (used by
the end-effector cost) and the obstacle constraints, not the dynamics.

Forward-kinematics convention (the paper underspecifies it, so we fix a concrete,
differentiable one): each link's orientation is expressed in the base frame by
spherical angles -- yaw = azimuth about +z, pitch = elevation from the xy-plane:

    d_i(p_i, y_i) = [cos(p_i) cos(y_i), cos(p_i) sin(y_i), sin(p_i)]

with joint positions p_0 = 0, p_i = p_{i-1} + L_i * d_i, and the end-effector
e = p_3. Link lengths are (1.0, 1.5, 1.0) m, so the reachable radius is 3.5 m.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple

from dynamics.base_system import DynamicalSystem

jax.config.update("jax_enable_x64", True)

LINK_LENGTHS = (1.0, 1.5, 1.0)


class RobotArm3Link(DynamicalSystem):
    """6-DOF double integrator for a 3-link arm.

    State:   x = [q (6), q_dot (6)]  with q = [p1, y1, p2, y2, p3, y3]
             (pitch_i, yaw_i for link i).
    Control: u = [tau (6)]           (joint angular accelerations).
    """

    def __init__(self, link_lengths=LINK_LENGTHS):
        self.L = tuple(float(l) for l in link_lengths)
        self._n_joints = 2 * len(self.L)          # 6 angles
        self._state_dim = 2 * self._n_joints       # 12
        self._control_dim = self._n_joints         # 6

        # Constant double-integrator Jacobians (precomputed once).
        n = self._n_joints
        self._A_c = np.block([
            [np.zeros((n, n)), np.eye(n)],
            [np.zeros((n, n)), np.zeros((n, n))],
        ])
        self._B_c = np.block([[np.zeros((n, n))], [np.eye(n)]])

        # Cached FK Jacobian of the end-effector w.r.t. the joint angles.
        self._jac_ee = jax.jit(jax.jacobian(self.end_effector))

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def control_dim(self) -> int:
        return self._control_dim

    @property
    def n_joints(self) -> int:
        return self._n_joints

    # ----- dynamics (linear double integrator) -----
    def dynamics(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        q_dot = x[self._n_joints:]
        return jnp.concatenate([q_dot, u])

    def continuous_jacobians(self, x: jnp.ndarray, u: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return jnp.asarray(self._A_c), jnp.asarray(self._B_c)

    # ----- forward kinematics -----
    def link_directions(self, q: jnp.ndarray) -> jnp.ndarray:
        """Unit direction of each link in the base frame, shape (n_links, 3)."""
        pitch = q[0::2]
        yaw = q[1::2]
        return jnp.stack([
            jnp.cos(pitch) * jnp.cos(yaw),
            jnp.cos(pitch) * jnp.sin(yaw),
            jnp.sin(pitch),
        ], axis=1)

    def joint_positions(self, q: jnp.ndarray) -> jnp.ndarray:
        """Cumulative joint positions p1, p2, p3 in the base frame, shape (n_links, 3)."""
        dirs = self.link_directions(q)
        lengths = jnp.asarray(self.L)[:, None]
        return jnp.cumsum(lengths * dirs, axis=0)

    def end_effector(self, q: jnp.ndarray) -> jnp.ndarray:
        """End-effector position e = p3, shape (3,)."""
        return self.joint_positions(q)[-1]

    def collision_points(self, q: jnp.ndarray, per_link: int = 2) -> jnp.ndarray:
        """Points sampled along the links (for obstacle/plane checks), shape (M, 3).

        Includes the base, each joint, and ``per_link`` interior samples per link.
        """
        dirs = self.link_directions(q)
        joints = self.joint_positions(q)
        base = jnp.zeros((1, 3))
        prev = jnp.concatenate([base, joints[:-1]], axis=0)  # start of each link
        pts = [base, joints]
        for j in range(1, per_link + 1):
            frac = j / (per_link + 1)
            pts.append(prev + frac * jnp.asarray(self.L)[:, None] * dirs)
        return jnp.concatenate(pts, axis=0)

    def end_effector_jacobian(self, q: jnp.ndarray) -> jnp.ndarray:
        """d e / d q, shape (3, n_joints)."""
        return self._jac_ee(q)


class RobotArm3DVisualizer:
    """3D visualisation of the arm trajectory, obstacles (vertical cylinders) and target."""

    @staticmethod
    def visualize(arm, q_traj, obstacles, target=None, figsize=(9, 8), cyl_height=3.0):
        """Render the arm task like Fig. 5: solid vertical red obstacle cylinders, the
        articulated arm at the start (green) and final (grey) poses, the end-effector
        path, and the target."""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        q_traj = np.asarray(q_traj)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        # Solid vertical obstacle cylinders (red), z in [0, cyl_height].
        th = np.linspace(0, 2 * np.pi, 40)
        z_lin = np.linspace(0.0, cyl_height, 2)
        th_g, z_g = np.meshgrid(th, z_lin)
        for cx, cy, r in np.asarray(obstacles):
            xg, yg = cx + r * np.cos(th_g), cy + r * np.sin(th_g)
            ax.plot_surface(xg, yg, z_g, color="firebrick", alpha=0.55, linewidth=0, antialiased=True)
            # Cap the top of the cylinder.
            cap_r = np.linspace(0, r, 2)[:, None] * np.ones_like(th)[None, :]
            ax.plot_surface(cx + cap_r * np.cos(th), cy + cap_r * np.sin(th),
                            np.full_like(cap_r, cyl_height), color="firebrick", alpha=0.55, linewidth=0)

        # End-effector path.
        ee = np.array([np.asarray(arm.end_effector(jnp.asarray(q))) for q in q_traj])
        ax.plot(ee[:, 0], ee[:, 1], ee[:, 2], "-", color="royalblue", lw=2, label="end-effector path")

        # Arm at the start (green) and final (grey) poses.
        for q, color, lbl in [(q_traj[0], "seagreen", "arm (start)"), (q_traj[-1], "dimgray", "arm (final)")]:
            joints = np.vstack([np.zeros(3), np.asarray(arm.joint_positions(jnp.asarray(q)))])
            ax.plot(joints[:, 0], joints[:, 1], joints[:, 2], "-", color=color, lw=4, solid_capstyle="round")
            ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color=color, s=35, label=lbl)

        if target is not None:
            ax.scatter(*target, color="gold", edgecolors="k", s=180, marker="*",
                       depthshade=False, label="target")

        # Aesthetics: equal aspect, axis limits and viewpoint similar to Fig. 5.
        all_x = np.concatenate([obstacles[:, 0], ee[:, 0], [0.0]])
        all_y = np.concatenate([obstacles[:, 1], ee[:, 1], [0.0]])
        ax.set_xlim(all_x.min() - 0.8, all_x.max() + 0.8)
        ax.set_ylim(all_y.min() - 0.8, all_y.max() + 0.8)
        ax.set_zlim(0.0, cyl_height)
        ax.set_box_aspect((ax.get_xlim()[1] - ax.get_xlim()[0],
                           ax.get_ylim()[1] - ax.get_ylim()[0],
                           cyl_height))
        ax.view_init(elev=22, azim=-58)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
        ax.set_title("3-Link Robot Arm — DT-MPC (paper-faithful task)")
        ax.legend(loc="upper left")
        plt.tight_layout()
        return fig
