import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod
from typing import Optional, Tuple

jax.config.update("jax_enable_x64", True)

# ==================================================================================================
# Contains only cost function definitions and their derivatives. Dynamic derivitives are in Dynamics
# ==================================================================================================

class BaseCost(ABC):
    """Abstract base class for cost functions"""

    @abstractmethod
    def evaluate(self, x: jnp.ndarray, u: Optional[jnp.ndarray] = None, k: Optional[int] = None) -> jnp.ndarray:
        """
        Returns scalar cost at a given state and control.
        
        u required for stage costs, optional for terminal cost.
        k is the optional timestep index for time-varying costs.
        
        Output is a scalar (0D array) representing the cost (using jnp array to maintain JAX compatibility).
        """
        pass

    @abstractmethod
    def get_derivatives(self, x: jnp.ndarray, u: Optional[jnp.ndarray] = None, k: Optional[int] = None) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Returns the necessary derivatives for DDP as a tuple:

        l_x: Gradient of stage cost w.r.t. state
        l_u: Gradient of stage cost w.r.t. control
        l_xx: Hessian of stage cost w.r.t. state
        l_uu: Hessian of stage cost w.r.t. control
        l_xu: Mixed partials of stage cost w.r.t. state and control
        """
        pass
    
    def update_reference(self, x_ref: Optional[jnp.ndarray] = None, u_ref: Optional[jnp.ndarray] = None):
        """ Updates the cost function with optional reference trajectory and control"""
        if x_ref is not None:
            self.x_ref = jnp.array(x_ref)
        if u_ref is not None:
            self.u_ref = jnp.array(u_ref)

class QuadraticCost(BaseCost):
    """Stage cost with optional reference tracking"""

    def __init__(self, Q: jnp.ndarray, R: jnp.ndarray, x_ref: Optional[jnp.ndarray] = None, u_ref: Optional[jnp.ndarray] = None):
        # All quadratic costs require Q and R
        self.Q = Q
        self.R = R

        # For anscillary tube MPC controller, tracking is required as well
        self.x_ref = jnp.array(x_ref) if x_ref is not None else None
        self.u_ref = jnp.array(u_ref) if u_ref is not None else None

    def evaluate(self, x: jnp.ndarray, u: Optional[jnp.ndarray] = None, k: Optional[int] = None) -> jnp.ndarray:
        """
        Computes the quadratic cost at a given state and control. 
        """
        x_ref = self.x_ref
        u_ref = self.u_ref

        if k is not None:
            if x_ref is not None and x_ref.ndim > 1:
                x_ref = x_ref[k]
            if u_ref is not None and u_ref.ndim > 1:
                u_ref = u_ref[k]

        dx = x if x_ref is None else x - x_ref
        du = u if u_ref is None else u - u_ref

        return dx.T @ self.Q @ dx + du.T @ self.R @ du

    def get_derivatives(self, x: jnp.ndarray, u: jnp.ndarray, k: Optional[int] = None) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Manual derivatives for QuadraticCost (faster and avoids JIT stale state issues)"""
        x_ref = self.x_ref
        u_ref = self.u_ref

        if k is not None:
            if x_ref is not None and x_ref.ndim > 1:
                x_ref = x_ref[k]
            if u_ref is not None and u_ref.ndim > 1:
                u_ref = u_ref[k]

        dx = x if x_ref is None else x - x_ref
        du = u if u_ref is None else u - u_ref

        l_x = 2.0 * self.Q @ dx
        l_u = 2.0 * self.R @ du
        l_xx = 2.0 * self.Q
        l_uu = 2.0 * self.R
        l_xu = jnp.zeros((x.shape[0], u.shape[0]))
        
        return l_x, l_u, l_xx, l_uu, l_xu


class TerminalCost(BaseCost):
    """Terminal cost is applied only to the final state in horizon"""
    def __init__(self, P: jnp.ndarray, x_ref: Optional[jnp.ndarray] = None):
        self.P = P
        self.x_ref = jnp.array(x_ref) if x_ref is not None else None

    def evaluate(self, x: jnp.ndarray, u: Optional[jnp.ndarray] = None, k: Optional[int] = None) -> jnp.ndarray:
        """Computes terminal cost at final state."""
        dx = x if self.x_ref is None else x - self.x_ref
        return dx.T @ self.P @ dx

    def get_derivatives(self, x: jnp.ndarray, u: Optional[jnp.ndarray] = None, k: Optional[int] = None) -> Tuple[jnp.ndarray, Optional[jnp.ndarray], jnp.ndarray, Optional[jnp.ndarray], Optional[jnp.ndarray]]:
        """Manual derivatives for TerminalCost"""
        dx = x if self.x_ref is None else x - self.x_ref

        phi_x = 2.0 * self.P @ dx
        phi_xx = 2.0 * self.P

        return phi_x, None, phi_xx, None, None


# ==================================================================================================
# End-effector (task-space) costs for the robot-arm system (SM5.3 / Appendix J.C).
#
# The cost is reparameterized in terms of a nonlinear forward-kinematics map
# e = fk(q), where q are the joint angles (the first ``n_joints`` state dims). The
# state layout is assumed to be the safety-embedded form [q, q_dot, b] (barrier last).
# A Gauss-Newton Hessian (2 J^T W J, with J = de/dq) is used so the cost Hessian is
# always PSD and stable for DDP/iLQR, matching the paper's solver.
# ==================================================================================================

class EndEffectorCost(BaseCost):
    """Nominal stage cost: ||fk(q) - target||^2_W + q_b * b^2 + u^T R u."""

    def __init__(self, fk_fn, W: jnp.ndarray, R: jnp.ndarray, target: jnp.ndarray,
                 n_joints: int, qb: float = 0.0, jac_fn=None):
        self.fk_fn = fk_fn
        self.W = jnp.asarray(W)
        self.R = jnp.asarray(R)
        self.target = jnp.asarray(target)
        self.n_joints = int(n_joints)
        self.qb = float(qb)
        self._jac = jax.jit(jax.jacobian(fk_fn)) if jac_fn is None else jac_fn

    def evaluate(self, x: jnp.ndarray, u: Optional[jnp.ndarray] = None, k: Optional[int] = None) -> jnp.ndarray:
        q = x[:self.n_joints]
        de = self.fk_fn(q) - self.target
        cost = de.T @ self.W @ de + self.qb * x[-1] ** 2
        if u is not None:
            cost = cost + u.T @ self.R @ u
        return cost

    def get_derivatives(self, x: jnp.ndarray, u: jnp.ndarray, k: Optional[int] = None) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        nj, nx, nu = self.n_joints, x.shape[0], u.shape[0]
        q = x[:nj]
        de = self.fk_fn(q) - self.target
        J = self._jac(q)                      # (3, nj)

        grad_q = 2.0 * J.T @ self.W @ de
        Hqq = 2.0 * J.T @ self.W @ J          # Gauss-Newton (PSD)

        l_x = jnp.zeros(nx).at[:nj].set(grad_q).at[nx - 1].set(2.0 * self.qb * x[-1])
        l_xx = jnp.zeros((nx, nx)).at[:nj, :nj].set(Hqq).at[nx - 1, nx - 1].set(2.0 * self.qb)
        l_u = 2.0 * self.R @ u
        l_uu = 2.0 * self.R
        l_xu = jnp.zeros((nx, nu))
        return l_x, l_u, l_xx, l_uu, l_xu


class EndEffectorTerminalCost(BaseCost):
    """Nominal terminal cost: ||fk(q) - target||^2_W + q_b * b^2."""

    def __init__(self, fk_fn, W: jnp.ndarray, target: jnp.ndarray,
                 n_joints: int, qb: float = 0.0, jac_fn=None):
        self.fk_fn = fk_fn
        self.W = jnp.asarray(W)
        self.target = jnp.asarray(target)
        self.n_joints = int(n_joints)
        self.qb = float(qb)
        self._jac = jax.jit(jax.jacobian(fk_fn)) if jac_fn is None else jac_fn

    def evaluate(self, x: jnp.ndarray, u: Optional[jnp.ndarray] = None, k: Optional[int] = None) -> jnp.ndarray:
        q = x[:self.n_joints]
        de = self.fk_fn(q) - self.target
        return de.T @ self.W @ de + self.qb * x[-1] ** 2

    def get_derivatives(self, x: jnp.ndarray, u: Optional[jnp.ndarray] = None, k: Optional[int] = None) -> Tuple[jnp.ndarray, Optional[jnp.ndarray], jnp.ndarray, Optional[jnp.ndarray], Optional[jnp.ndarray]]:
        nj, nx = self.n_joints, x.shape[0]
        q = x[:nj]
        de = self.fk_fn(q) - self.target
        J = self._jac(q)

        phi_x = jnp.zeros(nx).at[:nj].set(2.0 * J.T @ self.W @ de).at[nx - 1].set(2.0 * self.qb * x[-1])
        phi_xx = jnp.zeros((nx, nx)).at[:nj, :nj].set(2.0 * J.T @ self.W @ J).at[nx - 1, nx - 1].set(2.0 * self.qb)
        return phi_x, None, phi_xx, None, None
