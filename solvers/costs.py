import jax
import jax.numpy as jnp
from typing import Tuple, Optional
from abc import ABC, abstractmethod


# =====================================================================
# Contains only cost function definitions and their derivatives. Dynamic derivitives are in Dynamics
# =====================================================================

class BaseCost(ABC):
    """Abstract base class for cost functions"""

    @abstractmethod
    def evaluate(self, x: jnp.ndarray, u: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Returns scalar cost at a given state and control.
        
        u required for stage costs, optional for terminal cost.
        
        Output is a scalar (0D array) representing the cost (using jnp array to maintain JAX compatibility).
        """
        pass

    def get_derivatives(self, x: jnp.ndarray, u: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        # Because we are using JAX, this method can be implemented in the base case
        # Computing all necessary derivatives for DDP in one go is more efficient than separate calls
        """
        Returns the necessary derivatives for DDP as a tuple:

        l_x: Gradient of stage cost w.r.t. state
        l_u: Gradient of stage cost w.r.t. control
        l_xx: Hessian of stage cost w.r.t. state
        l_uu: Hessian of stage cost w.r.t. control
        l_xu: Mixed partials of stage cost w.r.t. state and control
        """
        # Using JAX to compute gradients and hessians automatically
        if u is None:
            # Terminal cost case derivitives (x only, no control u)
            phi_x = jax.grad(self.evaluate, argnums=0)(x)
            phi_xx = jax.hessian(self.evaluate, argnums=0)(x)
            return phi_x, None, phi_xx, None, None
        
        # State cost case
        l_x = jax.grad(self.evaluate, argnums=0)(x, u)
        l_u = jax.grad(self.evaluate, argnums=1)(x, u)
        l_xx = jax.hessian(self.evaluate, argnums=0)(x, u)
        l_uu = jax.hessian(self.evaluate, argnums=1)(x, u)
        l_xu = jax.jacobian(jax.grad(self.evaluate, argnums=0), argnums=1)(x, u)
        return l_x, l_u, l_xx, l_uu, l_xu

class QuadraticCost(BaseCost):
    """Stage cost with optional reference tracking"""

    def __init__(self, Q: jnp.ndarray, R: jnp.ndarray, x_ref: Optional[jnp.ndarray] = None, u_ref: Optional[jnp.ndarray] = None):
        # All quadratic costs require Q and R
        self.Q = Q
        self.R = R

        # For anscillary tube MPC controller, tracking is required as well
        self.x_ref = x_ref if x_ref is not None else None
        self.u_ref = u_ref if u_ref is not None else None

    def evaluate(self, x: jnp.ndarray, u: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Computes the quadratic cost at a given state and control. 
        
        If reference provided, cost is computed on deviation from ref.
        """
        dx = x if self.x_ref is None else x - self.x_ref
        du = u if self.u_ref is None else u - self.u_ref

        return dx.T @ self.Q @ dx + du.T @ self.R @ du


class TerminalCost(BaseCost):
    """Terminal cost is applied only to the final state in horizon"""
    def __init__(self, P: jnp.ndarray, x_ref: Optional[jnp.ndarray] = None):
        self.P = P
        self.x_ref = x_ref if x_ref is not None else None

    def evaluate(self, x: jnp.ndarray, u: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Computes terminal cost at final state."""
        dx = x if self.x_ref is None else x - self.x_ref

        return dx.T @ self.P @ dx
        