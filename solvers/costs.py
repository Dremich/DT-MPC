import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod
from typing import Optional, Tuple

jax.config.update("jax_enable_x64", True)


class BaseCost(ABC):

    @abstractmethod
    def evaluate(
        self,
        x: jnp.ndarray,
        u: Optional[jnp.ndarray] = None,
        k: int = 0,
    ) -> jnp.ndarray:
        """Scalar cost at state x, control u, time step k."""
        pass

    def get_derivatives(
        self,
        x: jnp.ndarray,
        u: Optional[jnp.ndarray] = None,
        k: int = 0,
    ) -> Tuple:
        """
        Returns (l_x, l_u, l_xx, l_uu, l_xu) via JAX autodiff.
        For terminal costs pass u=None; l_u, l_uu, l_xu are returned as None.
        """
        if u is None:
            phi_x  = jax.grad(lambda x_: self.evaluate(x_, None, k))(x)
            phi_xx = jax.hessian(lambda x_: self.evaluate(x_, None, k))(x)
            return phi_x, None, phi_xx, None, None

        l_x  = jax.grad(lambda x_: self.evaluate(x_, u,  k))(x)
        l_u  = jax.grad(lambda u_: self.evaluate(x,  u_, k))(u)
        l_xx = jax.hessian(lambda x_: self.evaluate(x_, u,  k))(x)
        l_uu = jax.hessian(lambda u_: self.evaluate(x,  u_, k))(u)
        # l_xu[i,j] = d²l / (dx_i  du_j)  →  shape (nx, nu)
        l_xu = jax.jacobian(
            lambda u_: jax.grad(lambda x_: self.evaluate(x_, u_, k))(x)
        )(u)
        return l_x, l_u, l_xx, l_uu, l_xu

    def update_reference(
        self,
        x_ref: Optional[jnp.ndarray] = None,
        u_ref: Optional[jnp.ndarray] = None,
    ) -> None:
        """Set a single-point or full-trajectory reference."""
        if x_ref is not None:
            self.x_ref = jnp.array(x_ref)
        if u_ref is not None:
            self.u_ref = jnp.array(u_ref)

    def set_reference_trajectory(
        self,
        x_ref: jnp.ndarray,
        u_ref: Optional[jnp.ndarray] = None,
    ) -> None:
        """Store a full (N+1, nx) state trajectory (and optionally (N, nu) controls)
        so that evaluate(x, u, k) tracks x_ref[k] / u_ref[k]."""
        self.x_ref = jnp.array(x_ref)
        if u_ref is not None:
            self.u_ref = jnp.array(u_ref)


class QuadraticCost(BaseCost):
    """Stage cost  l(x,u,k) = dx.T Q dx + du.T R du
    where dx = x - x_ref[k] (or x - x_ref if x_ref is 1-D)."""

    def __init__(
        self,
        Q: jnp.ndarray,
        R: jnp.ndarray,
        x_ref: Optional[jnp.ndarray] = None,
        u_ref: Optional[jnp.ndarray] = None,
    ):
        self.Q = jnp.array(Q)
        self.R = jnp.array(R)
        self.x_ref = jnp.array(x_ref) if x_ref is not None else None
        self.u_ref = jnp.array(u_ref) if u_ref is not None else None

    def evaluate(
        self,
        x: jnp.ndarray,
        u: Optional[jnp.ndarray] = None,
        k: int = 0,
    ) -> jnp.ndarray:
        if self.x_ref is not None:
            ref_x = self.x_ref[k] if jnp.ndim(self.x_ref) == 2 else self.x_ref
            dx = x - ref_x
        else:
            dx = x

        if u is None:
            return dx @ self.Q @ dx

        if self.u_ref is not None:
            ref_u = self.u_ref[k] if jnp.ndim(self.u_ref) == 2 else self.u_ref
            du = u - ref_u
        else:
            du = u

        return dx @ self.Q @ dx + du @ self.R @ du


class TerminalCost(BaseCost):
    """Terminal cost  phi(x) = dx.T P dx."""

    def __init__(self, P: jnp.ndarray, x_ref: Optional[jnp.ndarray] = None):
        self.P = jnp.array(P)
        self.x_ref = jnp.array(x_ref) if x_ref is not None else None
        self.u_ref = None

    def evaluate(
        self,
        x: jnp.ndarray,
        u: Optional[jnp.ndarray] = None,
        k: int = 0,
    ) -> jnp.ndarray:
        if self.x_ref is not None:
            ref_x = self.x_ref[k] if jnp.ndim(self.x_ref) == 2 else self.x_ref
            dx = x - ref_x
        else:
            dx = x
        return dx @ self.P @ dx
