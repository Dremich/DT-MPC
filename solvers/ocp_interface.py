from dataclasses import dataclass
from dynamics.base_system import DynamicalSystem
from solvers.costs import BaseCost 

import jax
import jax.numpy as jnp

@dataclass
class OCP:
    """The universal interface between MPC controllers and the DDP solver."""
    system: DynamicalSystem
    stage_cost: BaseCost
    terminal_cost: BaseCost
    horizon: int
    dt: float
    params: jnp.ndarray = None  # Generalized parameter array for learning (e.g., alpha)



    def update_params(self, grad_params: jnp.ndarray, learning_rate: float = 0.01):
        """
        Applies the gradient descent step to update the OCP parameters.
        (Corresponds to Line 6 of Algorithm 2)
        """
        self.params = self.params - learning_rate * grad_params
        
        # Optional: Add projection here if certain parameters must remain positive 
        # e.g., self.params = jnp.clip(self.params, a_min=0.0)

    def param_gradients(self, X_seq: jnp.ndarray, U_seq: jnp.ndarray):
        """
        Automatically computes the cross-derivatives of the dynamics and costs 
        with respect to the generalized parameter array `self.params`.
        """
        k_seq = jnp.arange(self.horizon)
        
        # 1. Dynamics Jacobian w.r.t parameters (f_theta)
        # Assuming system.step takes (x, u, dt, params)
        def dyn_fn(x, u):
            return self.system.step(x, u, self.dt, self.params["dynamics"])
        
        f_theta_fn = jax.vmap(jax.jacobian(dyn_fn, argnums=3), in_axes=(0, 0))
        f_theta = f_theta_fn(X_seq[:-1], U_seq)  # Shape: (horizon, nx, n_theta)

        # 2. Stage Cost Cross-Derivatives (L_theta_x, L_theta_u)
        # Using jax.jacobian twice gives us the mixed partial derivatives
        def stage_fn(x, u, k):
            return self.stage_cost.evaluate(x, u, k, self.params["stage_cost"])

        L_theta_x_fn = jax.vmap(jax.jacobian(jax.jacobian(stage_fn, argnums=3), argnums=0), in_axes=(0, 0, 0))
        L_theta_u_fn = jax.vmap(jax.jacobian(jax.jacobian(stage_fn, argnums=3), argnums=1), in_axes=(0, 0, 0))
        
        L_theta_x = L_theta_x_fn(X_seq[:-1], U_seq, k_seq)  # Shape: (horizon, n_theta, nx)
        L_theta_u = L_theta_u_fn(X_seq[:-1], U_seq, k_seq)  # Shape: (horizon, n_theta, nu)

        # 3. Terminal Cost Cross-Derivative (phi_theta_x)
        def term_fn(x):
            return self.terminal_cost.evaluate(x, self.params["terminal_cost"])

        phi_theta_x_fn = jax.jacobian(jax.jacobian(term_fn, argnums=1), argnums=0)
        phi_theta_x = phi_theta_x_fn(X_seq[-1])  # Shape: (n_theta, nx)

        # 4. Initial State Jacobian (xi_theta)
        # Since the initial state is provided by the environment, it doesn't depend on theta.
        nx = self.system.state_dim
        n_theta = self.params["dynamics"].shape[0] if hasattr(self.params["dynamics"], 'shape') else 1
        xi_theta = jnp.zeros((nx, n_theta))

        ##TODO: CHECK THE SHAPE OF n_theta HERE.

        return f_theta, L_theta_x, L_theta_u, xi_theta, phi_theta_x
