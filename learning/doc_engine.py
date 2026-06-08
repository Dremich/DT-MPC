"""Differentiable optimal control engine for DT-MPC."""

import jax
import jax.numpy as jnp


class LearningDOC:      

#   get_AB_seq = jax.vmap(OCP.system.discrete_jacobians, in_axes=(0,0,None)))
#   A_seq, B_seq = get_AB_seq(X_ancillary[:-1], U_ancillary, dt)

    @staticmethod
    def adjoint_backward_pass(L_X : jnp.ndarray, L_U: jnp.ndarray, A: jnp.ndarray, B: jnp.ndarray, Vxx: jnp.ndarray, cuu: jnp.ndarray, Kx: jnp.ndarray) -> dict:
        """
        Implements Algorithm 3: DOC Backward Pass.
        Returns V_x_tilde, V_xx, k_tilde, and K.
        """
        horizon = L_U.shape[0]
        nu = L_U.shape[1]

        # Line 1: Initialize V_x_tilde
        V_x_tilde_N = L_X[-1]

        def backward_step(V_x_tilde_next, k):
            # Lines 3 & 4: Calculate the Q-function gradients
            Q_x_tilde = L_X[k] + A[k].T @ V_x_tilde_next
            Q_u_tilde = L_U[k] + B[k].T @ V_x_tilde_next
            
            # Line 7: Reconstruct Quu (since we need its inverse for k_tilde)
            # We add the same 1e-4 regularization we used in DDP to prevent singular matrices
            Quu = cuu[k] + B[k].T @ Vxx[k+1] @ B[k] + jnp.eye(nu) * 1e-4
            Quu_inv = jnp.linalg.inv(Quu)
            
            # Line 8: Calculate the feedforward gradient correction
            Kf_tilde = -Quu_inv @ Q_u_tilde
            
            # Line 10: Calculate the new V_x_tilde to pass backwards
            # Note: Q_xu is just the transpose of Q_ux. 
            # By math identity, Kx = -Quu^-1 @ Q_ux, so Q_xu[k] = -Kx[k].T @ Quu
            Q_xu = -Kx[k].T @ Quu 
            V_x_tilde_curr = Q_x_tilde + Q_xu @ Kf_tilde
            
            return V_x_tilde_curr, (V_x_tilde_curr, Kf_tilde)
        
        V_x_tilde_0, (V_x_tilde_scan, Kf_tilde_seq) = jax.lax.scan(
            backward_step, 
            V_x_tilde_N, 
            jnp.arange(horizon),
            reverse=True
        )
        
        # Append the terminal V_x_tilde to the end of the sequence for use in the forward pass
        V_x_tilde_seq = jnp.vstack((V_x_tilde_scan, V_x_tilde_N.reshape(1, -1)))

        return { 
            "V_x_tilde_seq": V_x_tilde_seq,
            "Kf_tilde": Kf_tilde_seq
        }

    @staticmethod
    def adjoint_forward_pass(bwd_data: dict, derivs: dict, Kx: jnp.ndarray, A: jnp.ndarray, B: jnp.ndarray, Vxx: jnp.ndarray) -> jnp.ndarray:
        """
        Implements Algorithm 4: DOC Forward Pass.
        Returns the gradient of the loss nabla_theta L.
        """

        horizon = A.shape[0]
        nx = A.shape[1]

        # Lines 1-3: Initialize delta_x, delta_lambda, and the initial gradient
        dx_0 = jnp.zeros(nx)
        dlambda_0 = bwd_data["V_x_tilde_seq"][0]
        L_theta_0 = derivs["xi_theta"].T @ dlambda_0

        def forward_step(carry, k):
            dx, L_theta = carry

            # Line 5
            du = bwd_data["Kf_tilde"][k] + Kx[k] @ dx

            # Line 6
            dx_next = A[k] @ dx + B[k] @ du

            # Line 7
            dlambda_next = bwd_data["V_x_tilde_seq"][k+1] + Vxx[k+1] @ dx_next

            # Line 8: Accumulate the gradient contribution from this step
            L_theta_step = derivs["L_theta_x"][k] @ dx + derivs["L_theta_u"][k] @ du + derivs["f_theta"][k].T @ dlambda_next
            L_theta_next = L_theta + L_theta_step

            return (dx_next, L_theta_next), None

        # Lines 4: Run the forward pass through the trajectory
        (dx_N, L_theta), _ = jax.lax.scan(
            forward_step,
            (dx_0, L_theta_0),
            jnp.arange(horizon)
        )

        # Line 10 
        L_theta += derivs["terminal_c_theta_x"] @ dx_N  # Add terminal cost contribution

        return L_theta


    def compute_gradient(
        self, upper_loss_derivs: dict, trajectory_data: dict, dynamics
    ) -> jnp.ndarray:
        """
        Implements Algorithm 1: Differentiable Optimal Control (DOC).
        Coordinates the backward and forward passes to output final nabla_theta L.
        """
        raise NotImplementedError(
            "DifferentiableOptimalControl.compute_gradient is not implemented yet."
        )
