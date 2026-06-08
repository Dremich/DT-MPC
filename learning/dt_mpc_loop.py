"""Main DT-MPC training loop."""

import jax

from solvers.optimal_control import DDPSolver
from solvers.ocp_interface import OCP   
from solvers.tube_mpc import TubeMPC
from learning.doc_engine import LearningDOC
import jax.numpy as jnp
import jaxlib

class DTMPCTrainer:

    @jax.custom_vjp
    def solve_mpc(params, initial_state, ocp):
        """
        The Forward Pass: Just runs your standard DDP solver.
        (Make sure your run_ddp takes alpha as an argument and passes it to the system!)
        """
        new_ocp = ocp.copy(params=params)  # Create a new OCP with the updated parameters

        X_opt, U_opt, _ = DDPSolver.run_ddp(new_ocp, initial_state)
        return X_opt, U_opt

    @staticmethod
    def solve_mpc_fwd(params, initial_state, ocp):
        """
        Saves the Riccati matrices (solver_data) for the backward pass.
        """
        new_ocp = ocp.copy(params=params)  # Create a new OCP with the updated parameters

        X_opt, U_opt, solver_data = DDPSolver.run_ddp(new_ocp, initial_state)
        
        # Pack everything we need for Algorithm 3 & 4 into the "residual"
        res = (initial_state, params, X_opt, U_opt, solver_data, ocp)
        return (X_opt, U_opt), res
    
    @staticmethod
    def solve_mpc_bwd(res, g):
        """
        The Backward Pass: Bypasses DDP and runs Theorem 5 (Algorithms 3 & 4).
        """
        initial_state, params, X_opt, U_opt, solver_data, ocp = res
        L_X, L_U = g # These are passed in automatically by jax.grad from your Loss function!
        
        # 1. Recompute the local Jacobians along the trajectory
        get_AB = jax.vmap(ocp.system.discrete_jacobians, in_axes=(0, 0, None))
        A_seq, B_seq = get_AB(X_opt[:-1], U_opt, ocp.dt)
        
        # 2. Compute f_theta (The Jacobian of the dynamics specifically w.r.t alpha)
        f_theta, L_theta_x, L_theta_u, xi_theta, phi_theta_x = ocp.param_gradients()
        
        # 3. Build the generalized derivatives dictionary.
        # Since alpha is not in the cost function, all cross-terms are zero.
        nx = ocp.system.state_dim
        nu = ocp.system.control_dim
        n_theta = ocp.system.param_dim # We are only learning 1 parameter: alpha
        
        derivs = {
            "f_theta": f_theta.reshape((ocp.horizon, nx, n_theta)),
            "L_theta_x": L_theta_x.reshape((ocp.horizon, n_theta, nx)),
            "L_theta_u": L_theta_u.reshape((ocp.horizon, n_theta, nu)),
            "xi_theta": xi_theta.reshape((nx, n_theta)),
            "phi_theta_x": phi_theta_x.reshape((n_theta, nx))
        }
        
        # 4. Run your new Adjoint Learning classes!
        bwd_data = LearningDOC.adjoint_backward_pass(
            L_X, L_U, A_seq, B_seq, solver_data["Vxx"], solver_data["cuu"], solver_data["Kx"]
        )
        
        grad_alpha = LearningDOC.adjoint_forward_pass(
            bwd_data, solver_data["Kx"], A_seq, B_seq, solver_data["Vxx"], derivs
        )
        
        # JAX requires returning a gradient for EVERY input to solve_mpc.
        # We only care about alpha, so we return None for initial_state and ocp.
        return (jnp.squeeze(grad_alpha), None, None)

    # Bind the custom rules to the main function
    solve_mpc.defvjp(solve_mpc_fwd, solve_mpc_bwd)


    @staticmethod
    def compute_total_loss(params, initial_state, ocp, X_nom):
        """
        Full calculation
        """
        X_anc, U_anc = DTMPCTrainer.solve_mpc(params, initial_state, ocp)
    
        # 2. Evaluate Equation 9
        loss = DTMPCTrainer.loss(X_nom, X_anc)
        return loss
    

    get_param_gradients = jax.jit(jax.grad(compute_total_loss, argnums=0), static_argnums=(2,))

    @staticmethod
    def loss(X_nominal, X_ancillary):
        """
        Computes L(tau*(theta), tau_bar(theta_bar)) and its derivatives.
        Equation 9: L = ||tau_ancillary - tau_nominal||^2 + ||b*||^2
        where b* is the barrier violation along tau* (can be computed from the trajectory data).
        """
        tau_ancillary = X_ancillary[:, :-1]
        tau_nominal = X_nominal[:, :-1]

        b_ancillary = X_ancillary[:, -1]
        # b_nominal = X_nominal[:, -1]

        tracking_loss = jnp.sum((tau_ancillary - tau_nominal) ** 2)
        safety_loss = jnp.sum(b_ancillary ** 2)

        loss = tracking_loss + safety_loss
        return loss

    # ---------------------------------------------------------
    # The Unified End-to-End Pipeline
    # ---------------------------------------------------------
    @staticmethod
    def _unified_pipeline(params_ancillary, params_nominal, initial_state, ancillary_ocp, nominal_ocp):
        """
        Runs both solvers EXACTLY ONCE, computes the loss, and returns the trajectories as auxiliary data.
        """
        # Forward pass 1: Nominal
        X_nom, U_nom = DTMPCTrainer.solve_mpc(params_nominal, initial_state, nominal_ocp)
        
        # Forward pass 2: Ancillary
        X_anc, U_anc = DTMPCTrainer.solve_mpc(params_ancillary, initial_state, ancillary_ocp)

        # Compute Loss (Equation 9)
        loss = DTMPCTrainer.loss(X_nom, X_anc)

        # Return: (Differentiable Loss, Auxiliary Trajectory Data)
        return loss, (X_nom, U_nom, X_anc, U_anc)

    # JIT compile the master function ONCE.
    # argnums=(0, 1) computes gradients for BOTH theta and theta_bar.
    # static_argnums=(3, 4) keeps the OCP objects static.
    # has_aux=True tells JAX the second return item is just data payload, not part of the loss gradient.
    _unified_grad_fn = jax.jit(
        jax.value_and_grad(_unified_pipeline, argnums=(0, 1), has_aux=True), 
        static_argnums=(3, 4)
    )

    # ---------------------------------------------------------
    # Algorithm 2: DT-MPC Step
    # ---------------------------------------------------------
    @classmethod
    def train_step(cls, current_state: np.ndarray, tube_controller: 'TubeMPC'):
        """
        Implements a single epoch of Algorithm 2 with zero redundant solver calls.
        """
        # ONE call does it all: Solves both DDPs once, gets the loss, gets the trajectories, and gets both gradients!
        (loss, aux_data), (grad_theta, grad_theta_bar) = cls._unified_grad_fn(
            tube_controller.ancillary_problem.params,
            tube_controller.nominal_problem.params,
            current_state,
            tube_controller.ancillary_problem,
            tube_controller.nominal_problem
        )

        # Unpack the smuggled trajectories
        X_nom, U_nom, X_anc, U_anc = aux_data

        # Apply gradient descent step to update theta and theta_bar
        tube_controller.ancillary_problem.update_params(grad_theta)
        tube_controller.nominal_problem.update_params(grad_theta_bar)

        # Return the first control actions to step the true and nominal dynamics forward
        return U_anc[0], U_nom[0]



# what is the custom vjp?
# What does binding it mean?
# Can we call tube_mpc here instead of solve_mpc?
# Rn we aren't updated the ancillary_ocp with the nominal trajectory
# Many problems, needs more work
