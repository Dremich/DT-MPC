"""Optimal control solvers for DT-MPC."""

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from .ocp_interface import OCP
from typing import Tuple, Optional, Dict, Any

class DDPSolver:

    @staticmethod
    def run_ddp(ocp: 'OCP', initial_state: np.ndarray, initial_control: Optional[np.ndarray] = None, 
                max_iters: int = 20, cost_threshold: float = 1e-3) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        
        # ---------------------------------------------------------
        # 1. PYTHON PRE-COMPUTATION (Runs exactly once to setup shapes)
        # ---------------------------------------------------------
        nu = ocp.system.control_dim
        nx = ocp.system.state_dim
        horizon = ocp.horizon
        dt = ocp.dt
        
        if initial_control is None:
            uvec_init = jnp.zeros((horizon, nu))
        else:
            uvec_init = jnp.array(initial_control)
            
        xvec_init = jnp.zeros((horizon + 1, nx)).at[0].set(initial_state)
        
        # Extract the pure JAX-traceable math functions from your OOP instances
        step_fn = ocp.system.step
        jacobians_fn = ocp.system.discrete_jacobians
        stage_derivs_fn = ocp.stage_cost.get_derivatives
        term_derivs_fn = ocp.terminal_cost.get_derivatives
        stage_cost_fn = ocp.stage_cost.evaluate
        term_cost_fn = ocp.terminal_cost.evaluate

        # ---------------------------------------------------------
        # 2. THE JIT-COMPILED XLA ENGINE
        # ---------------------------------------------------------
        # static_argnames tells JAX to hardcode these integers into the C++ compilation
        @partial(jax.jit, static_argnames=['max_iters', 'horizon', 'nx', 'nu'])
        def ddp_core(xvec, uvec, max_iters, horizon, nx, nu):
            
            # --- Helper to calculate initial cost ---
            def eval_cost_step(carry, inputs):
                k, x, u = inputs
                return carry, stage_cost_fn(x, u, k)
            
            _, initial_stage_costs = jax.lax.scan(eval_cost_step, None, (jnp.arange(horizon), xvec[:-1], uvec))
            init_cost = jnp.sum(initial_stage_costs) + term_cost_fn(xvec[-1])
            
            # Outer Loop Condition: Continue if i < max_iters AND not converged
            def outer_cond(state):
                i, _, _, _, _, _, _, _, _, converged = state
                return jnp.logical_and(i < max_iters, jnp.logical_not(converged))
                
            # Outer Loop Body: One full iteration of DDP
            def outer_body(state):
                i, x_seq, u_seq, cost, prev_cost, Kx_seq, Kf_seq, Vx_out, Vxx_out, _ = state
                
                # --- BACKWARD PASS (lax.scan reverse=True) ---
                Vx, _, Vxx, _, _ = term_derivs_fn(x_seq[-1])
                
                def backward_step(carry, k):
                    Vx_curr, Vxx_curr = carry
                    xk, uk = x_seq[k], u_seq[k]
                    
                    A, B = jacobians_fn(xk, uk, dt)
                    cx, cu, cxx, cuu, cxu = stage_derivs_fn(xk, uk, k)
                    
                    Qx = cx + A.T @ Vx_curr
                    Qu = cu + B.T @ Vx_curr
                    Qxx = cxx + A.T @ Vxx_curr @ A
                    Quu = cuu + B.T @ Vxx_curr @ B
                    Qux = cxu.T + B.T @ Vxx_curr @ A 
                    
                    Quu += jnp.eye(nu) * 1e-4 
                    Quu_inv = jnp.linalg.inv(Quu)
                    
                    Kf_k = -Quu_inv @ Qu
                    Kx_k = -Quu_inv @ Qux

                    Vx_next  = Qx + Kx_k.T @ Quu @ Kf_k + Kx_k.T @ Qu + Qux.T @ Kf_k
                    Vxx_next = Qxx + Kx_k.T @ Quu @ Kx_k + Kx_k.T @ Qux + Qux.T @ Kx_k 
                    
                    return (Vx_next, Vxx_next), (Kx_k, Kf_k)

                (Vx_final, Vxx_final), (Kx, Kf) = jax.lax.scan(
                    backward_step, (Vx, Vxx), jnp.arange(horizon), reverse=True
                )
                
                # --- FORWARD PASS / LINE SEARCH (lax.while_loop) ---
                def ls_cond(ls_state):
                    ls_iter, alpha, _, _, _, accepted = ls_state
                    return jnp.logical_and(ls_iter < 10, jnp.logical_not(accepted))
                    
                def ls_body(ls_state):
                    ls_iter, alpha, _, _, _, _ = ls_state
                    
                    # Rollout simulation using lax.scan
                    def forward_step(x_curr, k):
                        dx = x_curr - x_seq[k]
                        u_new = u_seq[k] + alpha * Kf[k] + Kx[k] @ dx
                        x_next = step_fn(x_curr, u_new, dt)
                        step_cost = stage_cost_fn(x_curr, u_new, k)
                        return x_next, (x_next, u_new, step_cost)
                        
                    x_final, (x_future, u_new, costs) = jax.lax.scan(
                        forward_step, x_seq[0], jnp.arange(horizon)
                    )
                    
                    # Stack initial state with future states
                    x_new = jnp.vstack((x_seq[0].reshape(1, -1), x_future))
                    new_cost = jnp.sum(costs) + term_cost_fn(x_final)
                    
                    accepted = new_cost < cost
                    next_alpha = jnp.where(accepted, alpha, alpha / 2.0)
                    
                    return (ls_iter + 1, next_alpha, new_cost, x_new, u_new, accepted)
                    
                # Run line search
                init_ls_state = (0, 1.0, cost + 1e6, x_seq, u_seq, False)
                _, _, final_cost, final_x_seq, final_u_seq, ls_success = jax.lax.while_loop(ls_cond, ls_body, init_ls_state)
                
                # Check convergence
                actual_new_cost = jnp.where(ls_success, final_cost, cost)
                actual_x_seq = jnp.where(ls_success, final_x_seq, x_seq)
                actual_u_seq = jnp.where(ls_success, final_u_seq, u_seq)
                
                improvement = cost - actual_new_cost
                converged = jnp.logical_or(
                    jnp.logical_not(ls_success),  # Stop if line search completely fails
                    improvement < cost_threshold  # Stop if improvement is tiny
                )
                
                return (i + 1, actual_x_seq, actual_u_seq, actual_new_cost, cost, Kx, Kf, Vx_final, Vxx_final, converged)

            # --- Initialize and Run Outer Loop ---
            init_state = (
                0, xvec, uvec, init_cost, init_cost, 
                jnp.zeros((horizon, nu, nx)), jnp.zeros((horizon, nu)), 
                jnp.zeros(nx), jnp.zeros((nx, nx)), False
            )
            
            final_state = jax.lax.while_loop(outer_cond, outer_body, init_state)
            
            # Unpack final results
            _, best_x, best_u, _, _, final_Kx, final_Kf, final_Vx, final_Vxx, _ = final_state
            
            return best_x, best_u, final_Kx, final_Kf, final_Vx, final_Vxx

        # ---------------------------------------------------------
        # 3. EXECUTION
        # ---------------------------------------------------------
        # We run an initial un-JITed forward pass to seed the JAX loop
        for k in range(ocp.horizon):
            xvec_init = xvec_init.at[k+1].set(ocp.system.step(xvec_init[k], uvec_init[k], ocp.dt))
            
        # Call the JIT compiled core
        x_opt, u_opt, Kx, Kf, Vx, Vxx = ddp_core(xvec_init, uvec_init, max_iters, horizon, nx, nu)
        
        # Package for Algorithm 1
        solver_data = {
            "Kx": np.array(Kx),
            "Kf": np.array(Kf),
            "Vx": np.array(Vx),
            "Vxx": np.array(Vxx)
        }
        
        # Note: We convert back to numpy arrays at the very end so Daniel's MPC plotting scripts don't break
        return np.array(x_opt), np.array(u_opt), solver_data


    # This is the original un-JITed version for reference and debugging. It is not used in the main codebase.

    # @staticmethod
    # def run_ddp(ocp: OCP, initial_state: np.ndarray, initial_control : Optional[np.ndarray] = None, 
    #             max_iters: int = 20, cost_threshold: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
    #     if initial_control is None:
    #         uvec = np.zeros((ocp.horizon, ocp.system.control_dim))
    #     else:
    #         uvec = np.copy(initial_control)
            
    #     xvec = np.zeros((ocp.horizon + 1, ocp.system.state_dim))
    #     xvec[0] = initial_state
        
    #     # Initial Forward Pass
    #     for k in range(ocp.horizon):
    #         xvec[k+1] = ocp.system.step(xvec[k], uvec[k], ocp.dt)

    #     old_cost = ocp.terminal_cost.evaluate(xvec[-1])
    #     old_cost += sum(ocp.stage_cost.evaluate(xvec[k], uvec[k], k) for k in range(ocp.horizon))            
        
    #     # DDP Core Loop
    #     for _ in range(max_iters):
            
    #         Kx = np.zeros((ocp.horizon, ocp.system.control_dim, ocp.system.state_dim)) # Feedback gain
    #         Kf = np.zeros((ocp.horizon, ocp.system.control_dim))                       # Feedforward gain
            
    #         # Terminal costate
    #         Vx, _, Vxx, _, _ = ocp.terminal_cost.get_derivatives(xvec[-1])
            
    #         # Backwards Pass
    #         for k in reversed(range(ocp.horizon)):
    #             xk, uk = xvec[k], uvec[k]
                
    #             # Local linearization of the problem
    #             A, B = ocp.system.discrete_jacobians(xk, uk, ocp.dt)
                
    #             cx, cu, cxx, cuu, cxu = ocp.stage_cost.get_derivatives(xk, uk, k)
                
    #             # Build Q functions
    #             Qx = cx + A.T @ Vx
    #             Qu = cu + B.T @ Vx
    #             Qxx = cxx + A.T @ Vxx @ A
    #             Quu = cuu + B.T @ Vxx @ B
    #             Qux = cxu.T + B.T @ Vxx @ A 
                
    #             # Regularize and invert Quu
    #             Quu += np.eye(ocp.system.control_dim) * 1e-4 # prevents singular matrices
    #             Quu_inv = np.linalg.inv(Quu)
                
    #             # Find the gains
    #             Kf[k] = -Quu_inv @ Qu
    #             Kx[k] = -Quu_inv @ Qux

    #             # Update the value function derivatives and proceed to next step
    #             Vx  = Qx + Kx[k].T @ Quu @ Kf[k] + Kx[k].T @ Qu + Qux.T @ Kf[k]
    #             Vxx = Qxx + Kx[k].T @ Quu @ Kx[k] + Kx[k].T @ Qux + Qux.T @ Kx[k]      
                
    #         # Forward Pass w/ Line Search
    #         alpha = 1.0
    #         num_search = 10
            
    #         cost = 0.0
    #         for ls_iter in range(num_search):
                
    #             x_new = np.zeros((ocp.horizon + 1, ocp.system.state_dim))
    #             u_new = np.zeros((ocp.horizon, ocp.system.control_dim))
    #             x_new[0] = initial_state
    #             cost = 0.0
                    
    #             # Use faster JITed step if available
    #             step_func = ocp.system.step_jit if hasattr(ocp.system, "step_jit") else ocp.system.step

    #             for k in range(ocp.horizon):
    #                 # Apply the gains: u_new = u_old + alpha*Kf + Kx*(x_new - x_old)
    #                 dx = x_new[k] - xvec[k]
    #                 u_new[k] = uvec[k] + alpha * Kf[k] + Kx[k] @ dx
                    
    #                 # Simulate physics
    #                 x_new[k+1] = step_func(x_new[k], u_new[k], ocp.dt)
    #                 cost += ocp.stage_cost.evaluate(x_new[k], u_new[k], k)
                
    #             cost += ocp.terminal_cost.evaluate(x_new[-1])
                
    #             if cost < old_cost: # Success!
    #                 xvec = x_new
    #                 uvec = u_new
    #                 print(f"Iter {ls_iter}: Accepted alpha = {alpha} with cost {cost:.4f}")
    #                 break
    #             else:
    #                 alpha /= 2.0
    #                 continue
                
    #         else:
    #             # If the loop finishes without breaking, the line search failed.
    #             print(f"Iter {ls_iter}: Line search failed.")

    #         if abs(old_cost - cost) < cost_threshold:
    #             print(f"Converged with cost improvement {old_cost - cost:.6f} < {cost_threshold}")
    #             break
    #         else:
    #             old_cost = cost
    #             continue
            
    #     return xvec, uvec
        
        
        
