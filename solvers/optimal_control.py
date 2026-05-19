"""Optimal control solvers for DT-MPC."""

import numpy as np
from .ocp_interface import OCP
from typing import Tuple, Optional

class DDPSolver:

    @staticmethod
    def run_ddp(ocp: OCP, initial_state: np.ndarray, initial_control : Optional[np.ndarray] = None, 
                max_iters: int = 20, cost_threshold: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
        if initial_control is None:
            uvec = np.zeros((ocp.horizon, ocp.system.control_dim))
        else:
            uvec = np.copy(initial_control)
            
        xvec = np.zeros((ocp.horizon + 1, ocp.system.state_dim))
        xvec[0] = initial_state
        
        # Initial Forward Pass
        for k in range(ocp.horizon):
            xvec[k+1] = ocp.system.step(xvec[k], uvec[k], ocp.dt)

        old_cost = ocp.terminal_cost.evaluate(xvec[-1])
        old_cost += sum(ocp.stage_cost.evaluate(xvec[k], uvec[k]) for k in range(ocp.horizon))            
        
        # DDP Core Loop
        for _ in range(max_iters):
            
            Kx = np.zeros((ocp.horizon, ocp.system.control_dim, ocp.system.state_dim)) # Feedback gain
            Kf = np.zeros((ocp.horizon, ocp.system.control_dim))                       # Feedforward gain
            
            # Terminal costate
            Vx, _, Vxx, _, _ = ocp.terminal_cost.get_derivatives(xvec[-1])
            
            # Backwards Pass
            for k in reversed(range(ocp.horizon)):
                xk, uk = xvec[k], uvec[k]
                
                # Local linearization of the problem
                A, B = ocp.system.discrete_jacobians(xk, uk, ocp.dt)
                
                cx, cu, cxx, cuu, cxu = ocp.stage_cost.get_derivatives(xk, uk)
                
                # Build Q functions
                Qx = cx + A.T @ Vx
                Qu = cu + B.T @ Vx
                Qxx = cxx + A.T @ Vxx @ A
                Quu = cuu + B.T @ Vxx @ B
                Qux = cxu.T + B.T @ Vxx @ A 
                
                # Regularize and invert Quu
                Quu += np.eye(ocp.system.control_dim) * 1e-4 # prevents singular matrices
                Quu_inv = np.linalg.inv(Quu)
                
                # Find the gains
                Kf[k] = -Quu_inv @ Qu
                Kx[k] = -Quu_inv @ Qux

                # Update the value function derivatives and proceed to next step
                Vx  = Qx + Kx[k].T @ Quu @ Kf[k] + Kx[k].T @ Qu + Qux.T @ Kf[k]
                Vxx = Qxx + Kx[k].T @ Quu @ Kx[k] + Kx[k].T @ Qux + Qux.T @ Kx[k]      
                
            # Forward Pass w/ Line Search
            alpha = 1.0
            num_search = 10
            
            cost = 0.0
            for ls_iter in range(num_search):
                
                x_new = np.zeros((ocp.horizon + 1, ocp.system.state_dim))
                u_new = np.zeros((ocp.horizon, ocp.system.control_dim))
                x_new[0] = initial_state
                cost = 0.0
                    
                for k in range(ocp.horizon):
                    # Apply the gains: u_new = u_old + alpha*Kf + Kx*(x_new - x_old)
                    dx = x_new[k] - xvec[k]
                    u_new[k] = uvec[k] + alpha * Kf[k] + Kx[k] @ dx
                    
                    # Simulate physics
                    x_new[k+1] = ocp.system.step(x_new[k], u_new[k], ocp.dt)
                    cost += ocp.stage_cost.evaluate(x_new[k], u_new[k])
                
                cost += ocp.terminal_cost.evaluate(x_new[-1])
                
                if cost < old_cost: # Success!
                    xvec = x_new
                    uvec = u_new
                    print(f"Iter {ls_iter}: Accepted alpha = {alpha} with cost {cost:.4f}")
                    break
                else:
                    alpha /= 2.0
                    continue
                
            else:
                # If the loop finishes without breaking, the line search failed.
                print(f"Iter {ls_iter}: Line search failed.")

            if abs(old_cost - cost) < cost_threshold:
                print(f"Converged with cost improvement {old_cost - cost:.6f} < {cost_threshold}")
                break
            else:
                old_cost = cost
                continue
            
        return xvec, uvec
        
        
        
