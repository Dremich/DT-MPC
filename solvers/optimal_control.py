"""Optimal control solvers for DT-MPC."""

import numpy as np
from dynamics.base_system import DynamicalSystem
# from solvers.costs import QuadraticCost # Assuming you make this later

class DDPSolver:
    def __init__(self, control_problem: OCP):
        self.ocp = control_problem
        self.step = control_problem.step
        self.horizon = control_problem.horizon
        self.dt = control_problem.dt
        # self.cost = control_problem.QuadraticCost(...) 

    def solve(self, initial_state: np.ndarray, is_nominal: bool) -> dict:
        """
        Solves Problem 5 (if is_nominal=True) or Problem 6 (if False).
        Returns a dictionary containing:
        - 'states': Optimal state trajectory
        - 'controls': Optimal control trajectory
        - 'derivatives': Local derivatives along the optimal trajectory (needed for DOC)
        """
        # --- STUB IMPLEMENTATION ---
        # For now, just return a dummy trajectory so the rest of your pipeline doesn't crash

        NotImplementedError("DDPSolver.solve is not implemented yet. This is a WIP!")
        
        print(f"Running DDP Solver. Nominal Mode: {is_nominal}")
        
        # Initialize empty arrays
        states = np.zeros((self.horizon + 1, self.dynamics.state_dim))
        controls = np.zeros((self.horizon, self.dynamics.control_dim))
        
        states[0] = initial_state
        
        # Forward rollout with zero control to generate a dummy trajectory
        for k in range(self.horizon):
            controls[k] = np.zeros(self.dynamics.control_dim) # Placeholder zero control
            states[k+1] = self.dynamics.discrete_step(states[k], controls[k], self.dt)
            
        # DDP backward pass and line search would go here...

        return {
            'states': states,
            'controls': controls,
            'derivatives': {
                # These will be the A, B, Q, R matrices needed by doc_engine.py
                'A_seq': None, 
                'B_seq': None
            }
        }
        
        
    def run_ddp(ocp: OCP, initial_state: np.ndarray, initial_control : np.ndarray = None, max_iters = 10):
        if initial_control is None:
            uvec = np.zeros(ocp.horizon, ocp.control_dim)
        else:
            uvec = np.copy(initial_control)
            
        xvec = np.zeros(ocp.horizon + 1, ocp.state_dim)
        xvec[0] = intial_state
        
        # Initial Forward Pass
        for k in range(ocp.horizon):
            xvec[k+1] = ocp.system.step(xvec[k], uvec[k], ocp.dt)
            
            
        # DDP Core Loop
        for iter in range(max_iters):
            
            Kx = np.zeros((ocp.horizon, ocp.control_dim, ocp.state_dim)) # Feedback gain
            Kf = np.zeros((ocp.horizon, ocp.state_dim))                  # Feedforward gain
            
            # Terminal costate
            Vxx = ocp.cost.Vxx(xvec[-1])
            Vx = ocp.cost.Vx(xvec[-1])
            
            # Backwards Pass
            for k in reversed(range(ocp.horizon)):
                xk, uk = xvec[k], uvec[k]
                
                # Local linearization of the problem
                A, B = ocp.dynamics.discrete_jacobians(xk, uk, ocp.dt)
                
                cx = ocp.cost.cx(xk, uk)
                cu = ocp.cost.cu(xk, uk)
                cxx = ocp.cost.cxx(xk, uk)
                cuu = ocp.cost.cuu(xk, uk)
                cxu = ocp.cost.cxu(xk, uk)
                
                # Build Q functions
                Qx = cx + A.T @ Vx
                Qu = cu + B.T @ Vx
                Qxx = cxx + A.T @ Vxx @ A
                Quu = cuu + B.T @ Vxx @ B
                Qux = cxu.T + B.T @ Vxx @ A 
                
                # Regularize and invert Quu
                Quu += np.eye(ocp.control_dim) * 1e-4 # prevents singular matrices
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
            
              # Calculate cost
            old_cost = ocp.terminal_cost(xvec[-1])
            old_cost += sum(ocp.stage_cost(xvec[k], uvec[k]) for k in range(ocp.horizon))
            
            for ls_iter in range(num_search):
                
                x_new = np.zeros((ocp.horizon + 1, ocp.state_dim))
                u_new = np.zeros((ocp.horizon, ocp.control_dim))
                x_new[0] = initial_state
                cost = 0.0
                    
                for k in range(ocp.horizon):
                    # Apply the gains: u_new = u_old + alpha*Kf + Kx*(x_new - x_old)
                    dx = x_new[k] - xvec[k]
                    u_new[k] = uvec[k] + alpha * Kf[k] + Kx[k] @ dx
                    
                    # Simulate physics
                    x_new[k+1] = ocp.step(x_new[k], u_new[k], ocp.dt)
                    cost += ocp.stage_cost(x_new[k], u_new[k])
                
                cost += ocp.terminal_cost(x_new[-1])
                
                if cost < old_cost: # Success!
                    xvec = x_new
                    uvec = u_new
                    print(f"Iter {iteration}: Accepted alpha = {alpha} with cost {cost:.4f}")
                    break
                else:
                    alpha /= 2.0
                    continue
                
            else:
                # If the loop finishes without breaking, the line search failed.
                print(f"Iter {iteration}: Line search failed.")
            
        return x_seq, u_seq
        
        
        
