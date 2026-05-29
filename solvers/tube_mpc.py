import numpy as np
from typing import Optional
from solvers.optimal_control import DDPSolver
from solvers.costs import BaseCost
from solvers.ocp_interface import OCP
class TubeMPC:
    def __init__(self, 
                 nominal_problem: OCP, 
                 ancillary_problem: OCP, 
                 solver_engine: DDPSolver):
        
        self.nominal_problem = nominal_problem
        self.ancillary_problem = ancillary_problem
        self.solver = solver_engine
        self._prev_nominal_control: Optional[np.ndarray] = None

        # Holds previous control
        # Optional because not required for initial step
        self.previous_control: Optional[np.ndarray] = None
        self.current_nominal_state: Optional[np.ndarray] = None

    def step_tube(self, current_state: np.ndarray) -> np.ndarray:
        """Executes the Tube-MPC logic for a single timestep."""
        
        # Initialize the nominal state to the true state only at t=0
        if self.current_nominal_state is None:
            self.current_nominal_state = np.copy(current_state)

        # Solve the nominal problem
        nominal_state, nominal_control, _ = self.solver.run_ddp(self.nominal_problem, self.current_nominal_state, self.previous_control)

        # Increment previous control
        self.previous_control = np.roll(nominal_control, shift=-1, axis=0)
        self.previous_control[-1] = nominal_control[-1] # maintains array size

        # Update ancillary cost for tracking
        self.ancillary_problem.stage_cost.update_reference(nominal_state, nominal_control)
        self.ancillary_problem.terminal_cost.update_reference(nominal_state[-1])

        # Solve ancillary problem
        ancillary_state, ancillary_control, _ = self.solver.run_ddp(self.ancillary_problem, current_state)

        # Use ideal dynamics to step nominal state forward for next iteration
        self.current_nominal_state = nominal_state[1]

        return ancillary_control[0]
