import numpy as np
from solvers.optimal_control import DDPSolver
from solvers.ocp_interface import OCPFormulation

class TubeMPC:
    def __init__(self, 
                 nominal_problem: OCPFormulation, 
                 ancillary_problem: OCPFormulation, 
                 solver_engine: DDPSolver):
        
        self.nominal_problem = nominal_problem
        self.ancillary_problem = ancillary_problem
        self.solver = solver_engine

    def nominal_mpc(self, current_state: np.ndarray) -> np.ndarray:
        """
        Solves the nominal mpc for the current state.
        """
        self.solver.load_problem(self.nominal_problem)
        nominal_trajectory = self.solver.solve(current_state)
        pass
    
    def ancillary_mpc(self, current_state: np.ndarray) -> np.ndarray:
        """
        Solves the ancillary mpc for the current state.
        """
        self.solver.load_problem(self.ancillary_problem)
        safe_trajectory = self.solver.solve(current_state)
        pass

    def tube_mpc(self, current_state: np.ndarray) -> np.ndarray:
        """
        Executes the Tube-MPC logic for a single timestep.
        """
        NotImplementedError("tube_mpc unimplemented")
        
        # 1. Solve the Nominal Problem (The perfect world)
        # The solver engine temporarily loads the nominal rules
        self.nominal_mpc( ... )
        
        # 2. Update the Ancillary Problem to track the Nominal Trajectory
        # (Update the stage cost here so the ancillary 
        # controller knows where the center of the tube is)
        
        # something like
        self.ancillary_problem.stage_cost.set_reference_trajectory(nominal_trajectory['states'])

        # 3. Solve the Ancillary Problem (The real, safe world)
        self.ancillary_mpc( ... )

        # 4. Return the first action of the safe trajectory to apply to the plant
        return safe_trajectory['controls'][0]