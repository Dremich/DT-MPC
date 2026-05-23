import numpy as np
from typing import Optional, Dict

from solvers.optimal_control import DDPSolver
from solvers.ocp_interface import OCPFormulation


class TubeMPC:
    """
    Tube-based MPC controller.

    Two OCPs are maintained:
      nominal  – plans a trajectory in the disturbance-free world
      ancillary – tracks the nominal trajectory, compensating for disturbances

    Call tube_mpc(current_state) each control step to get the next action.
    """

    def __init__(
        self,
        nominal_problem: OCPFormulation,
        ancillary_problem: OCPFormulation,
        solver_engine: DDPSolver,
    ) -> None:
        self.nominal_problem  = nominal_problem
        self.ancillary_problem = ancillary_problem
        self.solver = solver_engine
        self._prev_nominal_control: Optional[np.ndarray] = None

    # ---──────────────────────────────────────────────────────

    def nominal_mpc(self, current_state: np.ndarray) -> Dict:
        """Solve the nominal (disturbance-free) OCP."""
        self.solver.load_problem(self.nominal_problem)
        return self.solver.solve(current_state, self._prev_nominal_control)

    def ancillary_mpc(self, current_state: np.ndarray) -> Dict:
        """Solve the ancillary (tracking) OCP."""
        self.solver.load_problem(self.ancillary_problem)
        return self.solver.solve(current_state)

    # ---─────────────────────────────────────────────────

    def tube_mpc(self, current_state: np.ndarray) -> np.ndarray:
        """
        Executes one Tube-MPC step.

        Returns the first control of the ancillary trajectory to apply to the plant.
        """
        # 1. Solve the nominal problem (perfect world)
        nominal_result   = self.nominal_mpc(current_state)
        nominal_states   = nominal_result["states"]    # (N+1, nx)
        nominal_controls = nominal_result["controls"]  # (N,   nu)

        # Warm-start next nominal solve with a time-shifted control sequence
        self._prev_nominal_control = np.roll(nominal_controls, shift=-1, axis=0)
        self._prev_nominal_control[-1] = nominal_controls[-1]

        # 2. Update the ancillary stage cost to track the nominal trajectory
        #    set_reference_trajectory stores the full (N+1, nx) state array and
        #    (N, nu) control array; evaluate(x, u, k) will index x_ref[k] / u_ref[k]
        self.ancillary_problem.stage_cost.set_reference_trajectory(
            nominal_states, nominal_controls
        )
        # Point the ancillary terminal cost at the end of the nominal trajectory
        self.ancillary_problem.terminal_cost.update_reference(nominal_states[-1])

        # 3. Solve the ancillary problem (real, safe world)
        safe_result = self.ancillary_mpc(current_state)

        # 4. Return the first ancillary control action
        return safe_result["controls"][0]
