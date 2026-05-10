"""Base physical system models for DT-MPC."""

import numpy as np


class DubinsCar:
    def __init__(self, dt: float):
        self.dt = dt
        self.state_dim = 3
        self.control_dim = 2

    def step(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """
        Advances the true physical state of the Dubins car by one timestep.
        """
        raise NotImplementedError("DubinsCar.step is not implemented yet.")
