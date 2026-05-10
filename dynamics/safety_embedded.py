"""Safety-embedded dynamics wrappers for DT-MPC."""

import numpy as np


class SafetyEmbeddedDynamics:
    def __init__(self, base_plant, dbas_params: dict):
        self.plant = base_plant
        self.theta = dbas_params

    def step(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """
        Computes x_{k+1} = f(x_k, u_k, theta) including the augmented barrier states.
        """
        raise NotImplementedError("SafetyEmbeddedDynamics.step is not implemented yet.")

    def get_derivatives(self, state: np.ndarray, control: np.ndarray):
        """
        Returns Jacobians and Hessians of f, ell, phi with respect to x, u, and theta.
        Required for the DDP solver and the DOC backward/forward passes.
        """
        raise NotImplementedError(
            "SafetyEmbeddedDynamics.get_derivatives is not implemented yet."
        )
