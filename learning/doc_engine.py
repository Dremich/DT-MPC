"""Differentiable optimal control engine for DT-MPC."""

import numpy as np


class DifferentiableOptimalControl:
    def backward_pass(self, trajectory_data: dict, dynamics_derivs: dict) -> dict:
        """
        Implements Algorithm 3: DOC Backward Pass.
        Returns V_x_tilde, V_xx, k_tilde, and K.
        """
        raise NotImplementedError(
            "DifferentiableOptimalControl.backward_pass is not implemented yet."
        )

    def forward_pass(self, backward_outputs: dict, param_derivs: dict) -> np.ndarray:
        """
        Implements Algorithm 4: DOC Forward Pass.
        Returns the gradient of the loss nabla_theta L.
        """
        raise NotImplementedError(
            "DifferentiableOptimalControl.forward_pass is not implemented yet."
        )

    def compute_gradient(
        self, upper_loss_derivs: dict, trajectory_data: dict, dynamics
    ) -> np.ndarray:
        """
        Implements Algorithm 1: Differentiable Optimal Control (DOC).
        Coordinates the backward and forward passes to output final nabla_theta L.
        """
        raise NotImplementedError(
            "DifferentiableOptimalControl.compute_gradient is not implemented yet."
        )
