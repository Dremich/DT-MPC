import numpy as np
from typing import Tuple

class QuadraticCost:
    def __init__(self, Q: np.ndarray, R: np.ndarray):
        self.Q = Q
        self.R = R

    def stage_cost(self, x: np.ndarray, u: np.ndarray) -> float:
        """The general l(x_k, u_k) from the paper."""
        return x.T @ self.Q @ x + u.T @ self.R @ u

    def stage_cost_derivatives(self, x: np.ndarray, u: np.ndarray):
        """Returns l_x, l_u, l_xx, l_uu, l_xu for the DDP backward pass."""
        l_x = 2 * self.Q @ x
        l_u = 2 * self.R @ u
        l_xx = 2 * self.Q
        l_uu = 2 * self.R
        l_xu = np.zeros((x.shape[0], u.shape[0]))
        return l_x, l_u, l_xx, l_uu, l_xu
    
    def jacobian_dynamics(self, x: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns f_x, f_u for the DDP backward pass."""
        raise NotImplementedError("QuadraticCost.jacobian_dynamics is not implemented yet.")
    
    def hessian_cost(self, x: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns l_xx, l_uu, l_xu for the DDP backward pass."""
        return NotImplementedError("QuadraticCost.hessian_cost is not implemented yet.")
