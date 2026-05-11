"""Base physical system models for DT-MPC."""

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np

class DynamicalSystem(ABC):
    """
    Abstract base class defining the standard interface for all physical plants.
    """
    
    @property
    @abstractmethod
    def state_dim(self) -> int:
        """Dimension of the state vector (n)."""
        pass

    @property
    @abstractmethod
    def control_dim(self) -> int:
        """Dimension of the control vector (m)."""
        pass

    @abstractmethod
    def dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Computes the continuous-time derivative dx/dt = f(x, u).
        """
        pass

    def discrete_step(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """
        Integrates the dynamics forward by one time step dt.
        (Euler integration provided as default, can be overridden with RK4).
        """
        return x + self.dynamics(x, u) * dt
    
    @abstractmethod
    def continuous_jacobians(self, x: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the continuous-time Jacobians A_c = df/dx and B_c = df/du.
        """
        pass

    def discrete_jacobians(self, x: np.ndarray, u: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Converts continuous Jacobians to discrete-time Jacobians using Forward Euler integration.
        
        In discrete time, the future state is approximated as:
            x_{k+1} = x_k + f_c(x_k, u_k) * dt
            
        Therefore, taking the derivative with respect to x_k yields the discrete A matrix (A_d):
            A_d = I + A_c * dt  
            (where I is the Identity matrix representing the x_k term)
            
        Taking the derivative with respect to u_k yields the discrete B matrix (B_d):
            B_d = B_c * dt
            
        Returns:
            A_d: The discrete state transition matrix.
            B_d: The discrete control input matrix.
        """

        A_c, B_c = self.continuous_jacobians(x, u)
        
        A_d = np.eye(self.state_dim) + A_c * dt
        B_d = B_c * dt
        
        return A_d, B_d
