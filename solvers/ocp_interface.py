from dataclasses import dataclass
from dynamics.base_system import DynamicalSystem
from solvers.costs import BaseCost 

@dataclass
class OCP:
    """The universal interface between MPC controllers and the DDP solver."""
    system: DynamicalSystem
    stage_cost: BaseCost
    terminal_cost: BaseCost
    horizon: int
    dt: float
