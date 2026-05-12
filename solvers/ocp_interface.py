from dataclasses import dataclass
from dynamics.base_system import DynamicalSystem
# TODO: Implement a BaseCost class that Cost functions inherit from
from solvers.costs import BaseCost 

@dataclass
class OCP:
    """The universal interface between MPC controllers and the DDP solver."""
    dynamics: DynamicalSystem
    stage_cost: BaseCost
    terminal_cost: BaseCost
    horizon: int
    dt: int