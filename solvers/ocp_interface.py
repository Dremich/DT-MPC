from dataclasses import dataclass
from dynamics.base_system import DynamicalSystem
from solvers.costs import BaseCost


@dataclass
class OCPFormulation:
    """Universal interface between MPC controllers and the DDP solver."""
    dynamics: DynamicalSystem
    stage_cost: BaseCost
    terminal_cost: BaseCost
    horizon: int
    dt: float


# Backwards-compatible alias
OCP = OCPFormulation
