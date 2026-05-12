# Software Architecture Design: Differentiable Tube-based Model Predictive Control (DT-MPC)

This document outlines the software architecture, repository structure, and core module interfaces for the implementation of the Differentiable Robust Model Predictive Control algorithm. 

The architecture is designed to implement the algorithms as presented in "Differentiable Robust Model Predictive Control", utilizing a flattened structure where the discrete barrier states (DBaS) embed the safety constraints directly into the augmented system dynamics.

## Repository Structure
dt_mpc_project/
├── dynamics/
│   ├── __init__.py
│   ├── base_system.py        # Base physical models (e.g., Dubins Car)
│   ├── dubins_car.py         # Base physical models (e.g., Dubins Car)
│   └── safety_embedded.py    # DBaS wrapper for augmented dynamics
├── solvers/
│   ├── __init__.py
│   └── optimal_control.py    # DDP solver for Problem 5 and Problem 6
├── learning/
│   ├── __init__.py
│   ├── doc_engine.py         # Algorithms 1, 3, and 4 (Implicit Differentiation)
│   └── dt_mpc_loop.py        # Algorithm 2 (Main Training Loop)
├── scripts/
│   └── run_training.py       # Main entry point to start the simulation
├── tests/                    # Unit tests
└── README.md


## File Descriptions and Core Interfaces
1. dynamics/base_system.py
Contains the physical equations of motion for the system without any safety or barrier considerations.

```
import numpy as np

class DubinsCar:
    def __init__(self, dt: float):
        self.dt = dt
        self.state_dim = 3   # [x, y, theta]
        self.control_dim = 2 # [v, omega]

    def step(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        \"\"\"
        Advances the true physical state of the Dubins car by one timestep.
        \"\"\"
        pass
```

2. dynamics/safety_embedded.py
Wraps the base physical system and augments the state with Discrete Barrier States (DBaS). This provides the unified $f(x, u, \theta)$ interface for the unconstrained solvers.

```
import numpy as np

class SafetyEmbeddedDynamics:
    def __init__(self, base_plant, dbas_params: dict):
        self.plant = base_plant
        self.theta = dbas_params  # Contains tunable parameters \gamma, \alpha
        
    def step(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        \"\"\"
        Computes x_{k+1} = f(x_k, u_k, \theta) including the augmented barrier states.
        \"\"\"
        pass

    def get_derivatives(self, state: np.ndarray, control: np.ndarray):
        \"\"\"
        Returns Jacobians and Hessians of f, \ell, \phi with respect to x, u, and \theta.
        Required for the DDP solver and the DOC backward/forward passes.
        \"\"\"
        pass
```

3. solvers/optimal_control.py
Contains the unconstrained optimal control solver (Differential Dynamic Programming - DDP) used to solve both the nominal and ancillary problems over the safety-embedded dynamics.

```
import numpy as np

class DDPSolver:
    def __init__(self, dynamics, horizon: int):
        self.dynamics = dynamics
        self.horizon = horizon

    def solve(self, initial_state: np.ndarray, is_nominal: bool) -> dict:
        \"\"\"
        Solves Problem 5 (if is_nominal=True) or Problem 6 (if False).
        Returns a dictionary containing:
        - 'states': Optimal state trajectory
        - 'controls': Optimal control trajectory
        - 'derivatives': Local derivatives along the optimal trajectory (needed for DOC)
        \"\"\"
        pass
```

4. learning/doc_engine.py
Implements the Differentiable Optimal Control algorithms (1, 3, and 4) to compute the gradient of the upper-level loss with respect to the parameters $\theta$.Pythonimport numpy as np

class DifferentiableOptimalControl:
    def backward_pass(self, trajectory_data: dict, dynamics_derivs: dict) -> dict:
        \"\"\"
        Implements Algorithm 3: DOC Backward Pass.
        Returns \widetilde{V}_x, V_{xx}, \widetilde{k}, and K.
        \"\"\"
        pass

    def forward_pass(self, backward_outputs: dict, param_derivs: dict) -> np.ndarray:
        \"\"\"
        Implements Algorithm 4: DOC Forward Pass.
        Returns the gradient of the loss \nabla_\theta L.
        \"\"\"
        pass

    def compute_gradient(self, upper_loss_derivs: dict, trajectory_data: dict, dynamics) -> np.ndarray:
        \"\"\"
        Implements Algorithm 1: Differentiable Optimal Control (DOC).
        Coordinates the backward and forward passes to output final \nabla_\theta L.
        \"\"\"
        pass

5. learning/dt_mpc_loop.py
The main outer loop managing the interaction between the dynamics, solvers, and differentiation engine to update parameters.

```
import numpy as np

class DTMPCTrainer:
    def __init__(self, plant, solver, doc_engine, learning_rate: float, horizon_H: int):
        self.plant = plant
        self.solver = solver
        self.doc_engine = doc_engine
        self.learning_rate = learning_rate
        self.horizon_H = horizon_H

    def compute_upper_level_loss(self, tau_star, tau_bar):
        \"\"\"
        Computes L(\tau^*(\theta), \bar{\tau}(\bar{\theta})) and its derivatives.
        \"\"\"
        pass

    def train_step(self, initial_state: np.ndarray):
        \"\"\"
        Implements a single epoch of Algorithm 2.
        1. Solves Problem 5 for \bar{\tau}
        2. Solves Problem 6 for \tau^*
        3. Computes upper-level loss and derivatives
        4. Calls DOC engine (Algorithm 1) to get \nabla_\theta L and \nabla_{\bar{\theta}} L
        5. Applies gradient descent step to update \theta and \bar{\theta}
        6. Steps true and nominal dynamics forward
        \"\"\"
        pass
```

## Mermaid Diagram

graph TD
    %% Define the Data Structures
    subgraph System Dynamics
        A[<b>Dubins Car Base Model</b><br>Standard kinematics] --> B(<b>DBaS Wrapper</b><br>Augments state with barrier variables)
        B --> C{Augmented Dynamics<br><i>f(x, u, &theta;)</i>}
    end

    %% Define the Solvers
    subgraph Optimal Control Solvers
        C --> D[<b>Nominal MPC Solver</b><br>Solves Problem 5 for <i>&tau;(&theta;)</i>]
        C --> E[<b>Ancillary MPC Solver</b><br>Solves Problem 6 for <i>&tau;*(&theta;)</i>]
    end

    %% Define the DOC Engine
    subgraph DOC Engine
        D --> F[<b>DOC Backward Pass</b><br>Algorithm 3]
        E --> F
        F --> G[<b>DOC Forward Pass</b><br>Algorithm 4]
        G --> H(<b>DOC Output</b><br>Algorithm 1: &nabla;<sub>&theta;</sub>L)
    end

    %% Define the Main Loop
    subgraph DT-MPC Learning Loop
        H --> I[<b>Algorithm 2 Main Loop</b><br>Applies Gradient Descent to &theta;]
        I -. Updates &theta; .-> B
        I -. Steps Dynamics .-> A
    end
    
    style C fill:#f9f,stroke:#333,stroke-width:2px
    style H fill:#bbf,stroke:#333,stroke-width:2px
