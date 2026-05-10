"""Main DT-MPC training loop."""

import numpy as np


class DTMPCTrainer:
    def __init__(
        self, plant, solver, doc_engine, learning_rate: float, horizon_H: int
    ):
        self.plant = plant
        self.solver = solver
        self.doc_engine = doc_engine
        self.learning_rate = learning_rate
        self.horizon_H = horizon_H

    def compute_upper_level_loss(self, tau_star, tau_bar):
        """
        Computes L(tau*(theta), tau_bar(theta_bar)) and its derivatives.
        """
        raise NotImplementedError(
            "DTMPCTrainer.compute_upper_level_loss is not implemented yet."
        )

    def train_step(self, initial_state: np.ndarray):
        """
        Implements a single epoch of Algorithm 2.
        1. Solves Problem 5 for tau_bar
        2. Solves Problem 6 for tau*
        3. Computes upper-level loss and derivatives
        4. Calls DOC engine (Algorithm 1) to get nabla_theta L and nabla_theta_bar L
        5. Applies gradient descent step to update theta and theta_bar
        6. Steps true and nominal dynamics forward
        """
        raise NotImplementedError("DTMPCTrainer.train_step is not implemented yet.")
