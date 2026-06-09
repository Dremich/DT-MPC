"""Main DT-MPC training loop (Algorithm 2).

Implements the online adaptation of the tube-based MPC barrier parameters
theta = [alpha, gamma] via gradient descent, as described in Oshin et al.,
"Differentiable Robust Model Predictive Control" (arXiv:2308.08426),
Algorithm 2. Each step solves the nominal MPC (Problem 5) and the ancillary MPC
(Problem 6), evaluates the upper-level loss (Eq. 9), uses the DOC engine
(Algorithm 1) to obtain nabla_theta L, and applies a projected gradient-descent
update to the shared safety-embedded system's [alpha, gamma].
"""

import numpy as np


class DTMPCTrainer:
    def __init__(
        self,
        plant,
        nominal_problem,
        ancillary_problem,
        solver,
        doc_engine,
        learning_rate: float = 1e-2,
        horizon_H: int = 75,
        alpha_min: float = 1e-3,
        gamma_bounds: tuple = (-1.0, 1.0),
        track_dims: tuple = (0, 1),
    ):
        """
        Args:
            plant: the shared SafetyEmbeddedDynamics whose [alpha, gamma] adapt.
            nominal_problem / ancillary_problem: the two tube-MPC OCPs.
            solver: DDP solver exposing ``run_ddp(ocp, x0, u0=None)``.
            doc_engine: DifferentiableOptimalControl instance (Algorithm 1).
            learning_rate: gradient-descent step size eta.
            horizon_H: task horizon (number of closed-loop steps).
            alpha_min: lower bound for alpha (relaxed barrier needs alpha > 0).
            gamma_bounds: (lo, hi) box for gamma (paper uses [-1, 1]).
            track_dims: base-state dimensions tracked in the loss (Eq. 9). The
                paper's Dubins/Robotarium loss tracks position only (x, y) plus
                the barrier magnitude, excluding orientation.
        """
        self.plant = plant
        self.nominal_problem = nominal_problem
        self.ancillary_problem = ancillary_problem
        self.solver = solver
        self.doc_engine = doc_engine
        self.learning_rate = learning_rate
        self.horizon_H = horizon_H
        self.alpha_min = alpha_min
        self.gamma_bounds = gamma_bounds
        self.track_dims = tuple(track_dims)

        # Receding-horizon state, mirroring solvers/tube_mpc.py.
        self.current_nominal_state = None
        self._prev_nominal_control = None

    def compute_upper_level_loss(self, tau_star, tau_bar):
        """Upper-level loss (Eq. 9) and its gradient w.r.t. the ancillary states.

        L = sum_k ||x*_track - x_bar_track||^2 + ||b*||^2, where ``track`` are the
        ``track_dims`` (position) and ``b*`` is the barrier state (last dim).

        Args:
            tau_star: ancillary state trajectory x*, shape (N+1, nx).
            tau_bar: nominal state trajectory x_bar, shape (N+1, nx).

        Returns:
            (loss, grad_x) where grad_x has shape (N+1, nx) and grad_u is implicitly
            zero (the loss does not depend on the controls).
        """
        x_star = np.asarray(tau_star)
        x_bar = np.asarray(tau_bar)

        pos = list(self.track_dims)
        diff = x_star[:, pos] - x_bar[:, pos]
        b = x_star[:, -1]

        loss = float(np.sum(diff ** 2) + np.sum(b ** 2))

        grad_x = np.zeros_like(x_star)
        grad_x[:, pos] = 2.0 * diff
        grad_x[:, -1] = 2.0 * b
        return loss, grad_x

    def train_step(self, current_state: np.ndarray):
        """A single epoch of Algorithm 2.

        1. Solve Problem 5 (nominal) from the current nominal state -> tau_bar.
        2. Update the ancillary tracking reference to tau_bar.
        3. Solve Problem 6 (ancillary) from the true state -> tau_star.
        4. Compute the upper-level loss and its state gradient (Eq. 9).
        5. Call the DOC engine (Algorithm 1) for nabla_theta L.
        6. Projected gradient-descent update of the shared [alpha, gamma].
        7. Advance the nominal state (the true state is advanced by the caller
           via ``plant.step_sim``).

        Returns:
            (u_applied, diagnostics) where ``u_applied`` is the ancillary control
            to apply and ``diagnostics`` holds loss, grad_theta, alpha, gamma.
        """
        # Initialize the nominal state to the true state at t=0.
        if self.current_nominal_state is None:
            self.current_nominal_state = np.copy(current_state)

        # 1. Nominal MPC (Problem 5).
        nom_x, nom_u, _ = self.solver.run_ddp(
            self.nominal_problem, self.current_nominal_state, self._prev_nominal_control
        )
        # Warm-start the next nominal solve.
        self._prev_nominal_control = np.roll(nom_u, shift=-1, axis=0)
        self._prev_nominal_control[-1] = nom_u[-1]

        # 2. Update ancillary tracking reference.
        self.ancillary_problem.stage_cost.update_reference(nom_x, nom_u)
        self.ancillary_problem.terminal_cost.update_reference(nom_x[-1])

        # 3. Ancillary MPC (Problem 6).
        anc_x, anc_u, _ = self.solver.run_ddp(self.ancillary_problem, current_state)

        # 4. Upper-level loss and gradient (Eq. 9).
        loss, grad_x = self.compute_upper_level_loss(anc_x, nom_x)

        # 5. Hypergradient nabla_theta L = [dL/d_alpha, dL/d_gamma] via DOC.
        grad_theta = self.doc_engine.compute_gradient(
            self.ancillary_problem, anc_x, anc_u, grad_x
        )

        # 6. Projected gradient-descent step on the shared barrier parameters.
        new_alpha = self.plant.alpha - self.learning_rate * float(grad_theta[0])
        new_gamma = self.plant.gamma - self.learning_rate * float(grad_theta[1])
        self.plant.alpha = float(np.clip(new_alpha, self.alpha_min, np.inf))
        self.plant.gamma = float(np.clip(new_gamma, self.gamma_bounds[0], self.gamma_bounds[1]))

        # 7. Advance the nominal state (true state advanced by the caller).
        u_applied = np.asarray(anc_u[0])
        self.current_nominal_state = np.asarray(nom_x[1])

        diagnostics = {
            "loss": loss,
            "grad_theta": np.asarray(grad_theta),
            "alpha": self.plant.alpha,
            "gamma": self.plant.gamma,
        }
        return u_applied, diagnostics
