"""Differentiable optimal control engine for DT-MPC (Algorithms 1, 3, 4).

Implements the Differentiable Optimal Control (DOC) method of Oshin et al.,
"Differentiable Robust Model Predictive Control" (arXiv:2308.08426). Given a
converged optimal-control solution (x*, u*) and the gradient of an upper-level
loss L w.r.t. the trajectory, it returns the hypergradient nabla_theta L without
unrolling the solver, by exploiting the Riccati structure of the problem.

The backward pass (Algorithm 3) is structurally identical to the DDP backward
pass: it reproduces the feedback gains K and value Hessians V_xx, and adds a
"tilde" recursion (V_x_tilde, k_tilde) driven by the upper-level loss gradient
instead of the lower-level cost gradient. The forward pass (Algorithm 4)
propagates the perturbations delta_z and accumulates nabla_theta L.

Second-order dynamics terms are dropped (Gauss-Newton / Diff-MPC variant,
Corollary 6), matching the DDP solver in solvers/optimal_control.py so the
implicit gradient is consistent with the solution the solver produced.

For DT-MPC the learnable parameters are the barrier parameters
theta = [alpha, gamma], which enter *only* through the dynamics f. Hence the
parameter derivatives of the cost and initial condition vanish
(L_theta_x = L_theta_u = xi_theta = phi_theta_x = 0) and the hypergradient
collapses to

    nabla_theta L = sum_{k=0}^{N-1} f_theta_k^T delta_lambda_{k+1}.
"""

import numpy as np


class DifferentiableOptimalControl:
    """Computes nabla_theta L through the solution of an optimal control problem."""

    REG = 1e-4  # Quu regularization (matches DDPSolver.run_ddp)

    def backward_pass(self, derivs: dict) -> dict:
        """Algorithm 3: DOC Backward Pass.

        Recomputes the DDP gains ``K`` and value Hessians ``Vxx`` along the
        converged trajectory and additionally propagates the value-gradient
        ``Vx_tilde`` and feedforward ``k_tilde`` driven by the upper-level loss
        gradient (``grad_x``, ``grad_u``).

        Returns per-step ``K`` (N), ``k_tilde`` (N), ``Vxx`` (N+1), ``Vx_tilde``
        (N+1).
        """
        A, B = derivs["A"], derivs["B"]
        l_xx, l_uu, l_xu = derivs["l_xx"], derivs["l_uu"], derivs["l_xu"]
        phi_xx = derivs["phi_xx"]
        grad_x, grad_u = derivs["grad_x"], derivs["grad_u"]
        N, nx, nu = derivs["N"], derivs["nx"], derivs["nu"]

        K = np.zeros((N, nu, nx))
        k_tilde = np.zeros((N, nu))
        Vxx = np.zeros((N + 1, nx, nx))
        Vx_tilde = np.zeros((N + 1, nx))

        # Terminal conditions: V_xx(N) = phi_xx (lower level), V_x_tilde(N) = grad_x(N) (upper).
        Vxx[N] = phi_xx
        Vx_tilde[N] = grad_x[N]

        reg = self.REG * np.eye(nu)
        for k in reversed(range(N)):
            Ak, Bk = A[k], B[k]
            Vxx_next = Vxx[k + 1]
            Vx_next = Vx_tilde[k + 1]

            # Lower-level Q-function Hessians (Gauss-Newton: cost + first-order dynamics).
            Qxx = l_xx[k] + Ak.T @ Vxx_next @ Ak
            Quu = l_uu[k] + Bk.T @ Vxx_next @ Bk + reg
            Qux = l_xu[k].T + Bk.T @ Vxx_next @ Ak

            Quu_inv = np.linalg.inv(Quu)
            K[k] = -Quu_inv @ Qux
            Vxx[k] = Qxx + Qux.T @ K[k]  # Q_xu @ K, with Q_xu = Q_ux^T

            # Upper-level "tilde" recursion (driven by the loss gradient).
            Qx_tilde = grad_x[k] + Ak.T @ Vx_next
            Qu_tilde = grad_u[k] + Bk.T @ Vx_next
            k_tilde[k] = -Quu_inv @ Qu_tilde
            Vx_tilde[k] = Qx_tilde + Qux.T @ k_tilde[k]

        return {"K": K, "k_tilde": k_tilde, "Vxx": Vxx, "Vx_tilde": Vx_tilde}

    def forward_pass(self, backward_outputs: dict, param_derivs: dict) -> np.ndarray:
        """Algorithm 4: DOC Forward Pass.

        Propagates ``delta_x`` / ``delta_lambda`` forward in time and accumulates
        the hypergradient. Since theta enters only through the dynamics, only the
        ``f_theta`` term contributes:

            nabla_theta L = sum_k f_theta_k^T delta_lambda_{k+1}.

        Returns the gradient ``nabla_theta L`` of shape ``(n_theta,)``.
        """
        A, B, f_theta = param_derivs["A"], param_derivs["B"], param_derivs["f_theta"]
        N, nx, n_theta = param_derivs["N"], param_derivs["nx"], param_derivs["n_theta"]
        K, k_tilde = backward_outputs["K"], backward_outputs["k_tilde"]
        Vxx, Vx_tilde = backward_outputs["Vxx"], backward_outputs["Vx_tilde"]

        delta_x = np.zeros(nx)            # delta_x0 = 0 (initial state independent of theta)
        grad_theta = np.zeros(n_theta)    # xi_theta = 0 -> no delta_lambda0 contribution
        for k in range(N):
            delta_u = k_tilde[k] + K[k] @ delta_x
            delta_x_next = A[k] @ delta_x + B[k] @ delta_u
            delta_lambda_next = Vx_tilde[k + 1] + Vxx[k + 1] @ delta_x_next
            grad_theta += f_theta[k].T @ delta_lambda_next
            delta_x = delta_x_next
        return grad_theta

    def compute_gradient(self, ocp, x_traj, u_traj, grad_x, grad_u=None) -> np.ndarray:
        """Algorithm 1: Differentiable Optimal Control (DOC).

        Assembles the dynamics and cost derivatives along the converged
        trajectory ``(x_traj, u_traj)`` (reusing the OCP's existing
        ``discrete_jacobians``, ``jacobian_wrt_theta`` and cost
        ``get_derivatives``), then runs the backward and forward passes.

        Args:
            ocp: the optimal-control problem (provides system, costs, horizon, dt).
            x_traj: optimal state trajectory, shape (N+1, nx).
            u_traj: optimal control trajectory, shape (N, nu).
            grad_x: gradient of the upper-level loss w.r.t. each state, (N+1, nx).
            grad_u: gradient w.r.t. each control, (N, nu); defaults to zeros.

        Returns:
            nabla_theta L = [dL/d_alpha, dL/d_gamma], shape (2,).
        """
        derivs = self._assemble_derivatives(ocp, x_traj, u_traj, grad_x, grad_u)
        backward_outputs = self.backward_pass(derivs)
        return self.forward_pass(backward_outputs, derivs)

    @staticmethod
    def _assemble_derivatives(ocp, x_traj, u_traj, grad_x, grad_u) -> dict:
        """Collect per-step dynamics/cost derivatives along the solution."""
        system = ocp.system
        dt = ocp.dt
        N = ocp.horizon
        nx = system.state_dim
        nu = system.control_dim

        x_traj = np.asarray(x_traj)
        u_traj = np.asarray(u_traj)
        grad_x = np.asarray(grad_x)
        grad_u = np.zeros((N, nu)) if grad_u is None else np.asarray(grad_u)

        A = np.zeros((N, nx, nx))
        B = np.zeros((N, nx, nu))
        f_theta = np.zeros((N, nx, 2))
        l_xx = np.zeros((N, nx, nx))
        l_uu = np.zeros((N, nu, nu))
        l_xu = np.zeros((N, nx, nu))

        for k in range(N):
            xk, uk = x_traj[k], u_traj[k]
            Ak, Bk = system.discrete_jacobians(xk, uk, dt)
            A[k] = np.asarray(Ak)
            B[k] = np.asarray(Bk)
            f_theta[k] = np.asarray(system.jacobian_wrt_theta(xk, uk, dt))
            _, _, lxx, luu, lxu = ocp.stage_cost.get_derivatives(xk, uk, k)
            l_xx[k] = np.asarray(lxx)
            l_uu[k] = np.asarray(luu)
            l_xu[k] = np.asarray(lxu)

        _, _, phi_xx, _, _ = ocp.terminal_cost.get_derivatives(x_traj[N])

        return {
            "A": A, "B": B, "f_theta": f_theta,
            "l_xx": l_xx, "l_uu": l_uu, "l_xu": l_xu,
            "phi_xx": np.asarray(phi_xx),
            "grad_x": grad_x, "grad_u": grad_u,
            "N": N, "nx": nx, "nu": nu, "n_theta": 2,
        }
