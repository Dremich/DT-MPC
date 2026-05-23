"""Optimal control solvers for DT-MPC."""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional, Dict

from solvers.ocp_interface import OCPFormulation

jax.config.update("jax_enable_x64", True)


class DDPSolver:
    """
    Differential Dynamic Programming (DDP) solver.

    Usage:
        solver = DDPSolver()
        solver.load_problem(ocp)
        result = solver.solve(initial_state)
        # result['states']      – (N+1, nx) optimal state trajectory
        # result['controls']    – (N,   nu) optimal control sequence
        # result['derivatives'] – {'A_seq': ..., 'B_seq': ...}
    """

    def __init__(self) -> None:
        self._ocp: Optional[OCPFormulation] = None

    def load_problem(self, ocp: OCPFormulation) -> None:
        self._ocp = ocp

    def solve(
        self,
        initial_state: np.ndarray,
        initial_control: Optional[np.ndarray] = None,
        max_iters: int = 20,
        cost_threshold: float = 1e-3,
    ) -> Dict:
        ocp = self._ocp
        if ocp is None:
            raise RuntimeError("Call load_problem() before solve().")

        nx = ocp.dynamics.state_dim
        nu = ocp.dynamics.control_dim
        N  = ocp.horizon
        dt = ocp.dt

        uvec = (
            np.zeros((N, nu))
            if initial_control is None
            else np.array(initial_control, dtype=np.float64)
        )

        # ---─────────────────────────────────────
        xvec = self._rollout(ocp, initial_state, uvec)
        old_cost = self._total_cost(ocp, xvec, uvec)

        # ---──────────────────────────────────────────────
        for _ in range(max_iters):
            Kx, Kf = self._backward_pass(ocp, xvec, uvec)
            xvec, uvec, cost = self._forward_pass(
                ocp, initial_state, xvec, uvec, Kx, Kf, old_cost
            )

            if abs(old_cost - cost) < cost_threshold:
                break
            old_cost = cost

        return {
            "states":   xvec,
            "controls": uvec,
            "derivatives": self._linearise(ocp, xvec, uvec),
        }

    # ---─────────────────────────────────────────────────

    def _rollout(self, ocp, x0, uvec):
        N  = ocp.horizon
        nx = ocp.dynamics.state_dim
        xvec = np.zeros((N + 1, nx))
        xvec[0] = x0
        for k in range(N):
            xvec[k + 1] = np.array(
                ocp.dynamics.step(jnp.array(xvec[k]), jnp.array(uvec[k]), ocp.dt)
            )
        return xvec

    def _total_cost(self, ocp, xvec, uvec):
        cost = float(ocp.terminal_cost.evaluate(jnp.array(xvec[-1]), None, ocp.horizon))
        for k in range(ocp.horizon):
            cost += float(
                ocp.stage_cost.evaluate(jnp.array(xvec[k]), jnp.array(uvec[k]), k)
            )
        return cost

    def _backward_pass(self, ocp, xvec, uvec):
        N  = ocp.horizon
        nx = ocp.dynamics.state_dim
        nu = ocp.dynamics.control_dim
        dt = ocp.dt

        Kx = np.zeros((N, nu, nx))
        Kf = np.zeros((N, nu))

        Vx, _, Vxx, _, _ = ocp.terminal_cost.get_derivatives(
            jnp.array(xvec[-1]), None, N
        )
        Vx  = np.array(Vx,  dtype=np.float64)
        Vxx = np.array(Vxx, dtype=np.float64)

        for k in reversed(range(N)):
            xk = jnp.array(xvec[k])
            uk = jnp.array(uvec[k])

            # Discrete Jacobians via JAX autodiff on the step function
            A = np.array(
                jax.jacobian(lambda x: ocp.dynamics.step(x, uk, dt))(xk),
                dtype=np.float64,
            )
            B = np.array(
                jax.jacobian(lambda u: ocp.dynamics.step(xk, u, dt))(uk),
                dtype=np.float64,
            )

            cx, cu, cxx, cuu, cxu = ocp.stage_cost.get_derivatives(xk, uk, k)
            cx  = np.array(cx,  dtype=np.float64)
            cu  = np.array(cu,  dtype=np.float64)
            cxx = np.array(cxx, dtype=np.float64)
            cuu = np.array(cuu, dtype=np.float64)
            cxu = np.array(cxu, dtype=np.float64)  # (nx, nu)

            Qx  = cx  + A.T @ Vx
            Qu  = cu  + B.T @ Vx
            Qxx = cxx + A.T @ Vxx @ A
            Quu = cuu + B.T @ Vxx @ B
            Qux = cxu.T + B.T @ Vxx @ A   # (nu, nx)

            Quu_reg = Quu + np.eye(nu) * 1e-4
            Quu_inv = np.linalg.inv(Quu_reg)

            Kf[k] = -Quu_inv @ Qu
            Kx[k] = -Quu_inv @ Qux

            Vx  = Qx  + Kx[k].T @ Quu_reg @ Kf[k] + Kx[k].T @ Qu  + Qux.T @ Kf[k]
            Vxx = Qxx + Kx[k].T @ Quu_reg @ Kx[k] + Kx[k].T @ Qux + Qux.T @ Kx[k]

        return Kx, Kf

    def _forward_pass(self, ocp, x0, xvec, uvec, Kx, Kf, old_cost):
        N  = ocp.horizon
        nx = ocp.dynamics.state_dim
        nu = ocp.dynamics.control_dim

        alpha = 1.0
        cost  = old_cost

        for _ in range(10):
            x_new = np.zeros((N + 1, nx))
            u_new = np.zeros((N, nu))
            x_new[0] = x0

            for k in range(N):
                dx = x_new[k] - xvec[k]
                u_new[k] = uvec[k] + alpha * Kf[k] + Kx[k] @ dx
                x_new[k + 1] = np.array(
                    ocp.dynamics.step(jnp.array(x_new[k]), jnp.array(u_new[k]), ocp.dt)
                )

            new_cost = self._total_cost(ocp, x_new, u_new)
            if new_cost < old_cost:
                return x_new, u_new, new_cost

            alpha /= 2.0

        # Line search failed; return unchanged trajectory
        return xvec, uvec, cost

    def _linearise(self, ocp, xvec, uvec):
        """Compute A, B Jacobian sequences along the final trajectory."""
        N  = ocp.horizon
        dt = ocp.dt
        A_seq, B_seq = [], []
        for k in range(N):
            xk = jnp.array(xvec[k])
            uk = jnp.array(uvec[k])
            A_seq.append(
                np.array(jax.jacobian(lambda x: ocp.dynamics.step(x, uk, dt))(xk))
            )
            B_seq.append(
                np.array(jax.jacobian(lambda u: ocp.dynamics.step(xk, u, dt))(uk))
            )
        return {"A_seq": np.array(A_seq), "B_seq": np.array(B_seq)}
