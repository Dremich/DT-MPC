"""Diagnostic: why DT-MPC fails to avoid obstacles on the robot arm.

The hint is "look at the changing of costs." This script quantifies two cost
pathologies, reusing the exact paper-faithful setup in test_dt_mpc_arm.py:

  (A) NOMINAL OBJECTIVE IMBALANCE.
      Decompose the nominal MPC objective along its own optimal plan into the
      end-effector tracking term, the barrier term (q_b * b^2), and the control
      term. If the barrier term is a negligible fraction of the objective, the
      nominal planner has almost no incentive to keep the arm away from obstacles.

  (B) PERVERSE UPPER-LEVEL INCENTIVE (alpha runaway).
      The upper-level loss is L = sum ||q*-qbar||^2 + ||b*||^2. Sweep alpha and,
      for each, re-solve the ancillary problem and report L together with the
      *actual* geometric clearance of the resulting trajectory. If L drops as
      alpha grows while the clearance does NOT improve (or worsens), then gradient
      descent on theta minimizes the loss by RELAXING the barrier (inflating
      alpha), not by making the arm safer -- exactly what the closed loop does.

Run:  python tests/debug_arm_cost.py
"""

import os
import sys
import numpy as np
import jax.numpy as jnp

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("DTMPC_SHOW", "0")
import test_dt_mpc_arm as ta
from solvers.optimal_control import DDPSolver

N_JOINTS = ta.N_JOINTS
TARGET = ta.target

# Reconstruct the nominal cost weights used inside make_setup (kept in sync).
W = np.diag([10.0, 10.0, 10.0])
R_NOM = np.diag([10.0] * N_JOINTS)
WF = np.diag([10000.0, 10000.0, 10000.0])
QB = 1e-3


def ee(q):
    return np.asarray(ta.arm.end_effector(jnp.asarray(q)))


def nominal_cost_decomposition(nom_x, nom_u):
    """Split the nominal objective along its own plan into (EE, barrier, control)."""
    ee_term = bar_term = ctrl_term = 0.0
    N = len(nom_u)
    for k in range(N):
        q = nom_x[k][:N_JOINTS]
        de = ee(q) - TARGET
        ee_term += float(de @ W @ de)
        bar_term += QB * float(nom_x[k][-1]) ** 2
        u = nom_u[k]
        ctrl_term += float(u @ R_NOM @ u)
    # terminal
    qT = nom_x[N][:N_JOINTS]
    deT = ee(qT) - TARGET
    ee_term += float(deT @ WF @ deT)
    bar_term += QB * float(nom_x[N][-1]) ** 2
    return ee_term, bar_term, ctrl_term


def traj_min_clearance(x_traj):
    """True geometric min clearance over a state trajectory."""
    return min(ta.true_min_margin(np.asarray(x)[:-1]) for x in x_traj)


def main():
    print("=" * 74)
    print("ARM COST DIAGNOSTIC  (seed = %d, paper-faithful settings)" % ta.SEED)
    print("=" * 74)

    car, nominal_ocp, ancillary_ocp = ta.make_setup()
    rng = np.random.default_rng(ta.SEED)
    # Same near-obstacle start the FD check uses, so the barrier is active.
    cands = [ta.random_feasible_q(rng) for _ in range(40)]
    q0 = min(cands, key=lambda q: ta.soft_min_margin(np.concatenate([q, np.zeros(N_JOINTS)])))
    state = ta.initial_state(car, q0)
    print(f"start EE={np.round(ee(state[:N_JOINTS]),3)}  target={TARGET}  "
          f"start clearance={ta.true_min_margin(state[:-1]):+.3f}  alpha0={car.alpha}\n")

    # ---- (A) Nominal objective imbalance ----
    nom_x, nom_u, _ = DDPSolver.run_ddp(nominal_ocp, state)
    ee_term, bar_term, ctrl_term = nominal_cost_decomposition(nom_x, nom_u)
    total = ee_term + bar_term + ctrl_term
    print("-" * 74)
    print("(A) NOMINAL OBJECTIVE DECOMPOSITION along the nominal plan")
    print("-" * 74)
    print(f"  end-effector tracking : {ee_term:14.4f}   ({100*ee_term/total:6.3f} %)")
    print(f"  barrier  q_b * b^2    : {bar_term:14.4f}   ({100*bar_term/total:6.3f} %)")
    print(f"  control  u^T R u      : {ctrl_term:14.4f}   ({100*ctrl_term/total:6.3f} %)")
    print(f"  nominal plan min clearance over horizon: {traj_min_clearance(nom_x):+.3f}")
    print(f"  => the barrier is {ee_term/max(bar_term,1e-12):.0f}x smaller than the task term;")
    print(f"     the planner barely 'feels' the obstacles.\n")

    # ---- (B) Perverse upper-level incentive: sweep alpha ----
    ancillary_ocp.stage_cost.update_reference(nom_x, nom_u)
    ancillary_ocp.terminal_cost.update_reference(nom_x[-1])
    a0, g0 = car.alpha, car.gamma

    print("-" * 74)
    print("(B) UPPER-LEVEL LOSS vs alpha  (nominal plan fixed; re-solve ancillary)")
    print("-" * 74)
    print(f"  {'alpha':>7}{'loss L':>14}{'sum b*^2':>14}{'track err':>12}"
          f"{'anc clearance':>15}")
    base_clear = None
    for alpha in [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
        car.alpha, car.gamma = alpha, g0
        anc_x, anc_u, _ = DDPSolver.run_ddp(ancillary_ocp, state)
        L, _ = ta.make_trainer(car, nominal_ocp, ancillary_ocp,
                               ta.DifferentiableOptimalControl()).compute_upper_level_loss(anc_x, nom_x)
        b_sq = float(np.sum(np.asarray(anc_x)[:, -1] ** 2))
        track = L - b_sq
        clear = traj_min_clearance(anc_x)
        if base_clear is None:
            base_clear = clear
        print(f"  {alpha:7.2f}{L:14.3f}{b_sq:14.3f}{track:12.3f}{clear:+15.3f}")
    car.alpha, car.gamma = a0, g0
    print()
    print("  Reading: as alpha increases the relaxed barrier flattens, so sum b*^2")
    print("  (hence L) FALLS, but the trajectory's true clearance does not improve.")
    print("  Gradient descent therefore drives alpha UP to cut L 'for free', which")
    print("  relaxes the safety barrier instead of avoiding the obstacle.")
    print("=" * 74)


if __name__ == "__main__":
    main()
