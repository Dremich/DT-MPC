"""Decompose the DT-MPC upper-level loss along the arm closed loop.

Answers: why does the upper-level loss rise in the middle of the run?

L = sum_k ||q*_k - qbar_k||^2  +  sum_k ||b*_k||^2
        \________ track ________/    \____ barrier ____/

This mirrors DTMPCTrainer.train_step exactly (warm start, ref update, alpha-anchor
update) but additionally records, per closed-loop step:
  * track   = sum over the horizon of the joint-angle tube error ||q*-qbar||^2
  * barrier = sum over the horizon of the ancillary barrier state b*^2
  * plan_min_margin = the TRUE geometric min clearance over the planned ancillary
    trajectory (how deep into the obstacle field the 1 s lookahead reaches)
  * ee_dist, realized margin
Then it plots the decomposition so we can see what the hump tracks.

Run:  python tests/debug_arm_loss.py   (uses the fixed defaults: qb=10, alpha_reg=20)
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

NJ = ta.N_JOINTS
STEPS = int(os.environ.get("DTMPC_STEPS", "220"))


def instrumented_step(trainer, state):
    """A copy of DTMPCTrainer.train_step that also returns the loss decomposition."""
    t = trainer
    if t.current_nominal_state is None:
        t.current_nominal_state = np.copy(state)

    nom_x, nom_u, _ = t.solver.run_ddp(t.nominal_problem, t.current_nominal_state,
                                       t._prev_nominal_control)
    t._prev_nominal_control = np.roll(nom_u, shift=-1, axis=0)
    t._prev_nominal_control[-1] = nom_u[-1]

    t.ancillary_problem.stage_cost.update_reference(nom_x, nom_u)
    t.ancillary_problem.terminal_cost.update_reference(nom_x[-1])

    anc_x, anc_u, _ = t.solver.run_ddp(t.ancillary_problem, state)
    loss, grad_x = t.compute_upper_level_loss(anc_x, nom_x)

    grad_theta = t.doc_engine.compute_gradient(t.ancillary_problem, anc_x, anc_u, grad_x)
    grad_update = np.clip(grad_theta, -t.grad_clip, t.grad_clip)
    if t.alpha_reg > 0.0:
        grad_update = grad_update.copy()
        grad_update[0] += 2.0 * t.alpha_reg * (t.plant.alpha - t.alpha_anchor)
    t.plant.alpha = float(np.clip(t.plant.alpha - t.learning_rate * float(grad_update[0]),
                                  t.alpha_min, t.alpha_max))
    t.plant.gamma = float(np.clip(t.plant.gamma - t.learning_rate * float(grad_update[1]),
                                  t.gamma_bounds[0], t.gamma_bounds[1]))
    u_applied = np.asarray(anc_u[0])
    t.current_nominal_state = np.asarray(nom_x[1])

    # ---- loss decomposition ----
    ax = np.asarray(anc_x); nx = np.asarray(nom_x)
    pos = list(t.track_dims)
    track = float(np.sum((ax[:, pos] - nx[:, pos]) ** 2))
    barrier = float(np.sum(ax[:, -1] ** 2))
    plan_min_margin = min(ta.true_min_margin(x[:-1]) for x in ax)
    return u_applied, loss, track, barrier, plan_min_margin


def main():
    print("Decomposing the upper-level loss along the arm run "
          f"(qb={ta.NOMINAL_QB}, alpha_reg={ta.ALPHA_REG}, seed={ta.SEED})\n")
    car, nom, anc = ta.make_setup()
    doc = ta.DifferentiableOptimalControl()
    trainer = ta.make_trainer(car, nom, anc, doc)
    rng = np.random.default_rng(ta.SEED)
    state = ta.initial_state(car, ta.random_feasible_q(rng))

    rows = []
    for k in range(STEPS):
        u, loss, track, barrier, plan_mm = instrumented_step(trainer, state)
        u = np.clip(u, -ta.TORQUE_LIMIT, ta.TORQUE_LIMIT)
        state = car.step_sim(state, u, ta.dt)
        ee_dist = float(np.linalg.norm(ta.ee_of(state) - ta.target))
        margin = ta.true_min_margin(state[:-1])
        rows.append((k, loss, track, barrier, plan_mm, ee_dist, margin))
        if k % 10 == 0:
            print(f"step {k:3d} | loss {loss:8.2f} | track {track:8.3f} | "
                  f"barrier {barrier:8.2f} | plan_min_margin {plan_mm:+.3f} | "
                  f"EE {ee_dist:5.2f} | margin {margin:+.3f}")
        if ee_dist < ta.SUCCESS_RADIUS:
            print(f"\nreached target at step {k}")
            break

    R = np.array(rows)
    kpk = int(R[np.argmax(R[:, 1]), 0])
    print(f"\nloss peaks at step {kpk}: total={R[:,1].max():.1f}  "
          f"(track={R[R[:,0]==kpk,2][0]:.2f}, barrier={R[R[:,0]==kpk,3][0]:.1f})")
    print(f"barrier term share of loss: mean {100*np.mean(R[:,3]/R[:,1]):.1f}%  "
          f"(min {100*np.min(R[:,3]/R[:,1]):.1f}%)")
    corr = np.corrcoef(R[:, 3], -R[:, 4])[0, 1]
    print(f"corr(barrier term, -plan_min_margin) = {corr:+.3f}  "
          f"(closer plan to obstacles -> larger barrier term)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax1 = plt.subplots(figsize=(9, 5))
        ax1.plot(R[:, 0], R[:, 1], "k-", lw=2, label="total loss L")
        ax1.plot(R[:, 0], R[:, 3], color="tab:orange", lw=1.8, label=r"barrier $\sum b^{*2}$")
        ax1.plot(R[:, 0], R[:, 2], color="tab:blue", lw=1.8, label=r"track $\sum\|q^*-\bar q\|^2$")
        ax1.set_xlabel("closed-loop step"); ax1.set_ylabel("loss component")
        ax1.legend(loc="upper left"); ax1.grid(alpha=0.3)
        ax2 = ax1.twinx()
        ax2.plot(R[:, 0], R[:, 4], color="tab:green", lw=1.2, ls="--",
                 label="plan min clearance")
        ax2.set_ylabel("plan min clearance (m)", color="tab:green")
        ax2.tick_params(axis="y", labelcolor="tab:green")
        ax2.legend(loc="upper right")
        ax1.set_title("Upper-level loss decomposition along the arm run")
        fig.tight_layout()
        os.makedirs("figures", exist_ok=True)
        fig.savefig("figures/arm_loss_decomposition.png", dpi=130)
        print("Saved figures/arm_loss_decomposition.png")
    except Exception as e:
        print(f"[plot skipped: {e}]")


if __name__ == "__main__":
    main()
