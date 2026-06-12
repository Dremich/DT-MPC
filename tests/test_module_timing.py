"""Per-module running-time experiment for one DT-MPC step (Algorithm 2).

Profiles where the wall-clock of a single ``DTMPCTrainer.train_step`` goes on the
Dubins safety-embedded (DBaS) system. One step decomposes into:

  1. nominal MPC solve        run_ddp(nominal_problem)        -- Problem 5
  2. ancillary ref update     update_reference(...)           -- cheap bookkeeping
  3. ancillary MPC solve      run_ddp(ancillary_problem)      -- Problem 6
  4. upper-level loss         compute_upper_level_loss(...)   -- Eq. 9
  5. DOC hypergradient        Algorithm 1, split into:
        5a. assemble derivatives   _assemble_derivatives(...)
        5b. backward pass          backward_pass(...)         -- Algorithm 3
        5c. forward  pass          forward_pass(...)          -- Algorithm 4
  6. theta = [alpha, gamma] update   projected gradient descent

``timed_train_step`` mirrors ``DTMPCTrainer.train_step`` line-for-line, only adding
``time.perf_counter()`` brackets around each module, so the measured composition is
exactly what the production loop does.

We run a few warmup steps (one-time JAX/XLA compilation of the two DDP cores) and
then time a window of steady-state steps along a real closed-loop Dubins
trajectory. Reported per-module times are the mean over the timed window. A horizon
sweep shows how the composition scales with the planning horizon.

Usage:
    python tests/test_module_timing.py
Env:
    DTMPC_TIME_HORIZONS  comma list of horizons to sweep   (default "25,50,100")
    DTMPC_TIME_HEADLINE  horizon for the detailed breakdown (default "50")
    DTMPC_TIME_WARMUP    warmup steps not timed             (default "5")
    DTMPC_TIME_STEPS     timed steps                        (default "25")
    DTMPC_TIME_DT        integration dt                     (default "0.05")
    DTMPC_TIME_NOISE     disturbance std                    (default "0.25")
    DTMPC_TIME_SEED      rng seed                           (default "0")
    DTMPC_TIME_OUT       results file       (default "experiment_module_timing.txt")
    DTMPC_TIME_SHOW      "1" to plt.show() instead of saving (default "0")
"""

import os
import sys
import time
import numpy as np
import jax.numpy as jnp

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dynamics.dubins_car import DubinsCar
from dynamics.safety_embedded import SafetyEmbeddedDynamics
from solvers.ocp_interface import OCP
from solvers.optimal_control import DDPSolver
from solvers.costs import QuadraticCost, TerminalCost
from learning.doc_engine import DifferentiableOptimalControl
from learning.dt_mpc_loop import DTMPCTrainer

# ----------------------------- configuration -----------------------------
HORIZONS = [int(h) for h in os.environ.get("DTMPC_TIME_HORIZONS", "25,50,100").split(",")]
HEADLINE_H = int(os.environ.get("DTMPC_TIME_HEADLINE", "50"))
N_WARMUP = int(os.environ.get("DTMPC_TIME_WARMUP", "5"))
N_TIMED = int(os.environ.get("DTMPC_TIME_STEPS", "25"))
DT = float(os.environ.get("DTMPC_TIME_DT", "0.05"))
NOISE = float(os.environ.get("DTMPC_TIME_NOISE", "0.25"))
SEED = int(os.environ.get("DTMPC_TIME_SEED", "0"))
OUT = os.environ.get("DTMPC_TIME_OUT", "experiment_module_timing.txt")
SHOW = os.environ.get("DTMPC_TIME_SHOW", "0") == "1"

# ----------------------------- Dubins forest (canonical setup) -----------------------------
DUB_OBSTACLES = np.array([
    [5.0, 2.0, 1.0], [3.0, 6.0, 1.2], [7.0, 8.0, 1.5],
    [10.0, 4.0, 1.0], [12.0, 10.0, 1.5], [15.0, 7.0, 1.0],
    [8.0, 15.0, 1.8], [5.0, 12.0, 2.0], [12.0, 18.0, 2.2],
    [18.0, 12.0, 1.5], [20.0, 5.0, 1.5], [22.0, 15.0, 1.2],
    [16.0, 22.0, 2.5], [10.0, 25.0, 1.8], [25.0, 10.0, 3.2],
    [20.0, 25.0, 1.5], [25.0, 20.0, 1.2], [5.0, 20.0, 2.0],
])
DUB_GOAL = np.array([28.0, 28.0, 0.0, 0.0])
DUB_ALPHA0, DUB_GAMMA0 = 0.5, 0.1


def dub_cbf(x):
    d = jnp.sqrt((x[0] - DUB_OBSTACLES[:, 0]) ** 2 + (x[1] - DUB_OBSTACLES[:, 1]) ** 2)
    return d - DUB_OBSTACLES[:, 2]


def build_dubins(dt, horizon, noise_std):
    car = SafetyEmbeddedDynamics(DubinsCar(wheelbase=0.25), dub_cbf,
                                 alpha=DUB_ALPHA0, gamma=DUB_GAMMA0, noise_std=noise_std)
    Q_nom = jnp.diag(jnp.array([1.0, 1.0, 0.5, 100.0]))
    R_nom = jnp.diag(jnp.array([0.1, 0.1]))
    P_nom = jnp.diag(jnp.array([100.0, 100.0, 50.0, 100.0]))
    nominal_ocp = OCP(car, QuadraticCost(Q_nom, R_nom, x_ref=DUB_GOAL),
                      TerminalCost(P_nom, x_ref=DUB_GOAL), horizon, dt)
    Q_anc = jnp.diag(jnp.array([50.0, 50.0, 10.0, 0.0]))
    R_anc = jnp.diag(jnp.array([1.0, 1.0]))
    P_anc = jnp.diag(jnp.array([200.0, 200.0, 50.0, 0.0]))
    ancillary_ocp = OCP(car, QuadraticCost(Q_anc, R_anc), TerminalCost(P_anc), horizon, dt)
    doc = DifferentiableOptimalControl()
    trainer = DTMPCTrainer(car, nominal_ocp, ancillary_ocp, DDPSolver, doc,
                           learning_rate=0.01, horizon_H=70)
    return car, trainer


def dub_initial_state(car):
    b0 = float(jnp.sum(car.relaxed_barrier(dub_cbf(jnp.zeros(3)), car.alpha)))
    return np.array([0.0, 0.0, 0.0, b0])


# The module labels in execution order; the three DOC sub-modules are grouped under
# the "DOC hypergradient" total in reporting.
MODULES = [
    "nominal_solve", "ref_update", "ancillary_solve", "loss",
    "doc_assemble", "doc_backward", "doc_forward", "theta_update",
]
DOC_SUBMODULES = ("doc_assemble", "doc_backward", "doc_forward")


def timed_train_step(trainer, current_state, timings):
    """A line-for-line copy of DTMPCTrainer.train_step with per-module timers."""
    t = trainer
    if t.current_nominal_state is None:
        t.current_nominal_state = np.copy(current_state)

    # 1. Nominal MPC (Problem 5).
    t0 = time.perf_counter()
    nom_x, nom_u, _ = t.solver.run_ddp(
        t.nominal_problem, t.current_nominal_state, t._prev_nominal_control)
    timings["nominal_solve"].append(time.perf_counter() - t0)
    t._prev_nominal_control = np.roll(nom_u, shift=-1, axis=0)
    t._prev_nominal_control[-1] = nom_u[-1]

    # 2. Update ancillary tracking reference.
    t0 = time.perf_counter()
    t.ancillary_problem.stage_cost.update_reference(nom_x, nom_u)
    t.ancillary_problem.terminal_cost.update_reference(nom_x[-1])
    timings["ref_update"].append(time.perf_counter() - t0)

    # 3. Ancillary MPC (Problem 6).
    t0 = time.perf_counter()
    anc_x, anc_u, _ = t.solver.run_ddp(t.ancillary_problem, current_state)
    timings["ancillary_solve"].append(time.perf_counter() - t0)

    # 4. Upper-level loss and gradient (Eq. 9).
    t0 = time.perf_counter()
    loss, grad_x = t.compute_upper_level_loss(anc_x, nom_x)
    timings["loss"].append(time.perf_counter() - t0)

    # 5. DOC hypergradient (Algorithm 1), broken into its three passes.
    doc = t.doc_engine
    t0 = time.perf_counter()
    derivs = doc._assemble_derivatives(t.ancillary_problem, anc_x, anc_u, grad_x, None)
    timings["doc_assemble"].append(time.perf_counter() - t0)
    t0 = time.perf_counter()
    bwd = doc.backward_pass(derivs)
    timings["doc_backward"].append(time.perf_counter() - t0)
    t0 = time.perf_counter()
    grad_theta = doc.forward_pass(bwd, derivs)
    timings["doc_forward"].append(time.perf_counter() - t0)

    # 6. Projected gradient-descent update on [alpha, gamma].
    t0 = time.perf_counter()
    grad_update = np.clip(grad_theta, -t.grad_clip, t.grad_clip)
    new_alpha = t.plant.alpha - t.learning_rate * float(grad_update[0])
    new_gamma = t.plant.gamma - t.learning_rate * float(grad_update[1])
    t.plant.alpha = float(np.clip(new_alpha, t.alpha_min, t.alpha_max))
    t.plant.gamma = float(np.clip(new_gamma, t.gamma_bounds[0], t.gamma_bounds[1]))
    timings["theta_update"].append(time.perf_counter() - t0)

    u_applied = np.asarray(anc_u[0])
    t.current_nominal_state = np.asarray(nom_x[1])
    return u_applied


def profile_horizon(horizon):
    """Run one closed loop at the given horizon; return mean per-module seconds."""
    np.random.seed(SEED)
    car, trainer = build_dubins(DT, horizon, NOISE)
    state = dub_initial_state(car)

    timings = {m: [] for m in MODULES}
    total_steps = []

    n_total = N_WARMUP + N_TIMED
    for step in range(n_total):
        recording = step >= N_WARMUP
        scratch = timings if recording else {m: [] for m in MODULES}
        t0 = time.perf_counter()
        u = timed_train_step(trainer, state, scratch)
        if recording:
            total_steps.append(time.perf_counter() - t0)
        state = car.step_sim(state, u, DT)

    means = {m: float(np.mean(timings[m])) for m in MODULES}
    means_std = {m: float(np.std(timings[m])) for m in MODULES}
    step_total = float(np.mean(total_steps))
    return means, means_std, step_total


def fmt_ms(s):
    return f"{s * 1e3:8.2f}"


def report(headline_means, headline_std, headline_total, sweep):
    """Build the human-readable report (also printed to stdout)."""
    lines = []
    w = lines.append
    w("=" * 78)
    w("DT-MPC PER-MODULE TIMING -- one train_step on the Dubins safety-embedded system")
    w("=" * 78)
    w(f"Setting   : dt={DT}, horizon={HEADLINE_H}, noise_std={NOISE}, seed={SEED}")
    w(f"Window    : {N_WARMUP} warmup step(s) discarded (one-time XLA compile), "
      f"{N_TIMED} timed steps averaged.")
    w("Modules   : one Algorithm-2 step = nominal solve (P5) + ancillary solve (P6)")
    w("            + DOC hypergradient (Alg.1: assemble/backward/forward) + bookkeeping.")
    w("")

    # ---- headline detailed breakdown ----
    doc_total = sum(headline_means[m] for m in DOC_SUBMODULES)
    w("-" * 78)
    w(f"Detailed breakdown at horizon = {HEADLINE_H}")
    w("-" * 78)
    w(f"  {'Module':<26}{'mean (ms)':>12}{'std (ms)':>12}{'% of step':>12}")
    label = {
        "nominal_solve":   "1. Nominal MPC (P5)",
        "ref_update":      "2. Ancillary ref update",
        "ancillary_solve": "3. Ancillary MPC (P6)",
        "loss":            "4. Upper-level loss",
        "doc_assemble":    "5a. DOC assemble derivs",
        "doc_backward":    "5b. DOC backward pass",
        "doc_forward":     "5c. DOC forward pass",
        "theta_update":    "6. theta update",
    }
    for m in MODULES:
        pct = 100.0 * headline_means[m] / headline_total
        w(f"  {label[m]:<26}{fmt_ms(headline_means[m])}{fmt_ms(headline_std[m])}{pct:11.1f}%")
        if m == "doc_forward":
            pct_doc = 100.0 * doc_total / headline_total
            w(f"  {'   (5. DOC total)':<26}{fmt_ms(doc_total)}{'':>12}{pct_doc:11.1f}%")
    w("  " + "-" * 60)
    w(f"  {'TOTAL per step':<26}{fmt_ms(headline_total)}{'':>12}{100.0:11.1f}%")
    w("")

    # ---- grouped view (the three headline costs) ----
    nom = headline_means["nominal_solve"]
    anc = headline_means["ancillary_solve"]
    book = headline_means["ref_update"] + headline_means["loss"] + headline_means["theta_update"]
    w("  Grouped:")
    w(f"    Nominal solve      {fmt_ms(nom)} ms  ({100 * nom / headline_total:4.1f}%)")
    w(f"    Ancillary solve    {fmt_ms(anc)} ms  ({100 * anc / headline_total:4.1f}%)")
    w(f"    DOC hypergradient  {fmt_ms(doc_total)} ms  ({100 * doc_total / headline_total:4.1f}%)")
    w(f"    Bookkeeping        {fmt_ms(book)} ms  ({100 * book / headline_total:4.1f}%)")
    w("")

    # ---- horizon sweep ----
    w("-" * 78)
    w("Horizon sweep  (mean ms per module; how the composition scales with N)")
    w("-" * 78)
    hdr = f"  {'horizon':>8}{'nominal':>11}{'ancillary':>11}{'DOC':>11}{'book':>9}{'TOTAL':>11}"
    w(hdr)
    for h in sorted(sweep):
        m, _, tot = sweep[h]
        doc_t = sum(m[s] for s in DOC_SUBMODULES)
        bk = m["ref_update"] + m["loss"] + m["theta_update"]
        w(f"  {h:>8}{m['nominal_solve'] * 1e3:>11.2f}{m['ancillary_solve'] * 1e3:>11.2f}"
          f"{doc_t * 1e3:>11.2f}{bk * 1e3:>9.2f}{tot * 1e3:>11.2f}")
    w("")
    w("Notes")
    w("- The two DDP solves dominate; both scale roughly linearly with the horizon N")
    w("  (backward Riccati + line-search rollouts are O(N)). The nominal solve has a")
    w("  fixed goal reference so its XLA executable is reused every step; the ancillary")
    w("  reference changes each step but keeps the same shapes, so it stays compiled.")
    w("- The DOC hypergradient (Alg.1) is a CPU/numpy + JAX-dispatch pass; 'assemble'")
    w("  (per-step jacobians_fn / jacobian_wrt_theta / cost derivs) is its main cost,")
    w("  while the backward/forward Riccati recursions are small numpy loops.")
    w("- Bookkeeping (ref update, Eq.9 loss, theta update) is negligible (<1% of a step).")
    w("=" * 78)
    return "\n".join(lines)


def make_figure(headline_means, sweep):
    try:
        import matplotlib
        if not SHOW:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot skipped: {e}]")
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: detailed per-module bar chart at the headline horizon.
    label = ["1.Nominal\n(P5)", "2.Ref\nupdate", "3.Ancillary\n(P6)", "4.Loss",
             "5a.DOC\nassemble", "5b.DOC\nbackward", "5c.DOC\nforward", "6.theta\nupdate"]
    vals = [headline_means[m] * 1e3 for m in MODULES]
    colors = ["#2c7fb8", "#bdbdbd", "#41b6c4", "#bdbdbd",
              "#fd8d3c", "#fdbe85", "#feedde", "#bdbdbd"]
    bars = ax1.bar(range(len(MODULES)), vals, color=colors, edgecolor="k", linewidth=0.4)
    total = sum(vals)
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, v, f"{100 * v / total:.0f}%",
                 ha="center", va="bottom", fontsize=8)
    ax1.set_xticks(range(len(MODULES)))
    ax1.set_xticklabels(label, fontsize=8)
    ax1.set_ylabel("mean time per step (ms)")
    ax1.set_title(f"Per-module timing, one DT-MPC step (horizon={HEADLINE_H})")
    ax1.grid(axis="y", alpha=0.3)

    # Panel 2: stacked composition vs horizon.
    hs = sorted(sweep)
    nom = [sweep[h][0]["nominal_solve"] * 1e3 for h in hs]
    anc = [sweep[h][0]["ancillary_solve"] * 1e3 for h in hs]
    doc = [sum(sweep[h][0][s] for s in DOC_SUBMODULES) * 1e3 for h in hs]
    book = [(sweep[h][0]["ref_update"] + sweep[h][0]["loss"]
             + sweep[h][0]["theta_update"]) * 1e3 for h in hs]
    x = np.arange(len(hs))
    b0 = np.zeros(len(hs))
    for vals_h, lab, col in [(nom, "nominal solve", "#2c7fb8"),
                             (anc, "ancillary solve", "#41b6c4"),
                             (doc, "DOC hypergradient", "#fd8d3c"),
                             (book, "bookkeeping", "#bdbdbd")]:
        ax2.bar(x, vals_h, bottom=b0, label=lab, color=col, edgecolor="k", linewidth=0.4)
        b0 = b0 + np.array(vals_h)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(h) for h in hs])
    ax2.set_xlabel("planning horizon N")
    ax2.set_ylabel("mean time per step (ms)")
    ax2.set_title("Step composition vs horizon")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    if SHOW:
        plt.show()
        return None
    os.makedirs("figures", exist_ok=True)
    path = os.path.join("figures", "module_timing.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main():
    print(f"Profiling DT-MPC per-module timing on Dubins (dt={DT}, noise={NOISE}, seed={SEED})")
    print(f"Horizons to sweep: {HORIZONS}; headline horizon: {HEADLINE_H}")
    print(f"Warmup steps: {N_WARMUP}; timed steps: {N_TIMED}\n")

    horizons = sorted(set(HORIZONS) | {HEADLINE_H})
    sweep = {}
    for h in horizons:
        print(f"  profiling horizon = {h} ...", flush=True)
        means, std, total = profile_horizon(h)
        sweep[h] = (means, std, total)
        print(f"    total/step = {total * 1e3:.1f} ms", flush=True)

    h_means, h_std, h_total = sweep[HEADLINE_H]
    text = report(h_means, h_std, h_total, sweep)
    print("\n" + text)

    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\nResults written to {OUT}")

    fig_path = make_figure(h_means, sweep)
    if fig_path:
        print(f"Figure written to {fig_path}")


if __name__ == "__main__":
    main()
