"""Experiment matrix for DT-MPC (Tables 2-5).

Each table row is run N_RUNS=10 times with different random seeds; we report the
success rate, violation rate, and average per-step controller time.

  * Table 2 (General): the Dubins (forest) and robot-arm safety-embedded systems.
  * Table 3 (dt):      Dubins, dt in {0.025, 0.05, 0.1}, horizon=50, noise_std=0.25.
  * Table 4 (horizon): Dubins, horizon in {25, 50, 100}, dt=0.05, noise_std=0.25.
  * Table 5 (noise):   Dubins, noise_std in {0.25, 1.0, 4.0, 10.0}, dt=0.05, horizon=50.

Definitions
  success   : the task target is reached within the step cap.
  violation : the true (hard) safety margin goes < 0 at any step (collision).
  step time : mean wall-clock seconds of one trainer.train_step (the DT-MPC compute).

Results are written incrementally to experiment_results.txt so partial progress is
preserved. Each closed loop terminates early on success or violation.
"""

import os
import sys
import time
import traceback
import numpy as np
import jax.numpy as jnp
import jax

# Make imports work whether run as `python tests/run_experiments.py` or `-m`.
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
from solvers.tube_mpc import TubeMPC
from learning.doc_engine import DifferentiableOptimalControl
from learning.dt_mpc_loop import DTMPCTrainer
import test_dt_mpc_arm as ta

OUT = os.environ.get("DTMPC_OUT", "experiment_results.txt")
N_RUNS = int(os.environ.get("DTMPC_RUNS", "10"))
DUB_MAX_STEPS = int(os.environ.get("DTMPC_DUB_STEPS", "70"))
ARM_MAX_STEPS = int(os.environ.get("DTMPC_ARM_STEPS", "25"))
ARM_HORIZON = int(os.environ.get("DTMPC_ARM_HORIZON", "50"))

def _avg_step_time(step_times):
    """Mean per-step controller time, excluding the first step (one-time JIT warmup)."""
    if len(step_times) > 1:
        return float(np.mean(step_times[1:]))
    return float(step_times[0]) if step_times else float("nan")


# ===================== Dubins (forest) safety-embedded system =====================
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


def dub_true_margin(state):
    return float(np.min(np.asarray(dub_cbf(jnp.asarray(state[:3])))))


def build_dubins_ocps(dt, horizon, noise_std):
    """Build the Dubins car + nominal/ancillary OCPs (shared by the sweeps + ablation)."""
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
    return car, nominal_ocp, ancillary_ocp


def build_dubins(dt, horizon, noise_std):
    car, nominal_ocp, ancillary_ocp = build_dubins_ocps(dt, horizon, noise_std)
    doc = DifferentiableOptimalControl()
    trainer = DTMPCTrainer(car, nominal_ocp, ancillary_ocp, DDPSolver, doc,
                           learning_rate=0.01, horizon_H=DUB_MAX_STEPS)
    return car, trainer


def dub_initial_state(car):
    b0 = float(jnp.sum(car.relaxed_barrier(dub_cbf(jnp.zeros(3)), car.alpha)))
    return np.array([0.0, 0.0, 0.0, b0])


def run_dubins(dt, horizon, noise_std, seed):
    """One closed-loop run. success and violation are INDEPENDENT: the loop is not
    terminated on a transient violation -- a run that violates mid-way but still
    reaches the goal counts as a success. Returns (success, violation, step_time,
    trajectory_length)."""
    np.random.seed(seed)
    car, trainer = build_dubins(dt, horizon, noise_std)
    state = dub_initial_state(car)

    success = violation = False
    step_times = []
    traj_len = 0.0
    prev_xy = state[:2].copy()
    for _ in range(DUB_MAX_STEPS):
        t0 = time.perf_counter()
        u, _ = trainer.train_step(state)
        step_times.append(time.perf_counter() - t0)
        state = car.step_sim(state, u, dt)
        traj_len += float(np.linalg.norm(state[:2] - prev_xy))
        prev_xy = state[:2].copy()
        if dub_true_margin(state) < 0.0:
            violation = True                 # record, but keep going
        if np.linalg.norm(state[:2] - DUB_GOAL[:2]) < 0.5:
            success = True
            break
    return success, violation, _avg_step_time(step_times), traj_len


# ===================== Robot-arm safety-embedded system =====================
def run_arm(seed):
    ta.horizon = ARM_HORIZON                  # use the paper N=50 for tractable timing
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    car, nom, anc = ta.make_setup()
    doc = DifferentiableOptimalControl()
    trainer = ta.make_trainer(car, nom, anc, doc)
    state = ta.initial_state(car, ta.random_feasible_q(rng))

    success = violation = False
    step_times = []
    for _ in range(ARM_MAX_STEPS):
        t0 = time.perf_counter()
        u, _ = trainer.train_step(state)
        step_times.append(time.perf_counter() - t0)
        u = np.clip(u, -ta.TORQUE_LIMIT, ta.TORQUE_LIMIT)
        state = car.step_sim(state, u, ta.dt)
        if ta.true_min_margin(state[:-1]) < 0.0:
            violation = True
            break
        if np.linalg.norm(ta.ee_of(state) - ta.target) < ta.SUCCESS_RADIUS:
            success = True
            break
    return success, violation, _avg_step_time(step_times)


# ===================== driver =====================
def aggregate(results):
    n = len(results)
    sr = 100.0 * sum(r[0] for r in results) / n
    vr = 100.0 * sum(r[1] for r in results) / n
    at = float(np.mean([r[2] for r in results]))
    return sr, vr, at


def run_row(run_fn, label, log):
    results = []
    for seed in range(N_RUNS):
        try:
            res = run_fn(seed)
        except Exception as exc:                      # keep going on a single failure
            print(f"    [{label}] seed {seed} ERROR: {exc}")
            traceback.print_exc()
            res = (False, False, float("nan"))
        results.append(res)
        print(f"    [{label}] seed {seed}: success={res[0]} violation={res[1]} "
              f"t/step={res[2]:.3f}s", flush=True)
    sr, vr, at = aggregate(results)
    line = f"{label:<16}{sr:>8.0f}%{vr:>12.0f}%{at:>20.3f}\n"
    log.write(line)
    log.flush()
    print(f"  => {label}: success={sr:.0f}% violation={vr:.0f}% avg_step_time={at:.3f}s\n", flush=True)


def header(log, title, cols):
    log.write("\n" + "=" * 60 + "\n")
    log.write(title + "\n")
    log.write(f"{cols:<16}{'Success':>9}{'Violation':>12}{'AvgStepTime(s)':>20}\n")
    log.write("-" * 60 + "\n")
    log.flush()


# ===================== parallel worker / merge modes =====================
# Layout of the Dubins tables (table key -> title, column header, ordered params).
TABLE_LAYOUT = {
    "3": ("Table 3: dt (horizon=50, noise_std=0.25)", "dt", ["0.025", "0.05", "0.1"]),
    "4": ("Table 4: horizon (dt=0.05, noise_std=0.25)", "horizon", ["25", "50", "100", "200"]),
    "5": ("Table 5: noise (dt=0.05, horizon=50)", "noise_std", ["0.25", "1.0", "4.0", "10.0"]),
}


def _row_fn(table, param):
    """Return a run function for one (table, param) Dubins row."""
    p = float(param)
    if table == "3":
        return lambda s: run_dubins(p, 50, 0.25, s)
    if table == "4":
        return lambda s: run_dubins(0.05, int(p), 0.25, s)
    if table == "5":
        return lambda s: run_dubins(0.05, 50, p, s)
    raise ValueError(f"unknown table {table}")


def worker_main():
    """Run a subset of rows/seeds; write raw per-seed lines for later merging.

    DTMPC_ROWS = comma-separated "table:param" (e.g. "3:0.025,4:100").
    DTMPC_SEEDS = comma-separated seeds (default 0..N_RUNS-1).
    """
    rows = os.environ["DTMPC_ROWS"].split(",")
    seeds = ([int(x) for x in os.environ["DTMPC_SEEDS"].split(",")]
             if os.environ.get("DTMPC_SEEDS") else list(range(N_RUNS)))
    with open(OUT, "w") as f:
        for row in rows:
            table, param = row.split(":")
            fn = _row_fn(table, param)
            for seed in seeds:
                try:
                    succ, viol, t, tl = fn(seed)
                except Exception as exc:
                    print(f"[{row}] seed {seed} ERROR: {exc}", flush=True)
                    traceback.print_exc()
                    succ, viol, t, tl = False, False, float("nan"), float("nan")
                f.write(f"{table}\t{param}\t{seed}\t{int(succ)}\t{int(viol)}\t{t:.4f}\t{tl:.4f}\n")
                f.flush()
                print(f"[{row}] seed {seed}: success={succ} violation={viol} "
                      f"t/step={t:.2f}s traj_len={tl:.1f}", flush=True)


def merge_main():
    """Aggregate raw per-seed lines (DTMPC_MERGE glob) into the final tables."""
    import glob
    data = {}
    for fn in sorted(glob.glob(os.environ["DTMPC_MERGE"])):
        with open(fn) as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 7:
                    continue
                table, param, _seed, su, vi, tm, tl = parts
                data.setdefault((table, param), []).append(
                    (int(su), int(vi), float(tm), float(tl)))

    with open(OUT, "w") as log:
        log.write("DT-MPC experimental results (parallel workers)\n")
        log.write(f"runs per row = {N_RUNS}; Dubins step cap = {DUB_MAX_STEPS}\n")
        log.write("success = reached goal (independent of violation); "
                  "violation = true safety margin < 0 at any step;\n")
        log.write("step time = mean train_step wall-clock (excl. first-step JIT warmup); "
                  "traj_len = mean closed-loop xy path length (m)\n")
        for tk in ("3", "4", "5"):
            title, col, params = TABLE_LAYOUT[tk]
            log.write("\n" + "=" * 72 + "\n" + title + "\n")
            log.write(f"{col:<14}{'Success':>9}{'Violation':>11}"
                      f"{'TrajLen(m)':>13}{'StepTime(s)':>14}\n")
            log.write("-" * 72 + "\n")
            for p in params:
                res = data.get((tk, p), [])
                if not res:
                    log.write(f"{p:<14}{'(missing)':>9}\n")
                    continue
                sr = 100.0 * sum(r[0] for r in res) / len(res)
                vr = 100.0 * sum(r[1] for r in res) / len(res)
                at = float(np.nanmean([r[2] for r in res]))
                tl = float(np.nanmean([r[3] for r in res]))
                log.write(f"{p:<14}{sr:>8.0f}%{vr:>10.0f}%{tl:>13.2f}{at:>14.3f}\n")
            log.flush()
    print(f"merged {sum(len(v) for v in data.values())} runs -> {OUT}", flush=True)


# ===================== ablation study =====================
# Compare, at the default setting, three controllers on the Dubins forest course:
#   dtmpc   : differentiable tube MPC WITH online theta=[alpha,gamma] adaptation.
#   ntmpc   : tube MPC with FIXED theta (no parameter update) -- the ancillary tube
#             still tracks the nominal, but nothing is learned.
#   nominal : the nominal MPC alone, applied directly to the true (noisy) system
#             (no ancillary tube).
ABL_DT, ABL_HORIZON, ABL_NOISE = 0.05, 100, 4.0
ABL_MAX_STEPS = int(os.environ.get("DTMPC_ABL_STEPS", "80"))
ABL_ORDER = ["dtmpc", "ntmpc", "nominal"]
ABL_NAMES = {"dtmpc": "DT-MPC (theta adapted)",
             "ntmpc": "NT-MPC (fixed theta)",
             "nominal": "Nominal MPC only"}


def run_ablation_variant(variant, seed):
    np.random.seed(seed)
    car, nominal_ocp, ancillary_ocp = build_dubins_ocps(ABL_DT, ABL_HORIZON, ABL_NOISE)
    state = dub_initial_state(car)

    if variant == "dtmpc":
        doc = DifferentiableOptimalControl()
        trainer = DTMPCTrainer(car, nominal_ocp, ancillary_ocp, DDPSolver, doc,
                               learning_rate=0.01, horizon_H=ABL_MAX_STEPS)
        control = lambda s: trainer.train_step(s)[0]
    elif variant == "ntmpc":
        tube = TubeMPC(nominal_ocp, ancillary_ocp, DDPSolver)
        control = lambda s: tube.step_tube(s)
    elif variant == "nominal":
        warm = {"u": None}

        def control(s):
            nx, nu, _ = DDPSolver.run_ddp(nominal_ocp, s, warm["u"])
            warm["u"] = np.roll(nu, shift=-1, axis=0)
            warm["u"][-1] = nu[-1]
            return nu[0]
    else:
        raise ValueError(variant)

    success = violation = False
    traj = [state[:2].copy()]
    traj_len = 0.0
    step_times = []
    for _ in range(ABL_MAX_STEPS):
        t0 = time.perf_counter()
        u = control(state)
        step_times.append(time.perf_counter() - t0)
        state = car.step_sim(state, u, ABL_DT)
        traj.append(state[:2].copy())
        traj_len += float(np.linalg.norm(traj[-1] - traj[-2]))
        if dub_true_margin(state) < 0.0:
            violation = True
        if np.linalg.norm(state[:2] - DUB_GOAL[:2]) < 0.5:
            success = True
            break
    return {"variant": variant, "seed": seed, "success": success, "violation": violation,
            "traj_len": traj_len, "step_time": _avg_step_time(step_times),
            "traj": np.array(traj)}


def ablation_worker_main():
    """Run ablation variant(s)/seeds; save scalars + trajectories to an .npz."""
    variants = os.environ["DTMPC_ABL"].split(",")
    seeds = ([int(x) for x in os.environ["DTMPC_SEEDS"].split(",")]
             if os.environ.get("DTMPC_SEEDS") else list(range(N_RUNS)))
    out_npz = os.environ.get("DTMPC_ABL_OUT", "ablation_part.npz")
    recs = []
    for v in variants:
        for s in seeds:
            try:
                r = run_ablation_variant(v, s)
            except Exception as exc:
                print(f"[abl {v}:{s}] ERROR: {exc}", flush=True)
                traceback.print_exc()
                continue
            recs.append(r)
            print(f"[abl {v}] seed {s}: success={r['success']} violation={r['violation']} "
                  f"traj_len={r['traj_len']:.1f} t/step={r['step_time']:.2f}s", flush=True)
    np.savez(out_npz,
             variant=np.array([r["variant"] for r in recs]),
             seed=np.array([r["seed"] for r in recs]),
             success=np.array([r["success"] for r in recs]),
             violation=np.array([r["violation"] for r in recs]),
             traj_len=np.array([r["traj_len"] for r in recs]),
             step_time=np.array([r["step_time"] for r in recs]),
             trajs=np.array([r["traj"] for r in recs], dtype=object))
    print(f"saved {len(recs)} ablation runs -> {out_npz}", flush=True)


def ablation_merge_main():
    """Load ablation .npz parts; append a quantitative table to OUT and save plots."""
    import glob
    recs = []
    for fn in sorted(glob.glob(os.environ["DTMPC_ABL_MERGE"])):
        z = np.load(fn, allow_pickle=True)
        for i in range(len(z["variant"])):
            recs.append({"variant": str(z["variant"][i]), "seed": int(z["seed"][i]),
                         "success": bool(z["success"][i]), "violation": bool(z["violation"][i]),
                         "traj_len": float(z["traj_len"][i]), "step_time": float(z["step_time"][i]),
                         "traj": z["trajs"][i]})

    def agg(v):
        rs = [r for r in recs if r["variant"] == v]
        n = max(len(rs), 1)
        return (100.0 * sum(r["success"] for r in rs) / n,
                100.0 * sum(r["violation"] for r in rs) / n,
                float(np.mean([r["traj_len"] for r in rs])) if rs else float("nan"),
                float(np.mean([r["step_time"] for r in rs])) if rs else float("nan"),
                len(rs))

    with open(OUT, "a") as log:
        log.write("\n\n" + "=" * 72 + "\n")
        log.write(f"ABLATION STUDY  (default: dt={ABL_DT}, horizon={ABL_HORIZON}, "
                  f"noise_std={ABL_NOISE}; {N_RUNS} runs each, step cap {ABL_MAX_STEPS})\n")
        log.write("success = reached goal; violation = true margin < 0 at any step "
                  "(independent of success)\n")
        log.write(f"{'Controller':<26}{'Success':>9}{'Violation':>11}"
                  f"{'TrajLen(m)':>13}{'StepTime(s)':>14}\n")
        log.write("-" * 72 + "\n")
        for v in ABL_ORDER:
            sr, vr, tl, st, n = agg(v)
            log.write(f"{ABL_NAMES[v]:<26}{sr:>8.0f}%{vr:>10.0f}%{tl:>13.2f}{st:>14.3f}\n")
        log.flush()

    # ---- plots ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    os.makedirs("figures", exist_ok=True)

    fig, axs = plt.subplots(1, 3, figsize=(16, 5.6))
    for ax, v in zip(axs, ABL_ORDER):
        rs = [r for r in recs if r["variant"] == v]
        for obs in DUB_OBSTACLES:
            ax.add_patch(Circle((obs[0], obs[1]), obs[2], color="red", alpha=0.35))
        for r in rs:
            tr = r["traj"]
            ax.plot(tr[:, 0], tr[:, 1], "-", lw=1.1, alpha=0.75,
                    color=("green" if r["success"] else "darkorange"))
        ax.plot(0, 0, "ks", ms=7)
        ax.plot(DUB_GOAL[0], DUB_GOAL[1], "b*", ms=16)
        sr, vr, _, _, _ = agg(v)
        ax.set_title(f"{ABL_NAMES[v]}\nsuccess {sr:.0f}%  violation {vr:.0f}%")
        ax.set_xlim(-2, 30); ax.set_ylim(-2, 30); ax.set_aspect("equal")
        ax.grid(alpha=0.3); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    fig.suptitle(f"Ablation @ dt={ABL_DT}, horizon={ABL_HORIZON}, noise_std={ABL_NOISE} "
                 f"(green=success, orange=did not reach; {N_RUNS} runs each)")
    fig.tight_layout()
    fig.savefig("figures/ablation_trajectories.png", dpi=120)

    fig2, axs2 = plt.subplots(1, 3, figsize=(13, 4))
    labels = [ABL_NAMES[v].split(" (")[0] for v in ABL_ORDER]
    succ = [agg(v)[0] for v in ABL_ORDER]
    viol = [agg(v)[1] for v in ABL_ORDER]
    tlen = [agg(v)[2] for v in ABL_ORDER]
    for ax, vals, ttl in [(axs2[0], succ, "Success rate (%)"),
                          (axs2[1], viol, "Violation rate (%)"),
                          (axs2[2], tlen, "Avg trajectory length (m)")]:
        ax.bar(labels, vals, color=["#2c7fb8", "#7fcdbb", "#fdae6b"])
        ax.set_title(ttl); ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", labelrotation=15)
    fig2.tight_layout()
    fig2.savefig("figures/ablation_metrics.png", dpi=120)
    print(f"ablation: appended table to {OUT}; saved figures/ablation_trajectories.png "
          f"+ figures/ablation_metrics.png ({len(recs)} runs)", flush=True)


def main():
    # Which tables to run, e.g. DTMPC_TABLES="3,4,5" (default: all).
    tables = set(os.environ.get("DTMPC_TABLES", "2,3,4,5").replace(" ", "").split(","))
    t_start = time.perf_counter()
    with open(OUT, "w") as log:
        log.write("DT-MPC experimental results\n")
        log.write(f"tables = {sorted(tables)}; runs per row = {N_RUNS}; "
                  f"Dubins step cap = {DUB_MAX_STEPS}; "
                  f"arm step cap = {ARM_MAX_STEPS}, arm horizon = {ARM_HORIZON}\n")
        log.write("success = target reached; violation = true safety margin < 0; "
                  "step time = mean train_step wall-clock (excl. first-step JIT warmup)\n")

        if "2" in tables:  # General
            header(log, "Table 2: General", "Scene")
            run_row(lambda s: run_dubins(0.05, 50, 0.25, s), "Dubin", log)
            run_row(run_arm, "Arm", log)

        if "3" in tables:  # dt (fix horizon=50, noise_std=0.25) -- Dubins
            header(log, "Table 3: dt (horizon=50, noise_std=0.25)", "dt")
            for dt in (0.025, 0.05, 0.1):
                run_row(lambda s, dt=dt: run_dubins(dt, 50, 0.25, s), f"{dt}", log)

        if "4" in tables:  # horizon (fix dt=0.05, noise_std=0.25) -- Dubins
            header(log, "Table 4: horizon (dt=0.05, noise_std=0.25)", "horizon")
            for h in (25, 50, 100, 200):
                run_row(lambda s, h=h: run_dubins(0.05, h, 0.25, s), f"{h}", log)

        if "5" in tables:  # noise (fix dt=0.05, horizon=50) -- Dubins
            header(log, "Table 5: noise (dt=0.05, horizon=50)", "noise_std")
            for ns in (0.25, 1.0, 4.0, 10.0):
                run_row(lambda s, ns=ns: run_dubins(0.05, 50, ns, s), f"{ns}", log)

        total = time.perf_counter() - t_start
        log.write("\n" + "=" * 60 + "\n")
        log.write(f"Total wall-clock: {total/60:.1f} min\n")
        log.flush()
    print(f"\nDONE. Total {total/60:.1f} min. Results -> {OUT}")


if __name__ == "__main__":
    if os.environ.get("DTMPC_ABL_MERGE"):
        ablation_merge_main()
    elif os.environ.get("DTMPC_ABL"):
        ablation_worker_main()
    elif os.environ.get("DTMPC_MERGE"):
        merge_main()
    elif os.environ.get("DTMPC_ROWS"):
        worker_main()
    else:
        main()
