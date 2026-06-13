import os
import time
import json
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

from dynamics.dubins_car import DubinsCar
from dynamics.safety_embedded import SafetyEmbeddedDynamics, SafetyEmbeddedVisualizer
from solvers.ocp_interface import OCP
from solvers.optimal_control import DDPSolver
from solvers.costs import QuadraticCost, TerminalCost, BaseCost
from learning.doc_engine import DifferentiableOptimalControl
from learning.dt_mpc_loop import DTMPCTrainer

# ========================================================================= #
# Custom Classes for True Dynamics Scenarios
# ========================================================================= #

class WindGustDubinsCar(DubinsCar):
    """A Dubins Car that experiences a sudden gust of wind between specific steps."""
    def __init__(self, wheelbase=0.25, gust_steps=(5, 15), gust_force=(0.0, -5.0, 0.0)):
        super().__init__(wheelbase=wheelbase)
        self.gust_steps = gust_steps
        self.gust_force = np.array(gust_force)
        self._current_step = 0
        
    def step(self, x, u, dt):
        """Custom step using numpy to track time and add force."""
        dx = np.array(self.dynamics(x, u))
        if self.gust_steps[0] <= self._current_step <= self.gust_steps[1]:
            dx += self.gust_force
        
        self._current_step += 1
        return x + dx * dt
        
    def reset_time(self):
        self._current_step = 0

class MismatchDubinsCar(DubinsCar):
    """A Dubins Car with a parameterized mismatch vs the nominal planner."""
    def __init__(self, true_wheelbase=0.5):
        # Planner assumes 0.25, true dynamics might have 0.5 (sluggish turning)
        super().__init__(wheelbase=true_wheelbase)

# ========================================================================= #
# Environment Definitions
# ========================================================================= #

ENVIRONMENTS = {}

# 1. Standard Forest
ENVIRONMENTS["forest"] = {
    "obstacles": np.array([
        [5.0, 2.0, 1.0], [3.0, 6.0, 1.2], [7.0, 8.0, 1.5],
        [10.0, 4.0, 1.0], [12.0, 10.0, 1.5], [15.0, 7.0, 1.0],
        [8.0, 15.0, 1.8], [5.0, 12.0, 2.0], [12.0, 18.0, 2.2],
        [18.0, 12.0, 1.5], [20.0, 5.0, 1.5], [22.0, 15.0, 1.2],
        [16.0, 22.0, 2.5], [10.0, 25.0, 1.8], [25.0, 10.0, 3.2],
        [20.0, 25.0, 1.5], [25.0, 20.0, 1.2], [5.0, 20.0, 2.0]
    ]),
    "goal": np.array([28.0, 28.0, 0.0, 0.0]),
    "start": (0.0, 0.0, 0.0)
}

ENVIRONMENTS["forest1"] = {
    "obstacles": np.array(ENVIRONMENTS["forest"]["obstacles"], copy=True),
    "goal": np.array(ENVIRONMENTS["forest"]["goal"], copy=True),
    "start": tuple(ENVIRONMENTS["forest"]["start"]),
}

ENVIRONMENTS["forest2"] = {
    "obstacles": np.array([
        [5.0, 2.0, 1.0], [3.0, 6.0, 1.2], [7.0, 8.0, 1.5],
        [10.0, 4.0, 1.0], [12.0, 10.0, 1.5], [15.0, 7.0, 1.0],
        [8.0, 15.0, 1.8], [5.0, 12.0, 2.0], [12.0, 18.0, 2.2],
        [18.0, 12.0, 1.5], [20.0, 5.0, 1.5], [22.0, 15.0, 1.2],
        [16.0, 22.0, 2.5], [10.0, 25.0, 1.8], [25.0, 10.0, 3.2],
        [20.0, 25.0, 1.5], [25.0, 20.0, 1.2], [5.0, 20.0, 2.0]
    ]),
    "goal": np.array([25.0, 4.0, 0.0, 0.0]),
    "start": (8.0, 22.0, 0.0)
}

# 2. Narrow Passageway
_narrow_obs = []
for x_pos in range(5, 25, 2):
    _narrow_obs.append([x_pos, 13.0, 1.2]) # Bottom wall
    _narrow_obs.append([x_pos, 17.0, 1.2]) # Top wall
ENVIRONMENTS["narrow"] = {
    "obstacles": np.array(_narrow_obs),
    "goal": np.array([28.0, 15.0, 0.0, 0.0]),
    "start": (0.0, 15.0, 0.0)
}

def cbf_factory(obs_array):
    def cbf(x):
        dists = jnp.sqrt((x[0] - obs_array[:, 0]) ** 2 + (x[1] - obs_array[:, 1]) ** 2)
        return dists - obs_array[:, 2]
    return cbf

# ========================================================================= #
# Unified Runner
# ========================================================================= #

def make_nominal_ancillary(car, goal_state, horizon=50, dt=0.1):
    Q_nom = jnp.diag(jnp.array([1.0, 1.0, 0.5, 100.0]))
    R_nom = jnp.diag(jnp.array([0.1, 0.1]))
    P_nom = jnp.diag(jnp.array([100.0, 100.0, 50.0, 0.0]))
    nominal_ocp = OCP(system=car,
                      stage_cost=QuadraticCost(Q_nom, R_nom, x_ref=goal_state),
                      terminal_cost=TerminalCost(P_nom, x_ref=goal_state),
                      horizon=horizon, dt=dt)

    Q_anc = jnp.diag(jnp.array([50.0, 50.0, 10.0, 100.0]))
    R_anc = jnp.diag(jnp.array([1.0, 1.0]))
    P_anc = jnp.diag(jnp.array([200.0, 200.0, 50.0, 0.0]))
    ancillary_ocp = OCP(system=car,
                        stage_cost=QuadraticCost(Q_anc, R_anc),
                        terminal_cost=TerminalCost(P_anc),
                        horizon=horizon, dt=dt)
    return nominal_ocp, ancillary_ocp

def get_initial_state(car, cbf_fn, start_tuple):
    base = jnp.array(list(start_tuple))
    b0 = float(jnp.sum(car.relaxed_barrier(cbf_fn(base), car.alpha)))
    return np.array([base[0], base[1], base[2], b0])

def run_episode(
    env_name="forest",
    true_sys_type="nominal", # "nominal", "wind", "mismatch"
    controller_type="learning", # "learning", "static", "pure_nominal"
    learning_rate=0.01,      # 0.0 for static baselines
    noise_std=0.25,          # standard noise multiplier
    noise_distribution="uniform", # "gaussian" or "uniform"
    noise_bound=None,        # half-width for uniform noise
    wheelbase=0.25,           # true system wheelbase (only for mismatch scenario)
    gust_force=(0.0, -5.0, 0.0), # only for wind scenario
    steps=60,
    alpha0=0.3,
    gamma0=0.1,
    seed=42,
    dt=0.1
):
    np.random.seed(seed)
    env = ENVIRONMENTS[env_name]
    cbf = cbf_factory(env["obstacles"])
    
    # 1. Setup True Dynamics System
    true_base_sys = None
    if true_sys_type == "wind":
        true_base_sys = WindGustDubinsCar(wheelbase=0.25, gust_steps=(15, 25), gust_force=gust_force)
        true_base_sys.reset_time()
    elif true_sys_type == "mismatch":
        # Planner expects wheelbase 0.25, system has 0.60
        true_base_sys = MismatchDubinsCar(true_wheelbase=wheelbase)
        
    # 2. Setup Planner System
    base_car = DubinsCar(wheelbase=0.25)
    car = SafetyEmbeddedDynamics(
        base_system=base_car, 
        constraint_func=cbf,
        alpha=alpha0, 
        gamma=gamma0, 
        noise_std=noise_std,
        true_base_system=true_base_sys
    )
    
    nominal_ocp, ancillary_ocp = make_nominal_ancillary(car, env["goal"], dt=dt)
    doc = DifferentiableOptimalControl()
    trainer = DTMPCTrainer(car, nominal_ocp, ancillary_ocp, DDPSolver, doc,
                           learning_rate=learning_rate, horizon_H=steps)

    current_state = get_initial_state(car, cbf, env["start"])
    states, nom_states, alpha_hist, gamma_hist, loss_hist = [current_state.copy()], [], [car.alpha], [car.gamma], []

    pure_nominal_state = None
    pure_nominal_prev_control = None
    
    status = "TIMEOUT"
    
    from time import perf_counter
    t0 = perf_counter()
    for k in range(steps):
        try:
            if controller_type == "pure_nominal":
                if pure_nominal_state is None:
                    pure_nominal_state = np.copy(current_state)

                nom_x, nom_u, _ = DDPSolver.run_ddp(
                    nominal_ocp,
                    pure_nominal_state,
                    pure_nominal_prev_control,
                )
                pure_nominal_prev_control = np.roll(nom_u, shift=-1, axis=0)
                pure_nominal_prev_control[-1] = nom_u[-1]
                pure_nominal_state = np.asarray(nom_x[1])
                u = nom_u[0]
                nom_states.append(nom_x[1].copy())
                diag = {"alpha": car.alpha, "gamma": car.gamma, "loss": 0.0}
            else:
                trainer.learning_rate = learning_rate if controller_type == "learning" else 0.0
                u, diag = trainer.train_step(current_state)
                nominal_state = trainer.current_nominal_state
                if nominal_state is None:
                    nominal_state = np.copy(current_state)
                nom_states.append(np.asarray(nominal_state).copy())
            
            # Step the simulation forward (will use true_sys if provided)
            current_state = car.step_sim(
                current_state,
                u,
                dt=dt,
                noise_distribution=noise_distribution,
                noise_bound=noise_bound,
            )
            
            states.append(current_state.copy())
            alpha_hist.append(diag["alpha"])
            gamma_hist.append(diag["gamma"])
            loss_hist.append(diag["loss"])
            
            dist = float(np.linalg.norm(current_state[0:2] - env["goal"][0:2]))
            
            if np.any(np.array(cbf(current_state)) < 0.0):
                status = "COLLIDED"
                break
            
            if dist < 0.5:
                status = "GOAL_REACHED"
                break
        except Exception as e:
            print(f"Error at step {k}: {e}")
            status = "CRASHED"
            break

    elapsed = perf_counter() - t0
        
    return {
        "status": status,
        "steps_taken": len(states) - 1,
        "time_s": elapsed,
        "states": np.array(states),
        "nom_states": np.array(nom_states),
        "alpha": np.array(alpha_hist),
        "gamma": np.array(gamma_hist),
        "loss": np.array(loss_hist)
    }

def print_result_summary(name, res):
    print(f"[{name.upper()}] Status: {res['status']} | Steps: {res['steps_taken']} | "
          f"Time: {res['time_s']:.2f}s | "
          f"Final alpha: {res['alpha'][-1]:.3f} | Final gamma: {res['gamma'][-1]:.3f}")

def experiment_baselines(env_name="forest", sys_type="nominal", noise_std=0.5, dt=0.1, wheelbase=0.25, gust_force=(0.0, -5.0, 0.0), plot_learning=True, steps=60, test_str="", seed=42):
    print(f"\n=== Baseline Comparison ({env_name}, {sys_type}) ===")
    
    # 1. DT-MPC with Learning
    res_learn = run_episode(env_name=env_name, true_sys_type=sys_type, controller_type="learning", 
                            learning_rate=0.01, noise_std=noise_std, steps=steps, dt=dt, wheelbase=wheelbase, gust_force=gust_force, seed=seed)
    print_result_summary("Learning", res_learn)

    # 2. Static DT-MPC (No Learning)
    res_static = run_episode(env_name=env_name, true_sys_type=sys_type, controller_type="static", 
                             learning_rate=0.0, noise_std=noise_std, steps=steps, dt=dt, wheelbase=wheelbase, gust_force=gust_force, seed=seed)
    print_result_summary("Static", res_static)

    # 3. Pure Nominal MPC (No Tube Tracking)
    res_nom = run_episode(env_name=env_name, true_sys_type=sys_type, controller_type="pure_nominal", 
                          learning_rate=0.0, noise_std=noise_std, steps=steps, dt=dt, wheelbase=wheelbase, gust_force=gust_force, seed=seed)
    print_result_summary("Nominal", res_nom)

    res_list = [res_learn] if plot_learning else None
    nominal_trajectories = [res["nom_states"] for res in res_list] if res_list is not None else None
    alphas_list = [res["alpha"] for res in res_list] if res_list is not None else None
    gammas_list = [res["gamma"] for res in res_list] if res_list is not None else None

    SafetyEmbeddedVisualizer.visualize_multiple_trajectories(
        trajectories=[res_learn["states"], res_static["states"], res_nom["states"]],
        obstacles=ENVIRONMENTS[env_name]["obstacles"],
        goal=ENVIRONMENTS[env_name]["goal"],
        figsize=(8,8),
        filename=f"figures/baseline_{env_name}_{sys_type}_{test_str}.png",
        labels=["DT-MPC (Learning)", "DT-MPC (Static)", "Nominal MPC"],
        nominal_trajectories=nominal_trajectories,
        alphas_list=alphas_list,
        gammas_list=gammas_list
    )

def experiment_multirun(env_name="forest", episodes=3, dt=0.05, test_str="", plot_learning=True):
    print(f"\n=== Multi-run Learning ({env_name}) ===")
    a, g = 0.3, 0.1
    res_list = []
    trajs = []
    labels = []
    for ep in range(episodes):
        res = run_episode(env_name=env_name, controller_type="learning", learning_rate=0.02, steps=80, alpha0=a, gamma0=g, dt=dt)
        print(f"Ep {ep+1} | Status: {res['status']:<12} | alpha: {a:.3f}->{res['alpha'][-1]:.3f} | gamma: {g:.3f}->{res['gamma'][-1]:.3f}")
        a, g = res["alpha"][-1], res["gamma"][-1]
        trajs.append(res["states"])
        labels.append(f"Episode {ep+1} (a={res['alpha'][-1]:.2f})")
        res_list.append(res)
    
    
    nominal_trajectories = [res["nom_states"] for res in res_list] if plot_learning else None
    alphas_list = [res["alpha"] for res in res_list] if plot_learning else None
    gammas_list = [res["gamma"] for res in res_list] if plot_learning else None
    
    SafetyEmbeddedVisualizer.visualize_multiple_trajectories(
        trajectories=trajs,
        obstacles=ENVIRONMENTS[env_name]["obstacles"],
        goal=ENVIRONMENTS[env_name]["goal"],
        figsize=(8,8),
        filename=f"figures/multirun_{env_name}_{test_str}.png",
        labels=labels,
        nominal_trajectories=nominal_trajectories,
        alphas_list=alphas_list,
        gammas_list=gammas_list
    )

def experiment_transfer_learning(dt=0.1):
    print(f"\n=== Transfer Learning (Forest -> Narrow) ===")
    a, g = 0.3, 0.1
    # Train on Forest
    res_forest = run_episode(env_name="forest", controller_type="learning", learning_rate=0.02, steps=80, alpha0=a, gamma0=g, dt=0.05)
    a, g = res_forest["alpha"][-1], res_forest["gamma"][-1]
    print(f"Pre-trained on Forest. Ending params: alpha={a:.3f}, gamma={g:.3f}")

    # Transfer to Narrow
    res_narrow_transfer = run_episode(env_name="narrow", controller_type="learning", learning_rate=0.02, steps=100, alpha0=a, gamma0=g, dt=0.05)
    print_result_summary("Transfer Narrow", res_narrow_transfer)

    # Compare to fresh Narrow
    res_narrow_fresh = run_episode(env_name="narrow", controller_type="learning", learning_rate=0.02, steps=100, alpha0=0.3, gamma0=0.1, dt=0.05)
    print_result_summary("Fresh Narrow", res_narrow_fresh)
    
    SafetyEmbeddedVisualizer.visualize_multiple_trajectories(
        trajectories=[res_narrow_transfer["states"], res_narrow_fresh["states"]],
        obstacles=ENVIRONMENTS["narrow"]["obstacles"],
        goal=ENVIRONMENTS["narrow"]["goal"],
        figsize=(8,8),nominal_trajectories=[res_narrow_transfer["nom_states"], res_narrow_fresh["nom_states"]],
        filename=f"figures/transfer_learning.png",
        labels=["Pre-trained Parameters", "Fresh Parameters"]
    )

def experiment_uniform_noise_sweep(
    env_name="forest1",
    noise_bounds=(0.1, 0.5, 1.0, 5.0),
    episodes=3,
    dt=0.05,
    steps=60,
    controller_type="learning",
    test_str="",
):
    print(f"\n=== Uniform Noise Sweep ({env_name}) ===")
    for bound in noise_bounds:
        print(f"-- noise in [-{bound}, {bound}] --")
        trajs = []
        nom_trajs = []
        labels = []
        for ep in range(episodes):
            res = run_episode(
                env_name=env_name,
                controller_type=controller_type,
                learning_rate=0.01 if controller_type == "learning" else 0.0,
                noise_std=bound,
                noise_distribution="uniform",
                noise_bound=bound,
                steps=steps,
                dt=dt,
                seed=42 + ep,
            )
            print_result_summary(f"{controller_type.title()} noise +/-{bound}", res)
            trajs.append(res["states"])
            nom_trajs.append(res["nom_states"])
            labels.append(f"run {ep+1}")

        SafetyEmbeddedVisualizer.visualize_multiple_trajectories(
            trajectories=trajs,
            obstacles=ENVIRONMENTS[env_name]["obstacles"],
            goal=ENVIRONMENTS[env_name]["goal"],
            figsize=(8, 8),
            filename=f"figures/{env_name}_uniform_noise_{str(bound).replace('.', 'p')}_{test_str}.png",
            labels=labels,
            nominal_trajectories=nom_trajs,
        )


if __name__ == "__main__":
    import argparse
    import sys
    
    print("Running experiments...")
    
    # # 1. Narrow Passageway

    ## experiment_uniform_noise_sweep("forest2", noise_bounds=[5.0], episodes=5, dt=0.05, steps=60, controller_type="learning", test_str="learning")
    # experiment_baselines("narrow", "nominal", noise_std=2.0, dt=0.05, steps=40)
    # experiment_baselines("narrow", "nominal", noise_std=0.0, dt=0.05, steps=40, test_str="no_noise", plot_learning=False)
    # experiment_baselines("narrow", "nominal", noise_std=0.25, dt=0.05, steps=40, test_str="n025", plot_learning=False)
    # experiment_baselines("narrow", "nominal", noise_std=0.5, dt=0.05, steps=40, test_str="n05", plot_learning=False)
    # experiment_baselines("narrow", "nominal", noise_std=1.0, dt=0.05, steps=40, test_str="n1", plot_learning=False)
    # experiment_baselines("narrow", "nominal", noise_std=2.0, dt=0.05, steps=40, test_str="n2", plot_learning=False)
    # experiment_uniform_noise_sweep("forest1", noise_bounds=(0.1, 0.5, 1.0, 5.0), episodes=3, dt=0.05, steps=60, controller_type="learning")

    # experiment_baselines("forest2", "nominal", noise_std=2.0, dt=0.05)
    # experiment_baselines("forest2", "nominal", noise_std=0.0, dt=0.05, seed=189, test_str="seed189_noise0", steps=80) # Base case    
    # experiment_baselines("forest2", "nominal", noise_std=0.0, dt=0.05, seed=42, test_str="seed42_noise0", steps=80) # Base case    
    # experiment_baselines("forest2", "nominal", noise_std=1.0, dt=0.05, seed=50, test_str="seed50_noise1", steps=80) # High noise case

    # # 2. Wind Gust
    ## experiment_baselines("forest2", "wind", gust_force=(0.0, -0.5, 0.0), dt=0.05, test_str="d05", steps=80)
    ## experiment_baselines("forest2", "wind", gust_force=(0.0, -1.0, 0.0), dt=0.05, test_str="d1", steps=80)
    ## experiment_baselines("forest2", "wind", gust_force=(0.0, -5.0, 0.0), dt=0.05, test_str="d5", steps=80)
    # experiment_baselines("forest", "wind", noise_std=0.0, dt=0.05)
    # experiment_baselines("forest2", "wind", noise_std=0.0, dt=0.05)
    # experiment_baselines("narrow", "wind", noise_std=0.0, dt=0.05, gust_force=(0.0, -50.0, 0.0), test_str="gust_d50", steps=80)

    # 3. Model Mismatch
    ## experiment_baselines("narrow", "mismatch", noise_std=0.0, dt=0.025, wheelbase=0.3, test_str="wb03")
    ## experiment_baselines("narrow", "mismatch", noise_std=0.0, dt=0.025, wheelbase=0.4, test_str="wb04")
    ## experiment_baselines("narrow", "mismatch", noise_std=0.0, dt=0.025, wheelbase=0.5, test_str="wb05")
    
    # 4. Large Noise
    # experiment_baselines("forest", "nominal", noise_std=0.5, dt=0.025)
    # experiment_baselines("forest", "nominal", noise_std=5.0, dt=0.025)
    # experiment_baselines("forest2", "nominal", noise_std=0.5, dt=0.025)
    # experiment_baselines("forest2", "nominal", noise_std=5.0, dt=0.025)

    
    # 5. Multi-run and Transfer
    experiment_multirun("forest", episodes=3, dt=0.05)
    # experiment_transfer_learning(dt = 0.05)

