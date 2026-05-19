import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import addcopyfighandler

# =====================================================================
# 1. THE PROBLEM DEFINITION (JAX-Compatible)
# =====================================================================
DT = 0.1
N = 50  # Horizon
NX = 3  # State dim [x, y, theta]
NU = 2  # Control dim [v, omega]

@jax.jit
def dynamics(x, u):
    """Nonlinear Dubins Car physics."""
    v, omega = u[0], u[1]
    theta = x[2]
    # Euler integration
    return jnp.array([
        x[0] + v * jnp.cos(theta) * DT,
        x[1] + v * jnp.sin(theta) * DT,
        x[2] + omega * DT
    ])

@jax.jit
def stage_cost(x, u):
    """Quadratic cost to reach the origin."""
    Q = jnp.diag(jnp.array([1.0, 1.0, 0.1])) # Penalize position mostly
    R = jnp.diag(jnp.array([0.1, 0.1]))      # Penalize control effort
    return x.T @ Q @ x + u.T @ R @ u

@jax.jit
def terminal_cost(x):
    """Heavier penalty for the final state."""
    P = jnp.diag(jnp.array([10.0, 10.0, 1.0]))
    return x.T @ P @ x

# =====================================================================
# 2. AUTOMATIC DIFFERENTIATION (The Magic)
# =====================================================================
# JAX automatically builds functions that return the matrices!
get_A = jax.jacobian(dynamics, argnums=0)
get_B = jax.jacobian(dynamics, argnums=1)

get_cx = jax.grad(stage_cost, argnums=0)
get_cu = jax.grad(stage_cost, argnums=1)
get_cxx = jax.hessian(stage_cost, argnums=0)
get_cuu = jax.hessian(stage_cost, argnums=1)
get_cxu = jax.jacobian(jax.grad(stage_cost, argnums=0), argnums=1)

get_vx = jax.grad(terminal_cost)
get_vxx = jax.hessian(terminal_cost)

# =====================================================================
# 3. THE DDP SOLVER
# =====================================================================
def run_ddp(x0, initial_u_guess, max_iters=10):
    # Initialize trajectories
    u_seq = np.copy(initial_u_guess)
    x_seq = np.zeros((N + 1, NX))
    x_seq[0] = x0
    
    # Initial Forward Pass
    for k in range(N):
        x_seq[k+1] = dynamics(x_seq[k], u_seq[k])
        
    for iteration in range(max_iters):
        print(f"--- DDP Iteration {iteration} ---")
        
        # Backward Pass Variables
        K_seq = np.zeros((N, NU, NX))
        k_seq = np.zeros((N, NU))
        
        # Start Value function at the terminal cost
        Vx = get_vx(x_seq[N])
        Vxx = get_vxx(x_seq[N])
        
        # BACKWARD PASS
        for k in reversed(range(N)):
            x, u = x_seq[k], u_seq[k]
            
            # 1. Get local linearizations using JAX
            A, B = get_A(x, u), get_B(x, u)
            cx, cu = get_cx(x, u), get_cu(x, u)
            cxx, cuu = get_cxx(x, u), get_cuu(x, u)
            cxu = get_cxu(x, u)
            
            # 2. Build Q-function expansion
            Qx = cx + A.T @ Vx
            Qu = cu + B.T @ Vx
            Qxx = cxx + A.T @ Vxx @ A
            Quu = cuu + B.T @ Vxx @ B
            Qux = cxu.T + B.T @ Vxx @ A  # Note: cxu.T is cu_x
            
            # REGULARIZATION: Prevent matrix inversion from exploding
            Quu += np.eye(NU) * 1e-4 
            Quu_inv = np.linalg.inv(Quu)
            
            # 3. Calculate gains
            k_seq[k] = -Quu_inv @ Qu
            K_seq[k] = -Quu_inv @ Qux
            
            # 4. Update Value function for the previous time step
            Vx = Qx + K_seq[k].T @ Quu @ k_seq[k] + K_seq[k].T @ Qu + Qux.T @ k_seq[k]
            Vxx = Qxx + K_seq[k].T @ Quu @ K_seq[k] + K_seq[k].T @ Qux + Qux.T @ K_seq[k]

        # FORWARD PASS
        alpha = 1.0
        num_search = 10
        
        old_cost = terminal_cost(x_seq[N]) + sum(stage_cost(x_seq[k], u_seq[k]) for k in range(N))
        
        for ll in range(num_search):
            x_new = np.zeros((N + 1, NX))
            u_new = np.zeros((N, NU))
            x_new[0] = x0
            cost = 0.0
            
            for k in range(N):
                # Apply the gains: u_new = u_old + k + K*(x_new - x_old)
                dx = x_new[k] - x_seq[k]
                u_new[k] = u_seq[k] + alpha * k_seq[k] + K_seq[k] @ dx
                
                # Simulate physics
                x_new[k+1] = dynamics(x_new[k], u_new[k])
                cost += stage_cost(x_new[k], u_new[k])
                
            cost += terminal_cost(x_new[N])
                
            if cost < old_cost:
                # Update trajectories
                x_seq = x_new
                u_seq = u_new
                print(f"Accepted alpha = {alpha} with cost {cost:.4f}")
                break
            else:
                alpha /= 2.0
                continue
            
        else:
            # (Optional) If the loop finishes without breaking, the line search failed.
            print("Line search failed to find an improvement.")
        
        # Calculate total cost to check progress
        total_cost = terminal_cost(x_seq[N]) + sum(stage_cost(x_seq[i], u_seq[i]) for i in range(N))
        print(f"Total Cost: {total_cost:.4f}")
        
    return x_seq, u_seq

# =====================================================================
# 4. TEST IT
# =====================================================================
if __name__ == "__main__":
    # Start at [x=-5, y=5, theta=-pi/2]
    start_state = np.array([-5.0, 5.0, -np.pi/2])
    
    # Initial guess: Do nothing (zero control)
    guess_u = np.zeros((N, NU))
    
    # Run DDP!
    optimal_X, optimal_U = run_ddp(start_state, guess_u)
    
    # Plotting
    plt.plot(optimal_X[:, 0], optimal_X[:, 1], '-o', label='Car Path')
    plt.plot(0, 0, 'rx', markersize=10, label='Goal')
    plt.title("DDP Solver Output")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.grid()
    plt.show()
