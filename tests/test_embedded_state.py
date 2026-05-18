import numpy as np
from dynamics.safety_embedded import SafetyEmbeddedDynamics, SafetyEmbeddedVisualizer

# Default environment artifacts (used by visualizer / examples)
OBSTACLES = np.array([
    [5.0, 5.0, 1.0],
    [3.0, 7.0, 1.0],
    [7.0, 3.0, 1.0],
    # Top-left obstacle
    [1.0, 9.0, 1.0],
])

def run_open_loop_simulation():
    # 1. Initialize the stateless plant
    dt = 0.1
    wheelbase = 0.25
    car = SafetyEmbeddedDynamics(wheelbase=wheelbase, obstacles=OBSTACLES)
    
    # 2. Define Reality (Initial State)
    current_state = np.array([0.0, 0.0, 0.0, 0.0]) # x, y, theta, barrier_state
    
    # 3. Create a predetermined trajectory (e.g., drive straight, then turn left)
    # Control vector is 2D: [v, steering_rate]. Set constant forward velocity v=1.0
    horizon = 50
    predetermined_steer = np.zeros(horizon)
    predetermined_steer[20:40] = 0.5  # Turn left in the middle of the trajectory
    
    # 4. Simulation Loop
    states_history = [current_state]
    
    for k in range(horizon):
        # control = [v, steering_rate]
        u_k = np.array([1.0, predetermined_steer[k]])
        
        # Step the physics forward using the stateless model
        current_state = car.step(current_state, u_k, dt) 
        print(f"Step {k}: State = {current_state}")
        states_history.append(current_state)
        
    states_history = np.array(states_history)
    
    # 5. Visualize the result using the shared visualizer
    SafetyEmbeddedVisualizer.visualize_trajectory(
        trajectory=states_history,
        obstacles=OBSTACLES,
        goal=np.array([10.0, 10.0]),
        goal_radius=0.25,
    )

if __name__ == "__main__":
    run_open_loop_simulation()
