# offline_computation.py
import numpy as np
from models import tracking_system_dynamics, planning_system_dynamics, TIME_STEP
from tqdm import tqdm # A library to show a progress bar, helpful for long computations

# --- Game & Simulation Parameters ---

# Define the range of actions the tracker (drone) can take.
# Here, we test 9 actions: full/half/no acceleration in x and z.
TRACKER_MAX_ACCEL = 2.0  # Max acceleration in m/s^2
_accel_options = [-TRACKER_MAX_ACCEL, 0, TRACKER_MAX_ACCEL]
TRACKER_CONTROL_OPTIONS = [np.array([ax, az]) for ax in _accel_options for az in _accel_options]

# Define the range of actions the planner can take.
PLANNER_MAX_VEL = 1.0 # Max velocity in m/s
_vel_options = [-PLANNER_MAX_VEL, 0, PLANNER_MAX_VEL]
PLANNER_CONTROL_OPTIONS = [np.array([vx, vz]) for vx in _vel_options for vz in _vel_options]

# Define the worst-case disturbance (wind).
DISTURBANCE_MAX = np.array([0.5, 0.0]) # Worst case is a constant wind of 0.5 m/s^2 from one side

# How long to simulate the game for each starting error.
SIMULATION_STEPS = 50

# --- Core Functions ---

def cost_function(relative_state):
    """
    Measures the size of the "gap" based on position.
    The relative state is 4D: [x_err, z_err, vx_err, vz_err]. We only care about position error.
    """
    return np.linalg.norm(relative_state[:2]) # Euclidean distance of [x_err, z_err]

def update_relative_dynamics(tracker_state, planner_state, tracker_control, planner_control, disturbance):
    """
    Calculates the next relative state after one time step.
    """
    # 1. Find the next state for each system individually
    next_tracker_state = tracking_system_dynamics(tracker_state, tracker_control, disturbance)
    next_planner_state = planning_system_dynamics(planner_state, planner_control)

    # 2. Compute the next relative state (the new "gap")
    # Note: We need to match the dimensions. The planner state is only 2D.
    q_matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]]).T # Maps 2D planner state to 4D
    next_relative_state = next_tracker_state - q_matrix @ next_planner_state

    return next_relative_state, next_tracker_state, next_planner_state


def compute_value_function(grid_of_initial_errors):
    """
    This is the main simulation of the pursuit-evasion game.
    It iterates through a grid of possible starting errors.
    """
    print("Starting offline computation of the value function...")
    value_function_grid = np.zeros_like(grid_of_initial_errors[:, :, 0])

    # Use tqdm to show a progress bar, as this can be slow.
    for i in tqdm(range(grid_of_initial_errors.shape[0])):
        for j in range(grid_of_initial_errors.shape[1]):
            
            initial_error = grid_of_initial_errors[i, j]
            
            # For simplicity, we assume the game starts with the planner at the origin
            # and the tracker's state IS the initial error.
            tracker_state = initial_error
            planner_state = np.array([0.0, 0.0])
            
            max_error_found = cost_function(initial_error)

            # Simulate the "game" for a fixed number of steps
            for _ in range(SIMULATION_STEPS):
                
                # --- Find the tracker's best move (to MINIMIZE next error) ---
                best_tracker_control = None
                min_future_cost = float('inf')

                for tracker_control_option in TRACKER_CONTROL_OPTIONS:
                    # Predict the cost of this move, assuming the evader makes its WORST move
                    worst_planner_control_for_this_option = None
                    max_inner_future_cost = float('-inf')
                    for planner_control_option in PLANNER_CONTROL_OPTIONS:
                        # Predict next state
                        predicted_rel_state, _, _ = update_relative_dynamics(
                            tracker_state, planner_state, tracker_control_option, 
                            planner_control_option, DISTURBANCE_MAX
                        )
                        predicted_cost = cost_function(predicted_rel_state)
                        if predicted_cost > max_inner_future_cost:
                            max_inner_future_cost = predicted_cost
                    
                    # Did this tracker control give a better outcome than others?
                    if max_inner_future_cost < min_future_cost:
                        min_future_cost = max_inner_future_cost
                        best_tracker_control = tracker_control_option
                
                # --- Find the evader's worst move (to MAXIMIZE next error) ---
                worst_planner_control = None
                max_future_cost = float('-inf')
                for planner_control_option in PLANNER_CONTROL_OPTIONS:
                    # Predict the cost of this move, assuming the tracker just made its BEST move
                    predicted_rel_state, _, _ = update_relative_dynamics(
                        tracker_state, planner_state, best_tracker_control, 
                        planner_control_option, DISTURBANCE_MAX
                    )
                    predicted_cost = cost_function(predicted_rel_state)
                    if predicted_cost > max_future_cost:
                        max_future_cost = predicted_cost
                        worst_planner_control = planner_control_option
                        
                # --- Update the state based on these "optimal" moves for one step ---
                _, tracker_state, planner_state = update_relative_dynamics(
                    tracker_state, planner_state, best_tracker_control, 
                    worst_planner_control, DISTURBANCE_MAX
                )
                
                # --- Keep track of the biggest error seen in this simulation run ---
                current_error_size = np.linalg.norm(tracker_state[:2] - planner_state[:2])
                if current_error_size > max_error_found:
                    max_error_found = current_error_size

            # Store the final max error as the value for this starting grid point
            value_function_grid[i, j] = max_error_found
            
    print("Computation finished!")
    return value_function_grid


# --- Example Usage (you can run this file to test it) ---
if __name__ == '__main__':
    # Create a small grid of initial errors to test the function
    # A real grid would be much larger and finer.
    print("--- Testing Offline Computation ---")
    x_errors = np.linspace(-1, 1, 5)
    z_errors = np.linspace(-1, 1, 5)
    xv, zv = np.meshgrid(x_errors, z_errors)
    
    # For this test, assume initial velocity errors are zero
    initial_errors_grid = np.stack([xv, zv, np.zeros_like(xv), np.zeros_like(zv)], axis=-1)
    
    value_function = compute_value_function(initial_errors_grid)
    
    print("\n--- Results ---")
    print("Shape of initial errors grid:", initial_errors_grid.shape)
    print("Shape of computed value function grid:", value_function.shape)
    print("Computed Values (Max Errors):")
    print(np.round(value_function, 2))