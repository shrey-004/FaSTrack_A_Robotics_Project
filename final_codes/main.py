# # main.py
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D # Required for 3D plotting

# # Import the functions you've created in the other files
# from offline_computation import cost_function, compute_value_function

# # --- Grid and Plotting Parameters ---
# GRID_RESOLUTION = 11  # Number of points in each dimension of the grid (odd number is good)
# MAX_POS_ERROR = 2.0   # The grid will span from -2.0 to +2.0 meters in position error
# MAX_VEL_ERROR = 2.0   # The grid will span from -2.0 to +2.0 m/s in velocity error

# def run_midsem_evaluation():
#     """
#     Main function to run the offline computation and generate the required plots.
#     """
#     # --- 1. Setup the Grid of Initial Errors ---
#     # We are creating a grid for the 4D relative state [x_err, z_err, vx_err, vz_err].
#     # For plotting, we will create a 2D slice where the initial velocity errors are zero.
#     print("Setting up grid of initial errors...")
#     pos_errors = np.linspace(-MAX_POS_ERROR, MAX_POS_ERROR, GRID_RESOLUTION)
#     vel_errors = np.linspace(-MAX_VEL_ERROR, MAX_VEL_ERROR, GRID_RESOLUTION)
    
#     # Create a meshgrid for the position dimensions (for plotting)
#     xv_pos, zv_pos = np.meshgrid(pos_errors, pos_errors)
    
#     # Create the full 4D grid of initial errors. For this slice, vx_err and vz_err are 0.
#     initial_errors_grid = np.stack([
#         xv_pos, 
#         zv_pos, 
#         np.zeros_like(xv_pos), # Initial velocity error in x is zero for this slice
#         np.zeros_like(zv_pos)  # Initial velocity error in z is zero for this slice
#     ], axis=-1)

#     # --- 2. Calculate the Initial Cost Function ---
#     # This is what the error looks like before the game, just the simple distance.
#     print("Calculating initial cost function...")
#     initial_cost_grid = np.zeros_like(xv_pos)
#     for i in range(GRID_RESOLUTION):
#         for j in range(GRID_RESOLUTION):
#             initial_cost_grid[i, j] = cost_function(initial_errors_grid[i, j])

#     # --- 3. Run the Main Offline Computation ---
#     # This calls your other script to simulate the game and find the max error.
#     value_function_grid = compute_value_function(initial_errors_grid)

#     # --- 4. Plotting the Results ---
#     print("Generating plots...")
#     fig = plt.figure(figsize=(12, 6))
#     fig.suptitle("FaSTrack Offline Computation Results", fontsize=16)

#     # Plot for the Initial Cost Function (a sharp cone)
#     ax1 = fig.add_subplot(1, 2, 1, projection='3d')
#     ax1.plot_surface(xv_pos, zv_pos, initial_cost_grid, cmap='viridis')
#     ax1.set_title("Initial Cost Function l(r)")
#     ax1.set_xlabel("Position Error in X (m)")
#     ax1.set_ylabel("Position Error in Z (m)")
#     ax1.set_zlabel("Cost (Distance)")

#     # Plot for the Computed Value Function (a blunted cone)
#     ax2 = fig.add_subplot(1, 2, 2, projection='3d')
#     ax2.plot_surface(xv_pos, zv_pos, value_function_grid, cmap='plasma')
#     ax2.set_title("Computed Value Function V(r) (Safety Bubble)")
#     ax2.set_xlabel("Position Error in X (m)")
#     ax2.set_ylabel("Position Error in Z (m)")
#     ax2.set_zlabel("Max Guaranteed Error (m)")

#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#     plt.show()

# # --- Run the main script ---
# if __name__ == '__main__':
#     run_midsem_evaluation()




# main.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Required for 3D plotting

# Import the functions you've created in the other files
from offline_computation import cost_function, compute_value_function

# --- Grid and Plotting Parameters ---
GRID_RESOLUTION = 11  # Number of points in each dimension of the grid (odd number is good)
MAX_POS_ERROR = 2.0   # The grid will span from -2.0 to +2.0 meters in position error
MAX_VEL_ERROR = 2.0   # The grid will span from -2.0 to +2.0 m/s in velocity error

def run_midsem_evaluation():
    """
    Main function to run the offline computation and generate the required plots.
    """
    # --- 1. Setup the Grid of Initial Errors ---
    print("Setting up grid of initial errors...")
    pos_errors = np.linspace(-MAX_POS_ERROR, MAX_POS_ERROR, GRID_RESOLUTION)
    
    # Create a meshgrid for the position dimensions (for plotting)
    xv_pos, zv_pos = np.meshgrid(pos_errors, pos_errors)
    
    # Create the full 4D grid of initial errors. For this slice, vx_err and vz_err are 0.
    initial_errors_grid = np.stack([
        xv_pos, 
        zv_pos, 
        np.zeros_like(xv_pos), # Initial velocity error in x is zero for this slice
        np.zeros_like(zv_pos)  # Initial velocity error in z is zero for this slice
    ], axis=-1)

    # --- 2. Calculate the Initial Cost Function ---
    print("Calculating initial cost function...")
    initial_cost_grid = np.zeros_like(xv_pos)
    for i in range(GRID_RESOLUTION):
        for j in range(GRID_RESOLUTION):
            initial_cost_grid[i, j] = cost_function(initial_errors_grid[i, j])

    # --- 3. Run the Main Offline Computation ---
    value_function_grid = compute_value_function(initial_errors_grid)

    # --- 4. Print Numerical Results (NEW SECTION) ---
    print("\n--- Numerical Tracking Error Bounds ---")
    for i in range(GRID_RESOLUTION):
        for j in range(GRID_RESOLUTION):
            # Get the starting position error [x_err, z_err]
            start_error_pos = initial_errors_grid[i, j, :2]
            
            # Get the corresponding computed max error (the safety bound)
            final_bound = value_function_grid[i, j]
            
            print(f"Initial Position Error (x, z): ({start_error_pos[0]:+5.2f}, {start_error_pos[1]:+5.2f}) m  "
                  f"-->  Guaranteed Tracking Bound: {final_bound:.2f} m")

    # --- 5. Plotting the Results ---
    print("\nGenerating plots...")
    fig = plt.figure(figsize=(12, 6))
    fig.suptitle("FaSTrack Offline Computation Results", fontsize=16)

    # Plot for the Initial Cost Function (a sharp cone)
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_surface(xv_pos, zv_pos, initial_cost_grid, cmap='viridis')
    ax1.set_title("Initial Cost Function l(r)")
    ax1.set_xlabel("Position Error in X (m)")
    ax1.set_ylabel("Position Error in Z (m)")
    ax1.set_zlabel("Cost (Distance)")

    # Plot for the Computed Value Function (a blunted cone)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot_surface(xv_pos, zv_pos, value_function_grid, cmap='plasma')
    ax2.set_title("Computed Value Function V(r) (Safety Bubble)")
    ax2.set_xlabel("Position Error in X (m)")
    ax2.set_ylabel("Position Error in Z (m)")
    ax2.set_zlabel("Max Guaranteed Error (m)")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# --- Run the main script ---
if __name__ == '__main__':
    run_midsem_evaluation()