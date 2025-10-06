# models.py
import numpy as np

# --- Simulation Parameters ---
# You can adjust this time step. A smaller value is more accurate but slower.
TIME_STEP = 0.1  # seconds

# --- Tracking System (Complex Drone Model) ---

def tracking_system_dynamics(current_state, control, disturbance):
    """
    Calculates the next state of a 4D quadrotor model using simple physics.

    Args:
        current_state (np.array): A 4-element array [x, z, vx, vz] representing
                                  the drone's position and velocity.
        control (np.array): A 2-element array [ax, az] representing the control
                            acceleration in the x and z directions.
        disturbance (np.array): A 2-element array [dx, dz] representing wind or
                                other disturbances as an acceleration.

    Returns:
        np.array: The new state vector [x', z', vx', vz'] after one time step.
    """
    # Unpack the current state for clarity
    x, z, vx, vz = current_state

    # Unpack control and disturbance
    ax, az = control
    dx, dz = disturbance

    # Update velocity based on control and disturbance accelerations
    # v_new = v_old + (a_control + a_disturbance) * dt
    new_vx = vx + (ax + dx) * TIME_STEP
    new_vz = vz + (az + dz) * TIME_STEP

    # Update position based on the new velocity
    # p_new = p_old + v_new * dt
    new_x = x + new_vx * TIME_STEP
    new_z = z + new_vz * TIME_STEP

    return np.array([new_x, new_z, new_vx, new_vz])

# --- Planning System (Simple Point Model) ---

def planning_system_dynamics(current_state, control):
    """
    Calculates the next state of a 2D point model.

    Args:
        current_state (np.array): A 2-element array [x, z] representing the
                                  planner's position.
        control (np.array): A 2-element array [vx, vz] representing the desired
                            velocity for the planner.

    Returns:
        np.array: The new state vector [x', z'] after one time step.
    """
    # Unpack the current state
    x, z = current_state

    # Unpack the control velocity
    vx, vz = control

    # Update position directly based on control velocity
    # p_new = p_old + v_control * dt
    new_x = x + vx * TIME_STEP
    new_z = z + vz * TIME_STEP

    return np.array([new_x, new_z])

# --- Example Usage (you can run this file to test it) ---
if __name__ == '__main__':
    # --- Test the Tracking System ---
    print("--- Testing Tracking System (Drone) ---")
    initial_drone_state = np.array([0.0, 0.0, 0.0, 0.0])  # At origin, at rest
    drone_control = np.array([1.0, 2.0])  # Apply acceleration (1 m/s^2 in x, 2 m/s^2 in z)
    wind_disturbance = np.array([-0.2, 0.0]) # A slight wind pushing left

    print(f"Initial Drone State: {initial_drone_state}")
    next_drone_state = tracking_system_dynamics(initial_drone_state, drone_control, wind_disturbance)
    print(f"Next Drone State:    {next_drone_state}\n")


    # --- Test the Planning System ---
    print("--- Testing Planning System (Point) ---")
    initial_plan_state = np.array([10.0, 10.0]) # Starts at (10, 10)
    plan_control = np.array([-5.0, 0.0]) # Move left at 5 m/s

    print(f"Initial Plan State: {initial_plan_state}")
    next_plan_state = planning_system_dynamics(initial_plan_state, plan_control)
    print(f"Next Plan State:    {next_plan_state}")