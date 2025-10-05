import numpy as np
import matplotlib.pyplot as plt

class TrackerSimulation:
    """
    A simple 2D simulation showing how a tracker (real robot)
    follows a planner while keeping the tracking error bounded.
    """

    def __init__(self, kp=1.5, dt=0.05, total_time=10.0):
        self.kp = kp            # feedback gain
        self.dt = dt            # time step
        self.total_time = total_time

    def simulate(self, r0=(4.0, 3.0), v_p=1.0):
        """
        Simulate relative dynamics between tracker and planner.
        r = [r_x, r_y] = (tracker - planner)
        """
        r = np.array(r0, dtype=float)
        trajectory = [r.copy()]
        t = [0]

        for _ in np.arange(0, self.total_time, self.dt):
            # Planner moves with constant velocity v_p in x
            # Tracker matches x velocity and corrects y error
            v_s = v_p
            w_s = -self.kp * r[1]  # proportional control in y

            # Relative dynamics
            drx = v_s - v_p
            dry = w_s

            r += np.array([drx, dry]) * self.dt
            trajectory.append(r.copy())
            t.append(t[-1] + self.dt)

        return np.array(trajectory), np.array(t)

    def plot(self, traj, t):
        r_norm = np.linalg.norm(traj, axis=1)
        plt.figure(figsize=(7, 5))
        plt.plot(t, r_norm, label='Tracking Error |r|')
        plt.xlabel("Time (s)")
        plt.ylabel("Tracking Error Magnitude")
        plt.title("Tracking Error Over Time")
        plt.grid(True)
        plt.legend()
        plt.show()

        # Plot trajectory in (r_x, r_y) space
        plt.figure(figsize=(6,6))
        plt.plot(traj[:,0], traj[:,1], '-o', markersize=3)
        plt.xlabel("r_x (X error)")
        plt.ylabel("r_y (Y error)")
        plt.title("Relative State Trajectory (r_x, r_y)")
        plt.axhline(0, color='k', linestyle='--', linewidth=0.8)
        plt.axvline(0, color='k', linestyle='--', linewidth=0.8)
        plt.grid(True)
        plt.axis('equal')
        plt.show()


if __name__ == "__main__":
    print("Running Tracker Simulation (FaSTrack-inspired)...")
    tracker = TrackerSimulation(kp=2.0, dt=0.05, total_time=8.0)
    traj, t = tracker.simulate(r0=(3.0, 2.0), v_p=1.0)
    tracker.plot(traj, t)
