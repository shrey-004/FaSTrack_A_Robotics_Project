import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Example RRT* path (if you already have final_path from your RRT* simulation, use that)
# Here we just simulate a sample path for visualization
path_points = np.array([
    [5, 5], [15, 10], [25, 20], [40, 30],
    [55, 45], [70, 60], [85, 75], [90, 90]
])

# Obstacles (x, y, radius)
obstacles = [
    (40, 40, 10),
    (70, 60, 10),
    (30, 70, 8),
    (60, 30, 7),
    (50, 80, 6)
]

# Safety buffer radius (Tracking Error Bound)
epsilon = 5.0

# Create figure
plt.figure(figsize=(8, 8))
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.title("Offline Computation: Safety Buffer Visualization")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

# Plot obstacles and their expanded safe zones
for (ox, oy, r) in obstacles:
    obs = Circle((ox, oy), r, color='gray', alpha=0.6, label="Obstacle")
    safe_zone = Circle((ox, oy), r + epsilon, color='gray', alpha=0.25, linestyle='--')
    plt.gca().add_patch(obs)
    plt.gca().add_patch(safe_zone)

# Plot RRT* path
plt.plot(path_points[:,0], path_points[:,1], '-b', linewidth=2, label="RRT* Path")

# Plot safety buffer (tracking error bound) around path points
for (x, y) in path_points:
    buffer_circle = Circle((x, y), epsilon, color='skyblue', alpha=0.3)
    plt.gca().add_patch(buffer_circle)

# Start and goal markers
plt.plot(path_points[0,0], path_points[0,1], "go", markersize=8, label="Start")
plt.plot(path_points[-1,0], path_points[-1,1], "ro", markersize=8, label="Goal")

plt.legend()
plt.grid(True)
plt.savefig("safety_buffer.png", dpi=300)
plt.show()
