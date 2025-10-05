import numpy as np
import matplotlib.pyplot as plt
import math
import random

# -----------------------------
# CONFIGURATION PARAMETERS
# -----------------------------
class Config:
    X_LIMITS = (0, 100)      # X range
    Y_LIMITS = (0, 100)      # Y range
    START = (5, 5)           # Start point
    GOAL = (90, 90)          # Goal point
    GOAL_RADIUS = 5.0        # Radius to consider goal reached
    STEP_SIZE = 5.0          # Step size for tree expansion
    SEARCH_RADIUS = 10.0     # Radius for rewiring
    MAX_NODES = 1500         # Maximum nodes to explore
    OBSTACLES = [
        (40, 40, 10),
        (70, 60, 10),
        (30, 70, 8),
        (60, 30, 7),
        (50, 80, 6)
    ]


# -----------------------------
# NODE CLASS
# -----------------------------
class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0


# -----------------------------
# RRT* MAIN CLASS
# -----------------------------
class RRTStar:
    def __init__(self, config):
        self.cfg = config
        self.start = Node(config.START[0], config.START[1])
        self.goal = Node(config.GOAL[0], config.GOAL[1])
        self.nodes = [self.start]

    # Generate a random point in space
    def get_random_point(self):
        if random.random() > 0.1:
            return (random.uniform(*self.cfg.X_LIMITS),
                    random.uniform(*self.cfg.Y_LIMITS))
        else:
            # Occasionally sample the goal to speed up convergence
            return self.cfg.GOAL

    # Find nearest node in the tree
    def get_nearest_node(self, x_rand):
        distances = [math.hypot(n.x - x_rand[0], n.y - x_rand[1]) for n in self.nodes]
        return self.nodes[int(np.argmin(distances))]

    # Steer function: move STEP_SIZE towards the sampled point
    def steer(self, from_node, to_point):
        theta = math.atan2(to_point[1] - from_node.y, to_point[0] - from_node.x)
        new_x = from_node.x + self.cfg.STEP_SIZE * math.cos(theta)
        new_y = from_node.y + self.cfg.STEP_SIZE * math.sin(theta)
        new_node = Node(new_x, new_y)
        new_node.parent = from_node
        new_node.cost = from_node.cost + self.cfg.STEP_SIZE
        return new_node

    # Check if node collides with any obstacle
    def is_collision_free(self, node):
        for (ox, oy, r) in self.cfg.OBSTACLES:
            if math.hypot(ox - node.x, oy - node.y) <= r:
                return False
        # Check boundaries
        if not (self.cfg.X_LIMITS[0] <= node.x <= self.cfg.X_LIMITS[1] and
                self.cfg.Y_LIMITS[0] <= node.y <= self.cfg.Y_LIMITS[1]):
            return False
        return True

    # Get nodes within search radius for rewiring
    def get_near_nodes(self, new_node):
        near_nodes = []
        for n in self.nodes:
            if math.hypot(n.x - new_node.x, n.y - new_node.y) <= self.cfg.SEARCH_RADIUS:
                near_nodes.append(n)
        return near_nodes

    # Rewire: update parent to minimize cost
    def rewire(self, new_node, near_nodes):
        for near_node in near_nodes:
            new_cost = new_node.cost + math.hypot(new_node.x - near_node.x, new_node.y - near_node.y)
            if new_cost < near_node.cost and self.is_collision_free(near_node):
                near_node.parent = new_node
                near_node.cost = new_cost

    # Check if goal reached
    def is_goal_reached(self, node):
        return math.hypot(node.x - self.cfg.GOAL[0], node.y - self.cfg.GOAL[1]) <= self.cfg.GOAL_RADIUS

    # Extract final path
    def extract_path(self, last_node):
        path = []
        node = last_node
        while node is not None:
            path.append((node.x, node.y))
            node = node.parent
        return path[::-1]

    # Main RRT* loop
    def plan(self, animate=True):
        plt.figure(figsize=(8, 8))
        plt.xlim(self.cfg.X_LIMITS)
        plt.ylim(self.cfg.Y_LIMITS)
        plt.title("RRT* Path Planning Simulation")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")

        # Plot obstacles
        for (ox, oy, r) in self.cfg.OBSTACLES:
            circle = plt.Circle((ox, oy), r, color='gray', alpha=0.6)
            plt.gca().add_patch(circle)

        plt.plot(self.start.x, self.start.y, "go", markersize=8, label="Start")
        plt.plot(self.goal.x, self.goal.y, "ro", markersize=8, label="Goal")
        plt.legend()

        for i in range(self.cfg.MAX_NODES):
            x_rand = self.get_random_point()
            nearest_node = self.get_nearest_node(x_rand)
            new_node = self.steer(nearest_node, x_rand)

            if not self.is_collision_free(new_node):
                continue

            # Find best parent (minimum cost)
            near_nodes = self.get_near_nodes(new_node)
            if near_nodes:
                costs = []
                for near_node in near_nodes:
                    cost = near_node.cost + math.hypot(near_node.x - new_node.x, near_node.y - new_node.y)
                    costs.append(cost)
                best_parent = near_nodes[int(np.argmin(costs))]
                new_node.parent = best_parent
                new_node.cost = min(costs)

            # Add to tree
            self.nodes.append(new_node)

            # Rewire
            if near_nodes:
                self.rewire(new_node, near_nodes)

            # Plot expansion
            if animate and i % 10 == 0:
                plt.plot([new_node.x, new_node.parent.x], [new_node.y, new_node.parent.y], "-b", linewidth=0.5)
                plt.pause(0.001)

            if self.is_goal_reached(new_node):
                print(f"Goal reached in {i} iterations.")
                final_path = self.extract_path(new_node)
                self.plot_final_path(final_path)
                plt.show()
                return final_path

        print("Goal not reached within max iterations.")
        plt.show()
        return None

    # Plot the final path
    def plot_final_path(self, path):
        px, py = zip(*path)
        plt.plot(px, py, '-r', linewidth=2.5, label="Final Path")
        plt.legend()
        plt.pause(0.01)


# -----------------------------
# MAIN FUNCTION
# -----------------------------
if __name__ == "__main__":
    print("Running RRT* Path Planning Simulation...")
    config = Config()
    planner = RRTStar(config)
    final_path = planner.plan(animate=True)

    if final_path:
        print("Final path found!")
    else:
        print("No feasible path found.")
