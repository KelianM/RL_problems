import numpy as np
import matplotlib.pyplot as plt

class GridWorld(object):
    def __init__(self, grid, start, goal) -> None:
        self.GRID = grid
        self.START = start
        self.GOAL = goal
        self.position = start
        self.ACTION_TO_DELTA = {
            0: np.array([-1, 0]),   # Up
            1: np.array([1, 0]),    # Down
            2: np.array([0, 1]),    # Right
            3: np.array([0, -1]),   # Left
            # For King's moves (include diagonals)
            4: np.array([-1, 1]),   # Up-Right (Diagonal)
            5: np.array([-1, -1]),  # Up-Left (Diagonal)
            6: np.array([1, 1]),    # Down-Right (Diagonal)
            7: np.array([1, -1]),    # Down-Left (Diagonal)
            # # # Stationary move (useful only with wind)
            # 8: np.array([0, 0])    # Down-Left (Diagonal)
        }
        self.NUM_ACTIONS = len(self.ACTION_TO_DELTA)
        
    def reset(self):
        self.position = self.START

    def get_state(self):
        return self.position

    def goal_reached(self):
        return self.position == self.GOAL

    def take_action(self, action):
        wind = self.GRID[self.position]
        # Stochastic wind (randomly change it's effect by 1 when present)
        if wind > 0:
            wind += np.random.randint(-1, 2)

        pos = np.array(self.position)
        pos[0] += wind
        pos += self.ACTION_TO_DELTA[action]

        rows, cols = self.GRID.shape
        pos[0] = np.clip(pos[0], 0, rows - 1)
        pos[1] = np.clip(pos[1], 0, cols - 1)
        self.position = tuple(pos)
        reward = 0 if self.goal_reached() else -1
        return reward

    def visualise_path(self, filename, path):
        grid = np.array(self.GRID)
        fig, ax = plt.subplots()
        
        # Create a colormap for wind strength
        cmap = plt.cm.Blues
        norm = plt.Normalize(vmin=0, vmax=2)  # Wind levels from 0 to 2

        # Plot the grid with wind strength
        ax.imshow(grid, cmap=cmap, norm=norm)

        # Mark the start and goal points
        ax.text(self.START[1], self.START[0], 'S', ha='center', va='center', color='green', fontsize=12, fontweight='bold')
        ax.text(self.GOAL[1], self.GOAL[0], 'G', ha='center', va='center', color='blue', fontsize=12, fontweight='bold')

        # Extract x and y coordinates from the path
        path_x = [pos[0] for pos in path]
        path_y = [pos[1] for pos in path]
        
        # Plot arrows along the path
        for i in range(len(path) - 1):
            dx = path_x[i+1] - path_x[i]
            dy = path_y[i+1] - path_y[i]
            ax.quiver(path_y[i], path_x[i], dy, dx, angles='xy', scale_units='xy', scale=1, color='red', width=0.005)

        # Set up the grid
        ax.set_xticks(np.arange(grid.shape[1]))
        ax.set_yticks(np.arange(grid.shape[0]))
        ax.set_xticklabels(np.arange(grid.shape[1]))
        ax.set_yticklabels(np.arange(grid.shape[0]))

        # Add gridlines
        ax.grid(color='black', linestyle='-', linewidth=1)
        ax.set_xticks(np.arange(-.5, grid.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-.5, grid.shape[0], 1), minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5)

        plt.savefig(filename)
        plt.show()
        
def epsilon_greedy_policy(Q, state, epsilon):
    """
    Implements epsilon-greedy policy for a given state (position).
    
    Parameters:
    - Q: The Q-matrix (rows x cols x actions)
    - position: Tuple representing the current (row, col) position
    - epsilon: Probability of choosing a random action (exploration)
    
    Returns:
    - Chosen action (index) based on epsilon-greedy strategy
    """
    row, col = state
    num_actions = Q.shape[2]
    
    # With probability (1 - epsilon), exploit: choose the best action
    if np.random.rand() > epsilon:
        action = np.argmax(Q[row, col])  # Greedy choice: action with the highest Q-value
    else:
        # With probability epsilon, explore: choose a random action
        action = np.random.randint(num_actions)  # Random action
        
    return action

def sarsa_td(gridworld: GridWorld, num_episodes, epsilon, alpha, discount):
    rows, cols = gridworld.GRID.shape
    Q = np.zeros((rows, cols, gridworld.NUM_ACTIONS), dtype=np.float32)
    num_actions_per_episode = []
    total_actions = 0
    paths = []
    for episode in range(num_episodes):
        print(f"Running episode {episode + 1}")
        gridworld.reset()
        s = gridworld.get_state()
        path = [s]
        a = epsilon_greedy_policy(Q, s, epsilon=epsilon)
        while not gridworld.goal_reached():
            total_actions += 1
            r = gridworld.take_action(a)
            s_next = gridworld.get_state()
            a_next = epsilon_greedy_policy(Q, s_next, epsilon=epsilon)
            s_a = s + (a,)
            s_a_next = s_next + (a_next,)
            Q[s_a] += alpha * (r + discount * Q[s_a_next] - Q[s_a])
            s, a = s_next, a_next
            path.append(s)
        num_actions_per_episode.append(total_actions)
        paths.append(path)
    return num_actions_per_episode, paths

def plot(filename, num_actions_per_episode):
    episodes = np.arange(len(num_actions_per_episode))
    plt.plot(num_actions_per_episode, episodes)
    plt.xlabel('Time Steps')
    plt.ylabel('Episodes')
    plt.title('Number of Actions per Episode (Switched Axes)')
    plt.grid(True)
    plt.savefig(filename)
    plt.show()

epsilon = 0.1
alpha = 0.5
discount = 1
# define windy grid as in textbook
grid = np.zeros((7, 10), dtype=int)
grid[:, [3, 4, 5, 8]] = 1
grid[:, [6, 7]] = 2

start = (3,  0)
goal = (3, 7)

gridworld = GridWorld(grid, start, goal)
num_actions_per_episode, paths = sarsa_td(gridworld, num_episodes=500, epsilon=epsilon, alpha=alpha, discount=discount)
plot("results/windy_gridworld/windy_gridworld_perf_king.png", num_actions_per_episode)
gridworld.visualise_path("results/windy_gridworld/windy_gridworld_path_king.png", paths[-1]) # Visualise the final (best) path