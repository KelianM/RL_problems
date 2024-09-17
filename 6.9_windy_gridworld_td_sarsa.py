import numpy as np
import matplotlib.pyplot as plt

class GridWorld(object):
    def __init__(self, grid, start, goal) -> None:
        self.GRID = grid
        self.START = start
        self.GOAL = goal
        self.position = start
        self.ACTION_TO_DELTA = {
            0: np.array([-1, 0]),  # Up: row decreases, column stays the same
            1: np.array([1, 0]),   # Down: row increases, column stays the same
            2: np.array([0, 1]),   # Right: row stays the same, column increases
            3: np.array([0, -1])   # Left: row stays the same, column decreases
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
        pos = np.array(self.position)
        pos[0] += wind
        pos += self.ACTION_TO_DELTA[action]

        rows, cols = self.GRID.shape
        pos[0] = np.clip(pos[0], 0, rows - 1)
        pos[1] = np.clip(pos[1], 0, cols - 1)
        self.position = tuple(pos)
        reward = 0 if self.goal_reached() else -1
        return reward

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
    for episode in range(num_episodes):
        print(f"Running episode {episode + 1}")
        gridworld.reset()
        s = gridworld.get_state()
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
        num_actions_per_episode.append(total_actions)
    return num_actions_per_episode

def plot(num_actions_per_episode):
    episodes = np.arange(len(num_actions_per_episode))
    plt.plot(num_actions_per_episode, episodes)
    plt.xlabel('Time Steps')
    plt.ylabel('Episodes')
    plt.title('Number of Actions per Episode (Switched Axes)')
    plt.grid(True)
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
num_actions_per_episode = sarsa_td(gridworld, num_episodes=170, epsilon=epsilon, alpha=alpha, discount=discount)
plot(num_actions_per_episode)