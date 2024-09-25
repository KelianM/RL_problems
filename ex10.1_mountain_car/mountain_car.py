import numpy as np
import matplotlib.pyplot as plt

from tiles import IHT, tiles

NUM_TILES = 4096
NUM_TILINGS = 8
POS_BOUNDS = [-1.2, 0.5]
VEL_BOUNDS = [-0.07, 0.07]
START_BOUNDS = [-0.6, -0.4]
NUM_ACTIONS = 3

class MountainCar(object):
    def __init__(self, grid, start, goal) -> None:
        self.GRID = grid
        self.position = 0
        self.velocity = 0
        self.ACTION_TO_DELTA = {
            0: -1,   # Backwards
            1: 0,    # Same
            2: 1,    # Forwards
        }
        self.GOAL = POS_BOUNDS[1]
        self.reset()

        
    def reset(self):
        self.position = np.random.uniform(START_BOUNDS[0], START_BOUNDS[1])
        self.velocity = 0

    def get_state(self):
        return self.position, self.velocity

    def goal_reached(self):
        return self.position == self.GOAL

    def take_action(self, action):
        self.position += self.velocity
        self.position = np.clip(self.position, POS_BOUNDS[0], POS_BOUNDS[1])

        self.velocity += 0.001*self.ACTION_TO_DELTA[action] - 0.0025 * np.cos(3 * self.position)
        self.velocity = np.clip(self.velocity, VEL_BOUNDS[0], VEL_BOUNDS[1])

        reward = 0 if self.goal_reached() else -1
        return reward

def get_tile_idxs(iht: IHT, s, a):
    pos, vel = s
    pos_range = POS_BOUNDS[1] - POS_BOUNDS[0]
    vel_range = VEL_BOUNDS[1] - VEL_BOUNDS[0]
    tile_idxs = tiles(iht,NUM_TILINGS,[NUM_TILINGS*pos/ pos_range, NUM_TILINGS*vel/vel_range], [a])
    return tile_idxs

def Q(iht: IHT, w, s, a):
    tile_idxs = get_tile_idxs(iht, s, a)
    return np.sum(w[tile_idxs])

def epsilon_greedy_policy(iht, w, s, epsilon):
    """
    Implements epsilon-greedy policy for a given state given the tiling object, weight matrix, state, and epislon.

    Returns:
    - Chosen action (index) based on epsilon-greedy strategy
    """    
    # With probability (1 - epsilon), exploit: choose the best action
    if np.random.rand() > epsilon:
        Q_values = [Q(iht, w, s, a) for a in range(NUM_ACTIONS)] # Generate Q values for each action
        action = np.argmax(Q_values)  # Greedy choice: action with the highest Q-value
    else:
        # With probability epsilon, explore: choose a random action
        action = np.random.randint(NUM_ACTIONS)  # Random action
        
    return action

def semi_grad_sarsa(mountain_car: MountainCar, num_episodes, epsilon, alpha, discount):
    iht = IHT(NUM_TILES)

    num_actions_per_episode = []
    w = np.zeros((NUM_TILES * NUM_TILINGS), dtype=np.float32)
    for episode in range(num_episodes):
        print(f"Running episode {episode}")
        num_actions = 0
        mountain_car.reset()
        s = mountain_car.get_state()
        a = epsilon_greedy_policy(iht, w, s, epsilon)
        while True:
            num_actions += 1
            r = mountain_car.take_action(a)
            s_next = mountain_car.get_state()
            s_tile_idxs = get_tile_idxs(iht, s, a)
            if mountain_car.goal_reached():
                w[s_tile_idxs] += alpha * (r - Q(iht, w, s, a))
                break
            a_next = epsilon_greedy_policy(iht, w, s_next, epsilon)
            w[s_tile_idxs] += alpha * (r + discount * Q(iht, w, s_next, a_next) - Q(iht, w, s, a))
            s, a = s_next, a_next
        num_actions_per_episode.append(num_actions)
    return num_actions_per_episode

def plot(filename, num_actions_per_episode):
    episodes = np.arange(len(num_actions_per_episode))
    plt.plot(episodes, num_actions_per_episode)
    plt.xlabel('Episodes')
    plt.ylabel('Time Steps')
    plt.title('Number of Actions per Episode')
    plt.grid(True)
    plt.semilogy()
    plt.savefig(filename)
    plt.show()

epsilon = 0.0
alpha = 0.5/NUM_TILINGS
discount = 1
# define windy grid as in textbook
grid = np.zeros((7, 10), dtype=int)
grid[:, [3, 4, 5, 8]] = 1
grid[:, [6, 7]] = 2

start = (3,  0)
goal = (3, 7)

gridworld = MountainCar(grid, start, goal)
num_actions_per_episode = semi_grad_sarsa(gridworld, num_episodes=500, epsilon=epsilon, alpha=alpha, discount=discount)
plot("results/mountain_car_sg_sarsa/mountain_car_sg_sarsa_perf.png", num_actions_per_episode)