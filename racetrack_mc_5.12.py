import numpy as np
import random
import matplotlib.pyplot as plt

class Race(object):
    def __init__(self, course):
        self._load_course(course)
        self.position = []
        self.velocity = []
        self.START_POSITIONS = self._get_start_positions()
        self.NOISE = 0
        self.MAX_VELOCITY = 4
        self.ACTIONS = [np.array([x, y]) for x in [-1, 0, 1] for y in [-1, 0, 1]]

    def _load_course(self, course):
        """
        Load course. Internally represented as numpy array
        Shamelessly stolen from https://gist.github.com/pat-coady/26fafa10b4d14234bfde0bb58277786d
        """
        y_size, x_size = len(course), len(course[0])
        self.course = np.zeros((x_size, y_size), dtype=np.int16)
        for y in range(y_size):
            for x in range(x_size):
                point = course[y][x]
                if point == 'o':
                    self.course[x, y] = 1
                elif point == '-':
                    self.course[x, y] = 0
                elif point == '+':
                    self.course[x, y] = 2
                elif point == 'W':
                    self.course[x, y] = -1
        # flip left/right so (0,0) is in bottom-left corner
        self.course = np.fliplr(self.course)
                    
    def _get_start_positions(self):
        start_positions =  np.where(self.course == 0)
        # return start positions as a list of 2D tuple indices
        return list(zip(start_positions[0], start_positions[1]))
 
    def _off_track(self):
        rows, cols = self.course.shape
        row, col = self.position
        
        # Check if the position is outside the array bounds
        if row < 0 or row >= rows or col < 0 or col >= cols:
            return True
        
        # Check if the value at the given position is -1
        if self.course[row, col] == -1:
            return True
        
        return False

    def race_complete(self):
        return self.course[self.position] == 2

    def reset(self):
        self.position = random.choice(self.START_POSITIONS)
        self.velocity = np.zeros(2, dtype=np.int16)

    def get_state(self):
        return self.position + tuple(self.velocity)

    def take_action(self, action):
        self.velocity += self.ACTIONS[action]
        self.velocity = np.clip(self.velocity, 0, self.MAX_VELOCITY)
        self.position = tuple(np.array(self.position) + self.velocity)

        if self._off_track():
            self.reset()

        return -1

    def visualize_path(self, filename, path):
        course = self.course  # The race course (np array)
        
        # Create a colour map for the course: -1 -> grey, 0 -> blue, 1 -> white, 2 -> green
        cmap = plt.cm.colors.ListedColormap(['grey', 'blue', 'white', 'green'])
        bounds = [-1.5, -0.5, 0.5, 1.5, 2.5]  # Boundaries for the colours
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

        # Plot the course
        plt.imshow(course, cmap=cmap, norm=norm)

        # Extract the x and y coordinates of the path
        path_x, path_y = zip(*path)

        # Plot the path as red dots
        plt.plot(path_y, path_x, marker='o', color='red', markersize=5, linestyle='-')

        # Display the plot
        plt.title('Race Course with Path')
        plt.savefig(filename)
        plt.close()


def epsilon_greedy_policy(Q, epsilon=0.1):
    # Q is a multi-dimensional array: (image_rows, image_cols, x_velocity, y_velocity, num_actions)
    state_dims = Q.shape[:-1]  # Extract dimensions of the state space (without actions)
    num_actions = Q.shape[-1]  # Last dimension corresponds to the number of actions
    
    # Create an epsilon-greedy policy for each state
    policy = np.ones(Q.shape) * (epsilon / num_actions)  # Initialize uniformly with epsilon
    
    # Get greedy actions for each state (multi-dimensional argmax)
    greedy_actions = np.argmax(Q, axis=-1)  # Greedy action along the action dimension
    
    # Assign the majority of the probability mass to the greedy action
    it = np.nditer(greedy_actions, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index  # Multi-dimensional index for the state
        policy[idx][greedy_actions[idx]] += (1.0 - epsilon)  # Adjust probability for the greedy action
        it.iternext()
    
    return policy

def update_epsilon_greedy_policy(policy, Q, s, epsilon=0.1):
    """
    Update the epsilon-greedy policy for a specific state 's' after Q(s) is updated.
    
    Parameters:
    - policy: the existing epsilon-greedy policy (multi-dimensional array).
    - Q: the action-value function (multi-dimensional array).
    - s: the multi-dimensional index of the state to update.
    - epsilon: the exploration parameter.
    """
    num_actions = Q.shape[-1]  # Assuming actions are the last dimension of Q

    # Get the greedy action for the updated state 's'
    greedy_action = np.argmax(Q[s])
    
    # Reset the policy probabilities for state 's'
    policy[s] = np.ones(num_actions) * (epsilon / num_actions)
    
    # Assign the majority of the probability mass to the greedy action
    policy[s][greedy_action] += (1.0 - epsilon)
    
    return policy

def run_episode(race: Race, policy):
    race.reset()

    path = []
    states = []
    actions = []
    rewards = []

    while not race.race_complete():
        state = race.get_state()
        path.append((state[0], state[1]))
        action = np.random.choice(len(policy[state]), p=policy[state])
        race.take_action(action)

        states.append(state)
        actions.append(action)
        rewards.append(race.take_action(action))
    # add the end state to the path as well
    state = race.get_state()
    path.append((state[0], state[1]))
    return path, states, actions, rewards

def MC_control(course, num_episodes=100):
    race = Race(course)
    rows, cols = race.course.shape
    num_actions = 9
    Q = np.ones((rows, cols, race.MAX_VELOCITY + 1, race.MAX_VELOCITY + 1, num_actions), dtype=np.float32) * -1e3
    C = np.zeros_like(Q)
    # Start with uniformly-random behavior policy and greedy target policy
    behavior_policy = np.ones_like(Q)/num_actions
    target_policy = epsilon_greedy_policy(Q, epsilon=0)

    for i in range(num_episodes):
        print(f'Simulating episode {i}')
        _, states, actions, rewards = run_episode(race, behavior_policy)
        num_steps = len(rewards)
        G = 0
        W = 1
        for t in reversed(range(num_steps)):
            r = rewards[t]
            s = states[t]
            a = actions[t]
            s_a_index = s + (a,)
            G += r
            C[s_a_index] += W
            Q[s_a_index] += (W/C[s_a_index]) * (G - Q[s_a_index])
            target_policy = update_epsilon_greedy_policy(target_policy, Q, s, epsilon=0)
            if np.argmax(target_policy[s]) != a:
                break
            W *= 1/behavior_policy[s_a_index]
        # Update behavior to epislon greedy for the next episode
        behavior_policy = epsilon_greedy_policy(Q, epsilon=0.5)

    # Visualise episodes with final greedy policy
    for i in range(10):
        print(f'Visualise episode {i}')
        path, _, _, _ = run_episode(race, target_policy)
        race.visualize_path(f'results/race_track_paths/path_{i}.png', path)


# Courses shamelessly stolen from https://gist.github.com/pat-coady/26fafa10b4d14234bfde0bb58277786d
big_course = ['WWWWWWWWWWWWWWWWWW',
              'WWWWooooooooooooo+',
              'WWWoooooooooooooo+',
              'WWWoooooooooooooo+',
              'WWooooooooooooooo+',
              'Woooooooooooooooo+',
              'Woooooooooooooooo+',
              'WooooooooooWWWWWWW',
              'WoooooooooWWWWWWWW',
              'WoooooooooWWWWWWWW',
              'WoooooooooWWWWWWWW',
              'WoooooooooWWWWWWWW',
              'WoooooooooWWWWWWWW',
              'WoooooooooWWWWWWWW',
              'WoooooooooWWWWWWWW',
              'WWooooooooWWWWWWWW',
              'WWooooooooWWWWWWWW',
              'WWooooooooWWWWWWWW',
              'WWooooooooWWWWWWWW',
              'WWooooooooWWWWWWWW',
              'WWooooooooWWWWWWWW',
              'WWooooooooWWWWWWWW',
              'WWooooooooWWWWWWWW',
              'WWWoooooooWWWWWWWW',
              'WWWoooooooWWWWWWWW',
              'WWWoooooooWWWWWWWW',
              'WWWoooooooWWWWWWWW',
              'WWWoooooooWWWWWWWW',
              'WWWoooooooWWWWWWWW',
              'WWWoooooooWWWWWWWW',
              'WWWWooooooWWWWWWWW',
              'WWWWooooooWWWWWWWW',
              'WWWW------WWWWWWWW']

# Tiny course for debug

tiny_course = ['WWWWWW',
               'Woooo+',
               'Woooo+',
               'WooWWW',
               'WooWWW',
               'WooWWW',
               'WooWWW',
               'W--WWW',]

MC_control(big_course, num_episodes=100)