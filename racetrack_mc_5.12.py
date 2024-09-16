import numpy as np
import random
import matplotlib.pyplot as plt

class RaceTrack(object):
    def __init__(self, course):
        self._load_course(course)
        self.position = []
        self.velocity = []
        self.START_POSITIONS = self._get_start_positions()
        self.NOISE = 0
        self.MAX_VELOCITY = 4
        self.NUM_ACTIONS = 9
        self.ACTION_TO_VEL = [np.array([x, y]) for x in [-1, 0, 1] for y in [-1, 0, 1]]
        self.VALID_ACTION_MASK = self._generate_valid_action_mask()

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
 
    def _generate_valid_action_mask(self):
        """
        Generate a multi-dimensional mask where each state-action pair is marked as valid (1) or invalid (0).
        
        Parameters:
        - course_shape: tuple representing the shape of the grid (rows, cols).
        - max_velocity: maximum allowed velocity for both velocity dimensions.
        - num_actions: number of possible actions.
        
        Returns:
        - valid_action_mask: a multi-dimensional mask of shape (rows, cols, max_velocity+1, max_velocity+1, num_actions)
                            indicating valid actions (1) and invalid actions (0).
        """
        rows, cols = self.course.shape
        valid_action_mask = np.ones((rows, cols, self.MAX_VELOCITY + 1, self.MAX_VELOCITY + 1, self.NUM_ACTIONS), dtype=int)

        # Iterate over all possible states (position and velocity)
        for r in range(rows):
            for c in range(cols):                    
                for vel_x in range(self.MAX_VELOCITY + 1):
                    for vel_y in range(self.MAX_VELOCITY + 1):                      
                        # For each state, iterate over all possible actions
                        for action in range(self.NUM_ACTIONS):
                            # If a state is already off course, or we can take the action without moving, it is invalid
                            if self.course[r, c] == -1:
                                valid_action_mask[r, c, vel_x, vel_y, action] = 0
                            else:
                                self.position = (r, c)
                                self.velocity = np.array([vel_x, vel_y])
                                self.velocity += self.ACTION_TO_VEL[action]
                                self.velocity = np.clip(self.velocity, 0, self.MAX_VELOCITY)
                                self.position = tuple(np.array(self.position) + self.velocity)
                                # if the position never changed or we went off track from the starting track
                                if self.position == (r, c) or ((r, c) in self.START_POSITIONS and self._off_track()):
                                    valid_action_mask[r, c, vel_x, vel_y, action] = 0
        self.reset()
        return valid_action_mask

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
        # state = self.position + tuple(self.velocity)
        # assert (self.VALID_ACTION_MASK[state + (action,)] == 1), f"invalid state {state} for action {self.ACTION_TO_VEL[action]}"

        self.velocity += self.ACTION_TO_VEL[action]
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

def epsilon_greedy_policy(Q, valid_actions_mask, epsilon=0.1):
    # Q is a multi-dimensional array: (image_rows, image_cols, x_velocity, y_velocity, num_actions)
    num_actions = Q.shape[-1]  # Last dimension corresponds to the number of actions
    
    policy = np.zeros(Q.shape, dtype=np.float32)
    
    # Get greedy actions for each state (multi-dimensional argmax)
    greedy_actions = np.argmax(Q, axis=-1)  # Greedy action along the action dimension
    
    # Iterate over all states
    it = np.nditer(greedy_actions, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index  # Multi-dimensional index for the state
        
        # Get the valid actions for this state
        valid_actions = valid_actions_mask[idx]
        num_valid_actions = np.sum(valid_actions)  # Count valid actions
        
        if num_valid_actions > 0:
            # Assign (1 - epsilon) to the greedy action if it's valid
            if valid_actions[greedy_actions[idx]] == 1:
                # Distribute epsilon among valid actions
                for a in range(num_actions):
                    if valid_actions[a] == 1:  # Only update valid actions
                        policy[idx][a] = epsilon / num_valid_actions
                policy[idx][greedy_actions[idx]] += (1.0 - epsilon)
            else:
                for a in range(num_actions):
                    if valid_actions[a] == 1:  # Only update valid actions
                        policy[idx][a] = 1 / num_valid_actions
        else:
            for a in range(num_actions):
                policy[idx][a] = 1 / num_actions
        
        it.iternext()

    return policy

def update_epsilon_greedy_policy(policy, Q, s, valid_actions_mask, epsilon=0.1):
    """
    Update the epsilon-greedy policy for a specific state 's' after Q(s) is updated, considering valid actions.
    
    Parameters:
    - policy: the existing epsilon-greedy policy (multi-dimensional array).
    - Q: the action-value function (multi-dimensional array).
    - s: the multi-dimensional index of the state to update.
    - valid_actions_mask: a mask (same shape as Q) where valid actions are marked as 1.
    - epsilon: the exploration parameter.
    
    Returns:
    - Updated policy for state 's'.
    """
    num_actions = Q.shape[-1]  # Number of actions
    
    # Get the valid actions for the given state 's'
    valid_actions = valid_actions_mask[s]
    num_valid_actions = np.sum(valid_actions)  # Number of valid actions
    
    assert num_valid_actions > 0, f"no valid actions available for state {s}"

    # Get the greedy action for the updated state 's' (considering only valid actions)
    greedy_action = np.argmax(np.where(valid_actions == 1, Q[s], -np.inf))

    # Reset the policy probabilities for state 's' (only valid actions)
    policy[s] = np.zeros(num_actions)
    
    # Assign epsilon equally among valid actions
    for a in range(num_actions):
        if valid_actions[a] == 1:
            policy[s][a] = epsilon / num_valid_actions
    
    # Assign the majority of the probability mass to the greedy action if it's valid
    policy[s][greedy_action] += (1.0 - epsilon)

    return policy

def run_episode(race: RaceTrack, policy):
    race.reset()

    path = []
    states = []
    actions = []
    rewards = []

    while not race.race_complete():
        state = race.get_state()
        path.append((state[0], state[1]))
        action = np.random.choice(len(policy[state]), p=policy[state])
        reward = race.take_action(action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
    # add the end state to the path as well
    state = race.get_state()
    path.append((state[0], state[1]))
    return path, states, actions, rewards

def MC_control(course, num_episodes=100):
    racetrack = RaceTrack(course)
    rows, cols = racetrack.course.shape
    num_actions = 9
    Q = np.ones((rows, cols, racetrack.MAX_VELOCITY + 1, racetrack.MAX_VELOCITY + 1, num_actions), dtype=np.float32) * -1e3
    Q = np.where(racetrack.VALID_ACTION_MASK == 1, Q, -np.inf)
    C = np.zeros_like(Q)
    # Start with uniformly-random policies
    behavior_policy = epsilon_greedy_policy(Q, racetrack.VALID_ACTION_MASK, epsilon=1)
    target_policy = epsilon_greedy_policy(Q, racetrack.VALID_ACTION_MASK, epsilon=0)

    for i in range(num_episodes):
        print(f'Simulating episode {i}')
        _, states, actions, rewards = run_episode(racetrack, behavior_policy)
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
            target_policy = update_epsilon_greedy_policy(target_policy, Q, s, racetrack.VALID_ACTION_MASK, epsilon=0)
            if np.argmax(target_policy[s]) != a:
                break
            W *= 1/behavior_policy[s_a_index]
        # Update behavior to epislon greedy for the next episode
        behavior_policy = epsilon_greedy_policy(Q, racetrack.VALID_ACTION_MASK, epsilon=0.5)

    # Visualise episodes with final greedy policy
    for i in range(10):
        print(f'Visualise episode {i}')
        path, _, _, _ = run_episode(racetrack, target_policy)
        racetrack.visualize_path(f'results/race_track_paths/path_{i}.png', path)


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