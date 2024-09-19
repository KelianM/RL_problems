import numpy as np
import matplotlib.pyplot as plt

class RandomWalk:
    def __init__(self, num_states=5):
        self.num_states = num_states
        self.state = num_states // 2  # Start at the center state
        self.end_states = [0, num_states + 1]

    def reset(self):
        self.state = self.num_states // 2
        return self.state

    def step(self, action=None):
        if np.random.rand() < 0.5:
            next_state = self.state - 1
        else:
            next_state = self.state + 1
        
        if next_state == 0:
            return next_state, -1, True
        elif next_state == self.num_states + 1:
            return next_state, 1, True
        else:
            self.state = next_state
            return next_state, 0, False

def n_step_td(random_walk: RandomWalk, n=3, alpha=0.1, gamma=1.0, episodes=100):
    # Value function initialized to 0
    V = np.zeros(random_walk.num_states + 2) # Num states + ending states on each side
    true_values = np.arange(-(random_walk.num_states // 2) - 1, random_walk.num_states // 2 + 2) / ((random_walk.num_states + 2) // 2)

    errors = []
    for episode in range(episodes):
        state = random_walk.reset()
        states = [state]
        rewards = [0]
        T = float('inf')
        t = 0
        while True:
            if t < T:
                next_state, reward, done = random_walk.step()
                rewards.append(reward)
                states.append(next_state)
                if done:
                    T = t + 1
            
            tau = t - n + 1
            if tau >= 0:
                G = sum([gamma ** (i - tau - 1) * rewards[i] for i in range(tau + 1, min(tau + n, T) + 1)])
                if tau + n < T:
                    G += gamma ** n * V[states[tau + n]]
                V[states[tau]] += alpha * (G - V[states[tau]])

            t += 1
            if tau == T - 1:
                break
        
        errors.append(np.sqrt(np.mean((V - true_values) ** 2)))

    return V, errors

def sum_of_td_errors(random_walk: RandomWalk, n=3, alpha=0.1, gamma=1.0, episodes=100):
    # Value function initialized to 0
    V = np.zeros(random_walk.num_states + 2) # Num states + ending states on each side
    true_values = np.arange(-(random_walk.num_states // 2) - 1, random_walk.num_states // 2 + 2) / ((random_walk.num_states + 2) // 2)

    errors = []
    for episode in range(episodes):
        state = random_walk.reset()
        states = [state]
        rewards = [0]
        T = float('inf')
        t = 0
        while True:
            if t < T:
                next_state, reward, done = random_walk.step()
                rewards.append(reward)
                states.append(next_state)
                if done:
                    T = t + 1

            tau = t - n + 1
            if tau >= 0:
                sum_delta = 0
                for k in range(tau, min(tau + n, T)):
                    V_next = V[states[k + 1]] if k + 1 < len(states) else 0
                    delta_k = rewards[k + 1] + gamma * V_next - V[states[k]]
                    sum_delta += delta_k
                V[states[tau]] += alpha * sum_delta

            t += 1
            if tau == T - 1:
                break

        errors.append(np.sqrt(np.mean((V - true_values) ** 2)))

    return V, errors

# Run the experiment
num_experiments = 1000
env = RandomWalk(num_states=5)
episodes = 100
alpha_values = [0.05, 0.1, 0.2, 0.3]
n = 3  # Number of steps in n-step TD

# Store the mean errors for plotting
mean_errors_td_alpha = {}
mean_errors_sum_td_alpha = {}

# Run experiments for different alpha values
for alpha in alpha_values:
    all_errors_td = []
    all_errors_sum_td = []
    print(f"Running experiments for alpha = {alpha}")
    for _ in range(num_experiments):
        V_td, errors_td = n_step_td(env, n=n, alpha=alpha, episodes=episodes)
        V_sum_td, errors_sum_td = sum_of_td_errors(env, n=n, alpha=alpha, episodes=episodes)
        all_errors_td.append(errors_td)
        all_errors_sum_td.append(errors_sum_td)
    all_errors_td = np.array(all_errors_td)
    all_errors_sum_td = np.array(all_errors_sum_td)

    # Calculate mean over experiments
    mean_errors_td = np.mean(all_errors_td, axis=0)
    mean_errors_sum_td = np.mean(all_errors_sum_td, axis=0)

    # Store the mean errors for plotting
    mean_errors_td_alpha[alpha] = mean_errors_td
    mean_errors_sum_td_alpha[alpha] = mean_errors_sum_td

# Plotting the Mean Squared Error (MSE) over episodes for all alphas on a single graph
plt.figure(figsize=(12, 8))
for alpha in alpha_values:
    plt.plot(mean_errors_td_alpha[alpha], label=f'n-step TD (alpha={alpha})')
    plt.plot(mean_errors_sum_td_alpha[alpha], linestyle='--', label=f'Sum of TD Errors (alpha={alpha})')

plt.xlabel('Episodes')
plt.ylabel('Mean Squared Error')
plt.title('Comparison of n-step TD vs Sum of TD Errors for Different Alpha Values')
plt.legend()
plt.show()