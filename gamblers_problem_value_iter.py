import numpy as np

def get_actions(capital):
    max_bet = min(capital, goal - capital)
    return np.arange(1, max_bet + 1)

def get_state_action_value(cur_state, action, state_values):
    win_state = cur_state + action
    lose_state = cur_state - action

    # Win case
    if win_state >= goal:
        reward_win = 1  # Reaches the goal
        value_win = p * (reward_win + state_values[goal])
    else:
        reward_win = 0
        value_win = p * (reward_win + state_values[win_state])

    # Lose case
    if lose_state <= 0:
        reward_lose = 0  # Gambler goes broke
        value_lose = (1 - p) * (reward_lose + state_values[0])
    else:
        reward_lose = 0
        value_lose = (1 - p) * (reward_lose + state_values[lose_state])

    return value_win + value_lose

def value_iteration():
    delta = 0
    # init state values randomly
    # state_values = np.random.randint(0, 10, len(states))
    state_values = np.zeros(len(states))
    # special terminal cases of lose and win
    state_values[0] = 0
    state_values[goal] = 1
    # compute values
    while True:
        old_state_values = state_values
        for state in states[1:goal]:
            actions = get_actions(state)
            # state value is the best action action value for your current state
            action_returns = np.zeros_like(actions, dtype=float)
            for i, action in enumerate(actions):
                action_returns[i] = get_state_action_value(state, action, state_values)
            state_values[state] = np.max(action_returns)
        delta = np.max(np.abs(old_state_values - state_values))
        if delta < theta:
            break

    # output greedy policy
    policy = np.zeros_like(states)
    for state in states[1:goal]:
        action_returns = np.zeros_like(actions, dtype=float)
        for i, action in enumerate(actions):
            action_returns[i] = get_state_action_value(state, action, state_values)
        policy[state] = actions[np.argmax(action_returns)]

    return state_values, policy

p = 0.4
theta = 0.0001
goal = 100
states = np.arange(goal + 1)
rewards = np.array([0, 1])
state_values, policy = value_iteration()
print(state_values)
print(policy)