# RL Problems

This repository contains implementations of selected exercises and examples from the "Reinforcement Learning: An Introduction" textbook by Sutton & Barto. Each script or folder corresponds to a different exercise or example, exploring core concepts of reinforcement learning such as value iteration, Monte Carlo methods, Temporal Difference learning, and control problems.

## Contents

- **4.9_gamblers_problem_value_it.py**: Solves the Gambler's Problem using value iteration, exploring optimal policy and state value computation.
  
- **5.12_racetrack_mc.py**: Implements the Racetrack problem using Monte Carlo methods for policy control and improvement.
  
- **6.9_windy_gridworld_td_sarsa.py**: Solves the Windy Gridworld using the SARSA algorithm, a form of Temporal Difference learning, to find an optimal policy. Also solved using Q-learning.
  
- **7.2_n_step_td_comparison.py**: Compares different n-step Temporal Difference (TD) learning methods to study their convergence and performance.

- **ex10.1_mountain_car/**: A folder containing scripts and code related to the Mountain Car problem, exploring control using various reinforcement learning algorithms. Includes tiling for extracting features and solved using the semi-gradient SARSA algorithm with linear function approximation.

## Results

All output results, including plots and data, are saved in the `results` folder.

## How to Use

Each script can be run individually to observe the implementation and results of the corresponding RL exercise. For example:
```bash
python 4.9_gamblers_problem_value_it.py
```