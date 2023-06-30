
from graph_plots import *
from TP2 import *
import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Get the absolute path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Add the parent directory to the Python module search path
sys.path.insert(0, parent_dir)


def simulate(q_table, grid, n_episodes):
    wins = 0
    loses = 0
    rows = len(q_table)
    cols = len(q_table[0])

    valid_indices = []
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 0:
                valid_indices.append((i, j))

    for i in range(n_episodes):
        state = random.choice(valid_indices)
        number_steps = 0
        while True:

            # print(state)
            action = choose_optimal_action(q_table, state, grid)

            next_state = get_next_state(state, action)

            reward = get_reward(grid, next_state, default_reward)

            if reward == -1 or number_steps > 300:
                loses += 1
                break
            if reward == 1:
                wins += 1
                break

            state = next_state
            number_steps += 1

    # returns winrate
    return wins/(wins+loses)


num_episodes, learning_rate, discount_factor, default_reward, e_greedy, grid_len, grid = read_input_file(
    './src/in1_t.txt')

start = find_agent(grid)
training_n = [100+i*500 for i in range(20)]
values = []
# simulating only normal restarting
for i in training_n:
    print(i)
    q_table, all_states, rewards, steps, average_max_q = q_learning(
        grid, i, learning_rate, discount_factor, 0.2, default_reward, start)
    values.append(simulate(q_table, grid, 1000))
plot_graph(training_n, values,
           "Number of steps", "Win rate", "win rate x steps", window_size=3)


# simulating normal restart x random restart
training_n = [100+i*100 for i in range(200)]
normal_start = []
random_start = []

for i in training_n:
    print(i)
    q_table, all_states, rewards, steps, average_max_q = q_learning(
        grid, i, learning_rate, discount_factor, 0.2, default_reward, start)
    normal_start.append(simulate(q_table, grid, 100))
    q_table, all_states, rewards, steps, average_max_q = q_learning(
        grid, i, learning_rate, discount_factor, 0.2, default_reward, start, True)
    random_start.append(simulate(q_table, grid, 100))

plot_comparison_graph(training_n, normal_start, training_n, random_start,
                      "Number of steps", "Win rate of model", "Normal restart", "Random restart", "Comparison of Q-laerning with normal and random restart", True)
