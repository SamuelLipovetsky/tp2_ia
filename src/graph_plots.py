
import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt
# Get the absolute path of the parent directory
# Get the absolute path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Add the parent directory to the Python module search path
sys.path.insert(0, parent_dir)
from TP2 import *



def plot_graph(x, y, xlabel, ylabel, title,window_size=30):
    plt.figure()
    plt.plot(x, y)
    # Define window size for smoothing
   

    # Create empty list for smoothed data points
    smoothed_y = []

    # Apply moving average filter
    for i in range(window_size//2, len(y) - window_size//2):
        smoothed_y.append(np.mean(y[i - window_size//2: i + window_size//2 + 1]))
    plt.plot(x[window_size//2: len(y) - window_size//2], smoothed_y,  color='red', alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(title+".jpg")


def plot_comparison_graph(x1, y1, x2, y2, xlabel, ylabel, t1, t2, title,sharey,window_size=20):
    # Creating a grid of subplots with 1 row and 2 columns
    fig = plt.figure()
    fig, axs = plt.subplots(1, 2, figsize=(10, 4),sharey=sharey)
    # window_size = 20

    # Create empty list for smoothed data points
    smoothed_y1 = []
    smoothed_y2 =[]

    # Apply moving average filter
    for i in range(window_size//2, len(y1) - window_size//2):
        smoothed_y1.append(np.mean(y1[i - window_size//2: i + window_size//2 + 1]))
    for i in range(window_size//2, len(y2) - window_size//2):
        smoothed_y2.append(np.mean(y2[i - window_size//2: i + window_size//2 + 1]))
   
    # Plotting the first line graph on the left subplot
    axs[0].plot(x1, y1)
    axs[0].set_xlabel(xlabel)
    axs[0].set_ylabel(ylabel)
    axs[0].set_title(t1)
    axs[0].plot(x1[window_size//2: len(y1) - window_size//2], smoothed_y1,  color='red', alpha=0.7)
 
    # Plotting the second line graph on the right subplot
    axs[1].plot(x2, y2)
    axs[1].set_xlabel(xlabel)
    axs[1].set_ylabel(ylabel)
    axs[1].set_title(t2)
    axs[1].plot(x2[window_size//2: len(y2) - window_size//2], smoothed_y2,  color='red', alpha=0.7)
    # Adjusting the spacing between subplots
   
    plt.subplots_adjust(top=0.8)
    plt.tight_layout()

    # Displaying the line plots
    plt.savefig(title+".jpg")

if __name__=="main":
    num_episodes, learning_rate, discount_factor, default_reward, e_greedy, grid_len, grid = read_input_file(
        './src/in1_t.txt')

    sns.heatmap(grid, square=True, cbar=False)

    start = find_agent(grid)
    #standard parameters
    q_table, all_states, rewards, steps, average_max_q = q_learning(
        grid, num_episodes, learning_rate, discount_factor, 0.2, default_reward, start)


    sns.heatmap(grid, square=True, cbar=False)
    plt.savefig("grid.jpg")

    max, ind = get_max_elements(q_table, grid,better_labels=True)

    sns.heatmap(max, cbar=True, square=True, annot=ind, fmt='', annot_kws={'size': 20})
    plt.tight_layout()
    plt.savefig("q_table_std.jpg")

    #printing standard graphs
    plot_graph([i for i in range(len(rewards))], rewards,
                "Episodes", "Total Rewards", "Rewards per episode")
    plot_graph([i for i in range(len(steps))], steps, "Episodes",
                "Steps per episode", "Steps per episode")
    plot_graph([i for i in range(len(average_max_q))], average_max_q, "Episodes",
                "Average max Q-value per episode", "Average max Q-value per episode")

    #running the grid for more episodes.
    q_table_me, all_states_me, rewards_me, steps_me, average_max_q_me = q_learning(
        grid, 1000, learning_rate, discount_factor, 0.2, default_reward, start)
    plt.figure()
    max, ind = get_max_elements(q_table_me, grid,better_labels=True)


    sns.heatmap(max, cbar=True, square=True, annot=ind, fmt='', annot_kws={'size': 20})
    plt.tight_layout()
    plt.savefig("q_table_me.jpg")

    #changing_e_greedy
    q_table_e_05, all_states_e_05, rewards_e_05, steps_e_05, average_max_q_e_05 = q_learning(
        grid, num_episodes, learning_rate, discount_factor, 0.5, default_reward, start)
    q_table_e_02, all_states_e_02, rewards_e_02, steps_e_02, average_max_q_e_02 = q_learning(
        grid, num_episodes, learning_rate, discount_factor, 0.2, default_reward, start)
    plt.figure()

    plot_comparison_graph([i for i in range(len(steps_e_02))], steps_e_02, [i for i in range(len(steps_e_05))], steps_e_05, "Episodes",
                        "Steps per episode", "Epsilon=0.2", "Epsilon=0.5", "Number of steps comparison with Epsilon changes",True)

    #changing learning rate 
    q_table_l_05, all_states_l_05, rewards_l_05, steps_l_05, average_max_l_05= q_learning(
        grid, num_episodes, 0.5, discount_factor, e_greedy, default_reward, start)
    q_table_l_001, all_states_l_001, reward_l_001, steps_l_001, average_max_l_001= q_learning(
        grid, num_episodes, 0.01, discount_factor, e_greedy, default_reward, start)

    plot_comparison_graph([i for i in range(len(steps_l_001))], steps_l_001, [i for i in range(len(steps_l_05))], steps_l_05, "Episodes",
                        "Steps per episode", "Alpha=0.01", "Alpha=0.5", "Number of steps comparison with alpha changes",True)


    #changing default reward
    q_table_dr_01, all_states_dr_01, rewards_dr_01, steps_dr_01, average_max_dr_01= q_learning(
        grid, num_episodes, learning_rate, discount_factor, e_greedy, -0.4, start)
    q_table_dr_0, all_states_dr_0, reward_dr_0, steps_dr_0, average_max_dr_0= q_learning(
        grid, num_episodes, learning_rate, discount_factor, e_greedy, -0.0001, start)

    plot_comparison_graph([i for i in range(len(steps_dr_0))], steps_dr_0, [i for i in range(len(steps_dr_01))], steps_dr_01, "Episodes",
                        "Steps per episode", "Defaut reward=-0.0001", "Default reward=-0.4", "Number of steps comparison with Default rewards changes",True)
