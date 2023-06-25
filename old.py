import numpy as np
import seaborn as sns

def get_available_actions(grid, state):

    actions = []
    row, col = state
    num_rows, num_cols = len(grid), len(grid[0])

    # Check up
    if row > 0 and grid[row-1][col] != str(-1):
        actions.append(0)

    # Check right
    if col < num_cols - 1 and grid[row][col+1] != str(-1):
        actions.append(1)

    # Check down
    if row < num_rows - 1 and grid[row+1][col] != str(-1):
        actions.append(2)

    # Check left
    if col > 0 and grid[row][col-1] != str(-1):
        actions.append(3)

    return actions


def initialize_q_table(grid_len, num_actions):
    grid = [[[0] * num_actions for _ in range(grid_len)]
            for _ in range(grid_len)]
    return grid

def choose_optimal_action(q_table, state, grid):

    available_actions = get_available_actions(grid, state)
    actions = [q_table[state[0]][state[1]][i] for i in available_actions ]
    action_index = np.argmax(actions)
    return available_actions[action_index]  

def get_reward(grid, state,default_reward):
    
    row, col = state
    cell_value = grid[row][col]
    if cell_value == str(7):
        return 1  
    elif cell_value == str(4):
        return -1 
    elif cell_value == str(-1):
        return None 
    elif cell_value == str(10):
        return default_reward
    elif cell_value == str(0):
        return default_reward  

def read_input_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    line1 = lines[0].split(" ")

    iterations = int(line1[0])
    learning_rate = float(line1[1])
    discount_factor = float(line1[2])
    default_reward = float(line1[3])
    e_greedy = 0
    if len(line1) > 4:
        e_greedy = float(line1[4])

    grid_len = int(lines[1].split(" ")[0])
    grid = []
    for line in lines[2:]:
        row = line.strip().split(' ')
        grid.append(row)

    return iterations, learning_rate, discount_factor, default_reward, e_greedy, grid_len,grid

def get_next_state(state, action):
    row, col = state

    # print(state,action)
    if action == 0:
        return (row -1 , col)
    elif action == 1:
        return (row, col + 1)
    elif action == 2:
        return (row +1, col)
    elif action == 3:
        return (row, col - 1)
   
   
def q_learning(grid, num_episodes, learning_rate, discount_factor, exploration_rate,default_reward,start):
    num_states = len(grid) * len(grid[0])
    num_actions = 4  # up, down, left, right
    q_table = initialize_q_table(num_states, num_actions)
    
    for episode in range(num_episodes):
        state=start
        total_reward = 0
        
        while True:
           
            # Choose an action based on epsilon-greedy strategy
            if np.random.rand() < exploration_rate:
                action = np.random.choice(get_available_actions(grid, state))
            else:
                action =  choose_optimal_action(q_table, state,grid)
              
            # Execute the action and observe the next state and reward
         
            next_state = get_next_state(state, action)
           
            reward = get_reward(grid, next_state,default_reward)

            if reward is not None:
               
                q_value = q_table[state[0]][state[1]][action]
                max_q_value = np.max(q_table[next_state[0]][next_state[1]])
                q_table[state[0]][state[1]][action]+= learning_rate * (reward + discount_factor * max_q_value - q_value)
                total_reward += reward
            
            state = next_state
            # print(state ,grid[state[0]][state[1]],reward)
            if reward == 1:  # Reached the goal state
                break
        
        print(f"Episode {episode+1}: Total Reward = {total_reward}")
    
    return q_table

def find_agent(matrix):
    for i, row in enumerate(matrix):
        for j, element in enumerate(row):
            if element == '10':
                return (i, j)
    return None

file_path = 'grid.txt'

num_episodes, learning_rate, discount_factor, default_reward, e_greedy, grid_len,grid= read_input_file(file_path)
sns.heatmap(grid,square=True,cbar=False)
start= find_agent(grid)
print(start)
q_table = q_learning(grid, num_episodes, learning_rate, discount_factor, e_greedy ,default_reward,start)


# # Example usage of the learned Q-table
# state = (3, 0)
# optimal_action = choose_optimal_action(q_table, state)
# print(f"Optimal action at state {state}: {optimal_action}")