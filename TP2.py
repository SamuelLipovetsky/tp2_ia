import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import copy
from matplotlib import animation
def get_available_actions(grid, state):

    actions = []
    row, col = state
    num_rows, num_cols = len(grid), len(grid[0])

    # Check up
    if row > 0 and grid[row-1][col] != -1:
        actions.append(0)

    # Check right
    if col < num_cols - 1 and grid[row][col+1] != -1:
        actions.append(1)

    # Check down
    if row < num_rows - 1 and grid[row+1][col] != -1:
        actions.append(2)

    # Check left
    if col > 0 and grid[row][col-1] != -1:
        actions.append(3)


    return actions


def initialize_q_table(grid_len, num_actions):
    
    grid = [[[0] * num_actions for _ in range(grid_len)]
            for _ in range(grid_len)]
    return grid

def choose_optimal_action(q_table, state, grid):

    available_actions = get_available_actions(grid, state)
    actions = [q_table[state[0]][state[1]][i] for i in available_actions ]
    #slip chance
    if np.random.rand() < 0.2:
        p_actions=[]
        if 1 in available_actions:
            p_actions.append(1)
        if 3 in available_actions:
            p_actions.append(3)
        if len(p_actions)>0:
            return np.random.choice(p_actions)

    action_index = np.argmax(actions)
    return available_actions[action_index]  

def get_reward(grid, state,default_reward):
    
    row, col = state
    cell_value = grid[row][col]
    if cell_value ==  7:
        return 1  
    elif cell_value == 4:
        return -1 
    elif cell_value ==  -1:
        return None 
    elif cell_value ==  0:
        return default_reward
    elif cell_value == 10:
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
    grid = np.array(grid).astype(float)
    print(grid)
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
    num_states = len(grid) 
    num_actions = 4  # up, down, left, right
    q_table = initialize_q_table(num_states, num_actions)
    all_states=[]
    for episode in range(num_episodes):
        state=start
        total_reward = 0
        
        while True:
            all_states.append(state)
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
       
            if reward == 1 :  # Reached the goal state
                break
        
        print(f"Episode {episode+1}: Total Reward = {total_reward}")
    
    return q_table,all_states

def find_agent(matrix):
    for i, row in enumerate(matrix):
        for j, element in enumerate(row):
            if element == 10:

                return (i, j)
    
    return None

def get_max_elements(matrix,grid):

    max_elements = [[-1 for _ in range(len(matrix[0]))] for _ in range(len(matrix[0]))]
    indexes =[["0" for _ in range(len(matrix[0]))] for _ in range(len(matrix[0]))]
   
    for i,row in enumerate(matrix):
        for j,col in  enumerate(row):
           
            actions = get_available_actions(grid,(i,j))
            
          
            to_be_max =[-10000,-10000,-10000,-10000 ]
            for k in actions:
                to_be_max[k] =matrix[i][j][k] 
          
            max =np.argmax(to_be_max)
            max_elements[i][j] = np.max(to_be_max)  

            if grid[i][j]==4 :
                indexes[i][j]="n"
                max_elements[i][j]=+1
            if grid[i][j]==7: 
                indexes[i][j]="n"
                max_elements[i][j]=1
            if grid[i][j]==-1:
                indexes[i][j]="n"
                max_elements[i][j]=0
            elif max==0:
                indexes[i][j]= "c"
            elif max==1:
                indexes[i][j]= "d"
            elif max==2:
                indexes[i][j]= "b"
            elif max==3:
                indexes[i][j]= "e"

            

    return max_elements, indexes

def init(N):
    sns.heatmap(np.zeros((N,N)) ,square=True,cbar=False)

def animate(i):
    data=data_list[i]
    sns.heatmap(data,square=True,cbar=False)

def make_gif(all_agent_states,grid,original_start,std_value):
    frames=[]
    grid[original_start[0]][original_start[1]]=0
    for i in all_agent_states:
        temp=copy.deepcopy(grid)
        temp[i[0]][i[1]]=10
        frames.append(temp)

    fig=plt.figure()
    anim =animation.FuncAnimation(fig,animate,init_func=init(len(grid[0])),frames=len(frames))
    pillow=animation.PillowWriter(fps=7)
    anim.save("teste.gif",writer=pillow)
file_path = 'grid.txt'

num_episodes, learning_rate, discount_factor, default_reward, e_greedy, grid_len,grid= read_input_file(file_path)
# print(get_available_actions(grid,(0,0)))

sns.heatmap(grid,square=True,cbar=False)
plt.savefig('grid.jpg')
start= find_agent(grid)
print(start)
q_table,all_states = q_learning(grid, num_episodes, learning_rate, discount_factor, e_greedy ,default_reward,start)
max,ind = get_max_elements(q_table,grid)
print(max,ind)
sns.heatmap(max,cbar=True,square=True,annot=ind,fmt='')
plt.plot()
plt.savefig("idk.jpg")
# make_gif(all_states,grid,start,default_reward)
data_list=[]

grid[start[0]][start[1]]=0
limit =0
for i in all_states:
        if limit ==200:
            break
        limit+=1
        temp=copy.deepcopy(grid)
        temp[i[0]][i[1]]=10
        data_list.append(temp)

fig=plt.figure()
anim =animation.FuncAnimation(fig,animate,init_func=init(len(grid[0])),frames=len(data_list))
pillow=animation.PillowWriter(fps=7)
anim.save("teste.gif",writer=pillow)
