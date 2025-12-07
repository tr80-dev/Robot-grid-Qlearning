
import numpy as np
import matplotlib.pyplot as plt
import time

#Loading the map and the qtable
MAP_FILE = "data/maps/map0.npy"
QTABLE_FILE = "models/qtable.npy"

grid = np.load(MAP_FILE)
qtable = np.load(QTABLE_FILE)

#specify the start and the end
start_pos = tuple(np.argwhere(grid==3)[0])
goal_pos  = tuple(np.argwhere(grid==2)[0])
robot_pos = start_pos

#give colors to the start , obstacle .....
colors = {0:(1,1,1), 1:(0,0,0), 2:(0,1,0), 3:(0,0,1)}

def plot_grid_step(grid, robot_pos, step_num):
    n,m = grid.shape
    img = np.zeros((n,m,3))
    for i in range(n):
        for j in range(m):
            img[i,j] = colors[int(grid[i,j])]
    img[robot_pos] = (1,0,0)
    plt.figure(figsize=(4,4))
    plt.imshow(img)
    plt.title(f"Step {step_num}")
    plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
    plt.show()

# ----------------------------
# the steps with images
# ----------------------------
print("Legend: Robot=Red | Start=Blue | Goal=Green | Obstacle=Black\n")

step_counter = 1
visited = {}   # mark visited position 
MAX_REPEAT = 30   # To avoid loop when there is no way :( 

while robot_pos != goal_pos:
    old_pos = robot_pos

    visited[robot_pos] = visited.get(robot_pos, 0) + 1
    if visited[robot_pos] > MAX_REPEAT:
        print("Robot stuck! (repeating same position)")
        break

    state_idx = robot_pos[0]*grid.shape[1] + robot_pos[1]
    action = np.argmax(qtable[state_idx])

    x,y = robot_pos

    if action==0 and x>0 and grid[x-1,y]!=1:
        robot_pos = (x-1,y)
    elif action==1 and x<grid.shape[0]-1 and grid[x+1,y]!=1:
        robot_pos = (x+1,y)
    elif action==2 and y>0 and grid[x,y-1]!=1:
        robot_pos = (x,y-1)
    elif action==3 and y<grid.shape[1]-1 and grid[x,y+1]!=1:
        robot_pos = (x,y+1)

    if robot_pos == old_pos:
        print("Robot stuck! (no valid movement)")
        break

    print(f"Step {step_counter}")
    plot_grid_step(grid, robot_pos, step_counter)
    step_counter += 1

if robot_pos == goal_pos:
    print("Goal reached!")
