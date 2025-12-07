
import numpy as np

# change the grid downthere to create new map :)
# the robot can't move diagonal!
# start = 3 , goal = 2 , obstacle = 1 , empty = 0
grid = np.array([
    [3,0,0,0,0],
    [0,1,1,0,0],
    [0,0,0,0,1],
    [1,1,1,0,0],
    [0,0,0,1,2]
])

np.save("data/maps/map0.npy", grid)
