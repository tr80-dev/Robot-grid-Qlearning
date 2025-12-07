
import numpy as np

# cells = value
EMPTY = 0
OBSTACLE = 1
GOAL = 2
START = 3

class GridEnv:
    def __init__(self, grid, start=None, max_steps=200):
        """
        map is grid 2D numpy array with values EMPTY/OBSTACLE/GOAL
        start stored as tuple and if none (0,0)

        """
        self.grid = grid.copy()
        self.n, self.m = grid.shape
        # find goal
        goal_pos = np.argwhere(grid == GOAL)
        if goal_pos.size == 0:
            raise ValueError("Grid must contain a GOAL cell (value 2).")
        self.goal = tuple(goal_pos[0])
        # start
        if start is None:
            s_pos = np.argwhere(grid == START)
            if s_pos.size > 0:
                self.start = tuple(s_pos[0])
            else:
                self.start = (0,0)
        else:
            self.start = start
        self.max_steps = max_steps
        self.reset()

    def reset(self, random_start=False):
        if random_start:
            empties = np.argwhere(self.grid == EMPTY)
            if empties.size == 0:
                self.pos = self.start
            else:
                idx = np.random.choice(len(empties))
                self.pos = tuple(empties[idx])
        else:
            self.pos = self.start
        self.steps = 0
        return self._state()

    def _state(self):
        # encode state as integer index: r*m + c
        return self.pos[0]*self.m + self.pos[1]

    def step(self, action):
        """
        action: 0=up,1=down,2=left,3=right
        returns: next_state (int), reward (float), done (bool), info (dict)
        """
        r, c = self.pos
        if action == 0:
            nr, nc = r-1, c
        elif action == 1:
            nr, nc = r+1, c
        elif action == 2:
            nr, nc = r, c-1
        elif action == 3:
            nr, nc = r, c+1
        else:
            raise ValueError("Invalid action")

        self.steps += 1
        # check bounds
        if not (0 <= nr < self.n and 0 <= nc < self.m):
            # hit wall -> stay
            nr, nc = r, c
            reward = -0.5
            done = False
        elif self.grid[nr, nc] == OBSTACLE:
            # obstacle -> stay
            nr, nc = r, c
            reward = -0.5
            done = False
        elif self.grid[nr, nc] == GOAL:
            self.pos = (nr, nc)
            reward = 1.0
            done = True
        else:
            # normal move
            self.pos = (nr, nc)
            reward = -0.01
            done = False

        if self.steps >= self.max_steps:
            done = True

        return self._state(), reward, done, {}

    def render(self):
        g = self.grid.copy()
        r, c = self.pos
        disp = np.array(g, dtype=object)
        for i in range(self.n):
            for j in range(self.m):
                if g[i,j] == EMPTY:
                    disp[i,j] = '.'
                elif g[i,j] == OBSTACLE:
                    disp[i,j] = '#'
                elif g[i,j] == GOAL:
                    disp[i,j] = 'G'
                elif g[i,j] == START:
                    disp[i,j] = 'S'
        disp[r,c] = 'R'  # robot
        print("\n".join("".join(row) for row in disp))
        print()
