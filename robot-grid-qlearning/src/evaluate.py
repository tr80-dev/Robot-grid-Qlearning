
import numpy as np
import os
from env import GridEnv
from agent import QAgent

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
MAP_DIR = os.path.join(DATA_DIR, 'maps')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
QPATH = os.path.join(MODEL_DIR, 'qtable.npy')

def load_map(name='map1.npy'):
    return np.load(os.path.join(MAP_DIR, name))

def evaluate(mapname='map1.npy', episodes=100, render=False):
    grid = load_map(mapname)
    env = GridEnv(grid)
    agent = QAgent(env.n*env.m)
    agent.load(QPATH)
    agent.epsilon = 0.0  # greedy

    successes = 0
    steps_list = []
    for ep in range(episodes):
        s = env.reset(random_start=False)
        done = False
        steps = 0
        while not done and steps < env.max_steps:
            a = agent.act(s)
            s2, r, done, _ = env.step(a)
            s = s2
            steps += 1
            if render:
                env.render()
        if env.pos == env.goal:
            successes += 1
            steps_list.append(steps)
    succ_rate = successes / episodes
    avg_steps = np.mean(steps_list) if steps_list else None
    print(f"Success rate: {succ_rate:.3f}, Avg steps (successful episodes): {avg_steps}")
    return succ_rate, avg_steps

if __name__ == "__main__":
    evaluate('map1.npy', episodes=50, render=False)
