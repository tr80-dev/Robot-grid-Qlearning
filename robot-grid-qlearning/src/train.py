
import numpy as np
import os
from env import GridEnv
from agent import QAgent

# hyperparameters 
EPISODES = 5000
MAX_STEPS = 200
ALPHA = 0.6
GAMMA = 0.99
EPS_START = 1.0
EPS_DECAY = 0.995
EPS_MIN = 0.01

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
MAP_DIR = os.path.join(DATA_DIR, 'maps')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)
QPATH = os.path.join(MODEL_DIR, 'qtable.npy')

def load_map(name='map1.npy'):
    p = os.path.join(MAP_DIR, name)
    return np.load(p)

def train_on_map(mapname='map1.npy'):
    grid = load_map(mapname)
    env = GridEnv(grid, max_steps=MAX_STEPS)
    n_states = env.n * env.m
    agent = QAgent(n_states, alpha=ALPHA, gamma=GAMMA,
                   epsilon=EPS_START, eps_decay=EPS_DECAY, eps_min=EPS_MIN)

    rewards = []
    success_count = 0

    for ep in range(1, EPISODES+1):
        s = env.reset(random_start=False)
        total_r = 0.0
        done = False
        for step in range(MAX_STEPS):
            a = agent.act(s)
            s2, r, done, _ = env.step(a)
            agent.learn(s, a, r, s2, done)
            s = s2
            total_r += r
            if done:
                # check if reached goal
                if r > 0:
                    success_count += 1
                break
        agent.decay_epsilon()
        rewards.append(total_r)

        if ep % 200 == 0:
            succ_rate = success_count / ep
            avg_r = np.mean(rewards[-200:])
            print(f"Ep {ep} | eps {agent.epsilon:.3f} | last_avg_r {avg_r:.3f} | succ_rate {succ_rate:.3f}")
            # optional: save intermediate Q
            agent.save(QPATH)

    # final save
    agent.save(QPATH)
    print("Training finished. Q saved to", QPATH)

if __name__ == "__main__":
    train_on_map('map0.npy')
