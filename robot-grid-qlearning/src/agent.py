
import numpy as np
import os

class QAgent:
    def __init__(self, n_states, n_actions=4, alpha=0.5, gamma=0.99, epsilon=1.0, eps_decay=0.995, eps_min=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.Q = np.zeros((n_states, n_actions), dtype=float)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.eps_min = eps_min

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return int(np.argmax(self.Q[state]))

    def learn(self, s, a, r, s_next, done):
        q = self.Q[s, a]
        if done:
            target = r
        else:
            target = r + self.gamma * np.max(self.Q[s_next])
        self.Q[s, a] += self.alpha * (target - q)

    def decay_epsilon(self):
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

    def save(self, path):
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        np.save(path, self.Q)

    def load(self, path):
        self.Q = np.load(path)
