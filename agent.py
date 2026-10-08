import numpy as np


class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, seed=0):
        self.q = np.zeros((n_states, n_actions))
        self.alpha = alpha      # learning rate
        self.gamma = gamma      # discount factor
        self.rng = np.random.default_rng(seed)

    def act(self, state, epsilon):
        #Explore with probability epsilon, otherwise take the best known action.
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.q.shape[1]))
        return int(np.argmax(self.q[state]))

    def update(self, state, action, reward, next_state, terminal):
        """Q(s,a) += alpha * (r + gamma * max Q(s',.) - Q(s,a))"""
        target = reward if terminal else reward + self.gamma * np.max(self.q[next_state])
        self.q[state, action] += self.alpha * (target - self.q[state, action])