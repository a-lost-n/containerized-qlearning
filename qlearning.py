import numpy as np
import gymnasium as gym
import time
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

class Environment():

    def __init__(self, grid_size=None, filename=None):
        if filename is None and grid_size is not None:
            self.qtable = np.zeros((grid_size**2, 4))
            self.environment = gym.make('FrozenLake-v1',
                                        desc=generate_random_map(size=grid_size),
                                        is_slippery=False,
                                        render_mode='ansi')
        elif filename is not None and grid_size is None:
            data = np.load(filename)
            self.qtable = data['qtable']
            self.environment = gym.make('FrozenLake-v1',
                                        desc=data['map'].tolist(),
                                        is_slippery=False,
                                        render_mode='ansi')



    def load(self, file):
        self.qtable = np.load(file)
    
    def train(self, episodes, alpha=0.5, gamma=0.9, epsilon=1.0, epsilon_decay=0.99999):
        t1 = time.perf_counter()
        for _ in range(episodes):
            state = self.environment.reset()[0]
            truncated = False
            terminated = False
            while not truncated and not terminated:
                if np.argmax(self.qtable[state]) == 0:
                    action = self.environment.action_space.sample()
                else:
                    action = np.argmax(self.qtable[state])
                new_state, reward, terminated, truncated, _ = self.environment.step(action)
                self.qtable[state][action] = self.qtable[state][action] +\
                      alpha * (reward + gamma * np.max(self.qtable[new_state]) - self.qtable[state][action])
                state = new_state
            epsilon = max(epsilon * epsilon_decay, 0.001)
        t2 = time.perf_counter()
        print("Solution was found after {:.2f}s and {} episodes.".format(t2-t1,episodes))
        return t2-t1, episodes

    def test_run(self):
        state = self.environment.reset()[0]
        truncated = False
        while not truncated:
            action = np.argmax(self.qtable[state])

            new_state, reward, terminated, truncated, _ = self.environment.step(action)

            state = new_state

            if terminated:
                if reward:
                    return "Success"
        return "Inconclusive"
    
