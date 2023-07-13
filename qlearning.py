import numpy as np
import gymnasium as gym
import time
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

class Environment():

    def __init__(self, grid_size=10, qtable = None, environment = None):
        self.grid_size = grid_size
        if qtable is None:
            self.qtable = np.zeros((grid_size**2, 4))
        else:
            self.qtable = qtable
        if environment is None:
            self.environment = gym.make('FrozenLake-v1',
                                        desc=generate_random_map(size=grid_size),
                                        is_slippery=False,
                                        render_mode='ansi')
        else:
            self.environment = environment

    
    def train(self, alpha=0.5, gamma=0.9):
        episodes = 0
        t1 = time.perf_counter()
        while np.amax(self.qtable[0]) == 0:
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
            episodes += 1
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
    
