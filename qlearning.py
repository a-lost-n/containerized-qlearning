import numpy as np
import gymnasium as gym
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
        
    def copy(self):
        return Environment(self.grid_size, self.qtable, self.environment)
    
    def train(self, episodes, alpha=0.5, gamma=0.9, epsilon=1.0, epsilon_decay=0.9999):
        self.outcomes = []
        for _ in range(episodes):
            state = self.environment.reset()[0]
            truncated = False

            self.outcomes.append("Failure")

            while not truncated:
                rnd = np.random.random()

                if rnd < epsilon or np.argmax(self.qtable[state]) == 0:
                    action = self.environment.action_space.sample()

                else:
                    action = np.argmax(self.qtable[state])
                        

                new_state, reward, terminated, truncated, _ = self.environment.step(action)

                self.qtable[state][action] = self.qtable[state][action] +\
                      alpha * (reward + gamma * np.max(self.qtable[new_state]) - self.qtable[state][action])

                state = new_state

                if terminated:
                    if reward:
                        self.outcomes[-1] = "Success"
                    break
                    
            epsilon = max(epsilon*epsilon_decay, 0.001)

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
    
    def last_n_sucesses(self, n):
        last_n_success = []
        last_n = []
        successes = 0
        for outcome in self.outcomes:
            last_n.append(outcome)
            if outcome == 'Success': successes += 1
            if len(last_n) > n:
                if last_n.pop(0) == 'Success': successes -= 1
            last_n_success.append(successes/len(last_n))
        return last_n_success
