import numpy as np
import gymnasium as gym
import requests
import json
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

worker_list_url = "http://127.0.0.1:5000/pods"
response = requests.get(worker_list_url)
worker_list = response.json()

class Master():

    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.qtable = np.zeros((grid_size**2, 4))
        self.map = generate_random_map(size=grid_size)
        self.environment = gym.make('FrozenLake-v1',
                                    desc=self.map,
                                    is_slippery=False,
                                    render_mode='ansi')

    def export(self):
        return json.dumps({'qtable': self.qtable.tolist(),
                           'map': self.map})

    def train(self):
        for worker_url in worker_list_url:
            response = requests.post(worker_url, json=self.export())

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
