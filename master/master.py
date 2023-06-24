import numpy as np
import gymnasium as gym
import requests
import json
import time
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from kubernetes import client, config

# Load the in-cluster Kubernetes configuration
config.load_incluster_config()

# Create an instance of the Kubernetes API client
v1 = client.CoreV1Api()

# Retrieve the IP addresses of the worker pods
worker_ips = []
pods = v1.list_namespaced_pod(namespace="default", label_selector="app=worker")
for pod in pods.items:
    print(pod.status.pod_ip)
    worker_ips.append(pod.status.pod_ip)

class Master():

    def __init__(self, grid_size=6):
        self.grid_size = grid_size
        self.qtable = np.zeros((grid_size**2, 4))
        self.map = generate_random_map(size=grid_size)
        self.environment = gym.make('FrozenLake-v1',
                                    desc=self.map,
                                    is_slippery=False,
                                    render_mode='ansi')

    def export(self):
        return {'qtable': self.qtable.tolist(),
                           'map': self.map,
                           'episodes': 1000}

    def train(self):
        qtables = []
        for worker_ip in worker_ips:
            response = requests.post(f"http://{worker_ip}:5000/train",
                                     json=self.export())
            qtables.append(response.json()['qtable'])
        self.qtable = np.mean(np.array(qtables), axis=0)

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


if __name__ == '__main__':
    master = Master(10)
    t1 = time.perf_counter()
    master.train()
    t2 = time.perf_counter()
    print(t2 - t1)
    print(master.test_run())