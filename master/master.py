import numpy as np
import gymnasium as gym
import json
import asyncio
import time
import aiohttp
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
        self.epsilon = 1.0
        self.epsilon_decay = 0.9999

    def export(self, episodes=1000):
        return {'qtable': self.qtable.tolist(),
                'map': self.map,
                'episodes': episodes,
                'epsilon': self.epsilon}

    async def send_and_fetch_table(self, worker_ip, episodes_per_worker):
        async with aiohttp.ClientSession() as session:
            for worker_ip in worker_ips:
                url = f"http://{worker_ip}:5000/train"
                async with session.post(url, json=self.export(episodes=episodes_per_worker)) as response:
                    data = await response.json()
                    return data['qtable']

    async def send_and_fetch_all_tables(self, episodes_per_worker):
        qtables = await asyncio.gather(*[self.send_and_fetch_table(worker_ip, episodes_per_worker) for worker_ip in worker_ips])
        return qtables

    def train(self, episodes=1000000, episodes_per_worker=10000):
        for _ in range(0, episodes, episodes_per_worker*len(worker_ips)):
            qtables = asyncio.run(self.send_and_fetch_all_tables(episodes_per_worker))
            self.qtable = np.mean(np.array(qtables), axis=0)
            self.epsilon = max(self.epsilon * self.epsilon_decay**(episodes_per_worker*len(worker_ips)), 0.001)

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
    master.train(episodes=1000000)
    t2 = time.perf_counter()
    print(t2 - t1)
    print(master.test_run())
    print(master.qtable[0])
