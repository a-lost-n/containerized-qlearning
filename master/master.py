import numpy as np
import gymnasium as gym
import asyncio
import time
import sys
import aiohttp
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from kubernetes import client, config

# Load the in-cluster Kubernetes configuration
config.load_incluster_config()

# Create an instance of the Kubernetes API client
v1 = client.CoreV1Api()

# Retrieve the IP addresses of the worker pods
set_up = False
while not set_up:
    worker_ips = []
    pods = v1.list_namespaced_pod(namespace="default", label_selector="app=worker")
    set_up = True
    for pod in pods.items:
        if pod.status.pod_ip is None:
            set_up = False
            break
        worker_ips.append(pod.status.pod_ip)

class Master():

    def __init__(self, grid_size):
        self.qtable = np.zeros((grid_size**2, 4))
        self.map = generate_random_map(size=grid_size)
        self.environment = gym.make('FrozenLake-v1',
                                    desc=self.map,
                                    is_slippery=False,
                                    render_mode='ansi')
        self.epsilon = 1.0
        self.epsilon_decay = 0.99999

    def export(self, episodes=1000):
        return {'qtable': self.qtable.tolist(),
                'episodes': episodes,
                'epsilon': self.epsilon,
                'epsilon_decay': self.epsilon_decay}

    async def send_and_fetch_table(self, worker_ip, episodes_per_worker):
        async with aiohttp.ClientSession() as session:
            url = f"http://{worker_ip}:5000/train"
            # print("[STATUS] Job sent to {}".format(worker_ip))
            async with session.post(url, json=self.export(episodes=episodes_per_worker)) as response:
                data = await response.json()
                return data['qtable']

    async def send_and_fetch_all_tables(self, episodes_per_worker):
        qtables = await asyncio.gather(*[self.send_and_fetch_table(worker_ip, episodes_per_worker) for worker_ip in worker_ips])
        return qtables
    
    async def send_map(self, worker_ip):
        async with aiohttp.ClientSession() as session:
            url = f"http://{worker_ip}:5000/map"
            async with session.post(url, json={'map': self.map}) as response:
                data = await response.json()
                return data['success']
            
    async def send_all_maps(self):
        success = await asyncio.gather(*[self.send_map(worker_ip) for worker_ip in worker_ips])
        return False not in success

    def train(self, episodes=1000000, episodes_per_worker=10000):
        if not asyncio.run(self.send_all_maps()):
            print("Error while trying to send map")
            return
        t1 = time.perf_counter()
        for episodes_done in range(0, episodes, episodes_per_worker*len(worker_ips)):
            qtables = asyncio.run(self.send_and_fetch_all_tables(episodes_per_worker))
            self.qtable = np.mean(np.array(qtables), axis=0)
            self.epsilon = max(self.epsilon * self.epsilon_decay**(episodes_per_worker*len(worker_ips)), 0.001)
        t2 = time.perf_counter()
        print("Solution returned {} after {:.2f}s and {} episodes.".format(self.test_run(),t2-t1,episodes))
        return t2-t1, episodes

    def efficiency_test(self, episodes_per_worker):
        if not asyncio.run(self.send_all_maps()):
            print("Error while trying to send map")
            return
        t1 = time.perf_counter()
        episodes = 0
        while np.amax(self.qtable[0]) == 0:
            qtables = asyncio.run(self.send_and_fetch_all_tables(episodes_per_worker))
            self.qtable = np.mean(np.array(qtables), axis=0)
            episodes += episodes_per_worker*len(worker_ips)
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


if __name__ == '__main__':
    grid_size = int(sys.argv[1])
    episodes = int(sys.argv[2])
    episodes_per_worker = int(sys.argv[3])
    master = Master(grid_size=grid_size)
    master.train(episodes=episodes, episodes_per_worker=episodes_per_worker)
    np.savez("model/model.npz",qtable=master.qtable,map=master.map)
    # seconds_vec = []
    # episodes_vec = []
    # for _ in range(10):
    #     master = Master(10)
    #     seconds, episodes = master.train(episodes_per_worker=1000)
    #     seconds_vec.append(seconds)
    #     episodes_vec.append(episodes)
    # np.savez("measurements.npz", np.array(seconds_vec), np.array(episodes_vec))
