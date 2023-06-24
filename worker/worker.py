import numpy as np
import gymnasium as gym
import json
from flask import Flask, request, jsonify

app = Flask(__name__)

class Worker():

    def __init__(self, qtable, map):
        self.qtable = qtable
        self.environment = gym.make('FrozenLake-v1',
                                    desc=map,
                                    is_slippery=False,
                                    render_mode='ansi')
    
    def train(self, episodes, alpha=0.5, gamma=0.9, epsilon=1.0, epsilon_decay=0.01):

        for _ in range(episodes):
            state = self.environment.reset()[0]
            truncated = False

            while not truncated:

                if np.random.random() < epsilon or np.argmax(self.qtable[state]) == 0:
                    action = self.environment.action_space.sample()
                else:
                    action = np.argmax(self.qtable[state])
                        
                new_state, reward, terminated, truncated, _ = self.environment.step(action)

                self.qtable[state][action] = self.qtable[state][action] +\
                      alpha * (reward + gamma * np.max(self.qtable[new_state]) - self.qtable[state][action])

                state = new_state

                if terminated:
                    break
                    
            epsilon = max(epsilon - epsilon_decay, 0)
    
    def export(self):
        return {'qtable': self.qtable.tolist()}

@app.route('/train', methods=['POST'])
def train_batch():
    qtable = np.array(request.json.get('qtable'))
    map = request.json.get('map')
    episodes = request.json.get('episodes')
    worker = Worker(qtable, map)
    worker.train(episodes)
    return jsonify(worker.export())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)