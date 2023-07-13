import numpy as np
import gymnasium as gym
import json
from flask import Flask, request, jsonify

app = Flask(__name__)

class Worker():

    def set_map(self, map):
        self.environment = gym.make('FrozenLake-v1',
                                    desc=map,
                                    is_slippery=False,
                                    render_mode='ansi')
        
    def set_qtable(self, qtable):
        self.qtable = qtable
    
    def train(self, episodes, alpha=0.5, gamma=0.9, epsilon=0, epsilon_decay=0.999):

        for _ in range(episodes):
            state = self.environment.reset()[0]
            truncated = False
            terminated = False
            while not truncated and not terminated:

                if np.random.random() < epsilon or np.argmax(self.qtable[state]) == 0:
                    action = self.environment.action_space.sample()
                else:
                    action = np.argmax(self.qtable[state])
                        
                new_state, reward, terminated, truncated, _ = self.environment.step(action)

                self.qtable[state][action] = self.qtable[state][action] +\
                      alpha * (reward + gamma * np.max(self.qtable[new_state]) - self.qtable[state][action])

                state = new_state
                
                if epsilon != 0:
                    epsilon = max(epsilon*epsilon_decay, 0.001)
    
    def export(self):
        return {'qtable': self.qtable.tolist()}

worker = Worker()

@app.route('/map', methods=['POST'])
def build_worker():
    global worker
    map = request.json.get('map')
    worker.set_map(map)
    return jsonify({'success': True})

@app.route('/train', methods=['POST'])
def train_batch():
    global worker
    qtable = np.array(request.json.get('qtable'))
    episodes = request.json.get('episodes')
    epsilon = request.json.get('epsilon')
    epsilon_decay = request.json.get('epsilon_decay')
    worker.set_qtable(qtable)
    worker.train(episodes, epsilon=epsilon, epsilon_decay=epsilon_decay)
    return jsonify(worker.export())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)