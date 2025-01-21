import torch
import random
import numpy as np
from env import SnakeGameAI
from env import Direction, Point
from collections import deque

MAX_MEMORY = 100_000
BATCH = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0    # exploration/exploitation factor
        self.gamma = 0      # discount factor
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = None
        self.trainer = None

    def get_state(self, env):
        head = env.body[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = env.direction == Direction.LEFT
        dir_r = env.direction == Direction.RIGHT
        dir_u = env.direction == Direction.UP
        dir_d = env.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and env.is_collision(point_r)) or 
            (dir_l and env.is_collision(point_l)) or 
            (dir_u and env.is_collision(point_u)) or 
            (dir_d and env.is_collision(point_d)),

            # Danger right
            (dir_u and env.is_collision(point_r)) or 
            (dir_d and env.is_collision(point_l)) or 
            (dir_l and env.is_collision(point_u)) or 
            (dir_r and env.is_collision(point_d)),

            # Danger left
            (dir_d and env.is_collision(point_r)) or 
            (dir_u and env.is_collision(point_l)) or 
            (dir_r and env.is_collision(point_u)) or 
            (dir_l and env.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            env.food.x < env.head.x,  # food left
            env.food.x > env.head.x,  # food right
            env.food.y < env.head.y,  # food up
            env.food.y > env.head.y  # food down
            ]
        
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        action = [0,0,0]

        if random.randint(0,200) < self.epsilon:
            move = random.randint(0,2)
            action[move] = 1
        else:
            state = torch.tensor(state, dtype=torch.float)
            pred = self.model.predict(state)
            move = torch.argmax(pred).item()
            action[move] = 1
        
        return action


    def train_long_memory(self):
        if len(self.memory) > BATCH:
            mini_sample  = random.sample(self.memory, BATCH)
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memeory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    best_score = 0
    agent = Agent()
    env = SnakeGameAI()

    # training loop
    while True:
        # get agent state
        old_state = agent.get_state(env)

        # get action
        action = agent.get_action(old_state)

        # make action step
        reward, done, score = env.play_step(action)
        new_state = agent.get_state(env)

        # train short memory
        agent.train_short_memeory(old_state, action, reward, new_state, done)

        # rememeber all the params
        agent.remember(old_state, action, reward, new_state, done)

        # experience replay
        if done:
            env.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > best_score:
                best_score = score
                # TODO: agent.mode.save()

if __name__ == "__main__":
    train()

