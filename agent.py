import torch
import random
import numpy as np
from env import SnakeGameAI
from env import Direction, Point
from collections import deque
from model import QTrainer, DQN
from utils import plot
import argparse

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0    # exploration/exploitation factor
        self.gamma = 0.9    # discount factor
        self.max_memory = 100_000
        self.batch = 1000
        self.lr = 0.001
        self.memory = deque(maxlen=self.max_memory)

        self.model = DQN(11,256,3)
        self.trainer = QTrainer(self.model, lr=self.lr, gamma=self.gamma)

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

    def get_action(self, state, training = True):
        if training:
            self.epsilon = 80 - self.n_games
        else:
            self.epsilon = -1
        action = [0,0,0]

        if random.randint(0,200) < self.epsilon:
            move = random.randint(0,2)
            action[move] = 1
        else:
            state = torch.tensor(state, dtype=torch.float)
            pred = self.model(state)
            move = torch.argmax(pred).item()
            action[move] = 1
        
        return action


    def train_long_memory(self):
        if len(self.memory) > self.batch:
            mini_sample  = random.sample(self.memory, self.batch)
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memeory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

def train(episodes):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    best_score = 0
    agent = Agent()
    env = SnakeGameAI()

    # training loop
    for i in range(episodes):
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
                agent.model.save()

            plot_scores.append(score)
            total_score+=score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

def play(episodes):
    agent = Agent()
    env = SnakeGameAI()
    agent.model.load_state_dict((torch.load('model/model.pth', weights_only=True)))
    agent.model.eval() 
    
    for _ in range(episodes):
        high_score = 0
        done = False
        state = agent.get_state(env)
        while not done:
            # get action in inference
            action = agent.get_action(state, training = False)

            # make action step
            reward, done, score = env.play_step(action)
            state = agent.get_state(env)
        env.reset()
        high_score = max(score,high_score)
        print("High Score: ", high_score)
    

def main():
    parser = argparse.ArgumentParser(description="Train or play the snake game using a DQN model.")
    parser.add_argument('--train', action='store_true', help="Train the DQN model.")
    parser.add_argument('--episodes', type=int, help="Number of training/ inference episodes.", default=100)
    parser.add_argument('--play', action='store_true', help="Play the game using the trained DQN model.")
    args = parser.parse_args()
    
    if args.train:
        print("Training mode selected.")
        train(args.episodes)
    elif args.play:
        print("Play mode selected.")
        play(args.episodes)
    else:
        print("Please specify either --train or --play.")

if __name__ == "__main__":
    main()

