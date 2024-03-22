import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point

import torch.nn as nn           #neural net
import torch.optim as optim     #optimiser
from model import Linear_QNet   #imports linear_qnet function from model 
from model import train_model   #imports train model 

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    
    def __init__(self, input_size, hidden_size, output_size):
        self.model = Linear_QNet(input_size, hidden_size, output_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.criterion = nn.MSELoss()
        self.n_games = 0
        self.epsilon = 1.0          # Starting exploration rate
        self.epsilon_decay = 0.995  # exploration decline
        self.epsilon_min = 0.01     # minimum exploration
        self.gamma = 0 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.high_score = 0
        self.batch_size = BATCH_SIZE
        
        # TODO: model, trainer
 
    def get_state(self, game):
        
        state = [ 
            """ 
            ## example state 
            # Food location
            1 if food_is_ahead else 0,
            0 if food_is_to_the_right else 1,
            0 if food_is_to_the_left else 1,
            """
        ]
        # Convert the state list into a PyTorch tensor.
        return torch.tensor(state, dtype=torch.float).unsqueeze(0)  # Add batch dimension at 0, so batch size is 1 and can be processed by the model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) #adds to memory for future use

    
    def get_action(self, state):
        # self.epsilon = 80 - self.n_games  # Decrease epsilon with the number of games
        if random.randint(0, 200) < self.epsilon:
            return random.randint(0, '2') # 0 to total amount of actions to pick, chooses one random action
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model.predict(state0) 
            return torch.argmax(prediction).item() # calls the predict funtion in the model to return the states best(reward) action
    
    def train_short_memory(self, state, action, reward, next_state, done): # t
        self.train_model(self.model, self.optimizer, self.criterion, state, next_state, reward, action, done)

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # List of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.train_model(self.model, self.optimizer, self.criterion, states, next_states, rewards, actions, dones)


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                # agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            # TODO: plot

if _name_ == '_main_':
    train()