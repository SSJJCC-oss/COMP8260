import torch
import random
import numpy as np
# Deque is imported for memory management
from collections import deque
# Importing the classes from the files I had created (game.py, model.py and helper.py)
from game import SnakeGameAI, Direction, Point
from model import DeepQNetwork, QLearningTrainer
from helper import plot

MAX_MEMORY = 100_000
LEARNING_RATE = 0.001
BATCH = 1000

class Agent:

    def __init__(self):
        self.n_games = 0 # number of games
        self.epsilon = 0 # randomness in action selection
        self.discount = 0.9 # discount rate for future rewards
        self.memory = deque(maxlen=MAX_MEMORY) # memory to store experiences for training the model (state, action, reward, next_state, done)
        # Next two are instances of classes from the model
        self.model = DeepQNetwork(11, 256, 3)
        self.trainer = QLearningTrainer(self.model, lr=LEARNING_RATE, discount=self.discount)


    def get_state(self, game):
        # retrieves the coordinate of the snake's head and defines the four points surrounding the snake's head in four direction
        head = game.snake[0]
        left_point = Point(head.x - 20, head.y)
        right_point = Point(head.x + 20, head.y)
        up_point = Point(head.x, head.y - 20)
        down_point = Point(head.x, head.y + 20)
        
        # Boolean values determining the current direction of the snake
        left_direction = game.direction == Direction.LEFT
        right_direction = game.direction == Direction.RIGHT
        up_direction = game.direction == Direction.UP
        down_direction = game.direction == Direction.DOWN


        # Initializing the current state of the snake by constructing a list with the danger indicators, direction flags and food location
        state = [(right_direction and game.is_colliding(right_point)) or (left_direction and game.is_colliding(left_point)) or 
            (up_direction and game.is_colliding(up_point)) or (down_direction and game.is_colliding(down_point)),
            (up_direction and game.is_colliding(right_point)) or (down_direction and game.is_colliding(left_point)) or 
            (left_direction and game.is_colliding(up_point)) or (right_direction and game.is_colliding(down_point)),
            (down_direction and game.is_colliding(right_point)) or (up_direction and game.is_colliding(left_point)) or 
            (right_direction and game.is_colliding(up_point)) or (left_direction and game.is_colliding(down_point)),
            left_direction, right_direction, up_direction, down_direction,
            
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        # Returns the constructed state into NumPy array of integers
        return np.array(state, dtype=int)

    # Stores the current experience (state, action, reward, next_state, done) in the agent's memory buffer
    def remember_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # Trains the agent using experiences stored in the memory's buffer
    def train_memory_batch(self):
        # If the length of the memory greater than 1000, then only a sample of experiences is selected
        if len(self.memory) > BATCH:
            sample = random.sample(self.memory, BATCH)
        else:
            sample = self.memory

        # Unpacking the sample into lists for states, action, rewards, next_step and done and then performs a training step
        states, actions, rewards, next_states, dones = zip(*sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    # This function trains individual experiences
    def train_single_experience(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # Choosing the best known actions and then creating a list 
        self.epsilon = 80 - self.n_games
        deciding_move = [0,0,0]
        # If the random number between 0 and 200 is less than the epsilon, a random number between 0 and 2 is selected representing one of those three moves
        if random.randint(0, 200) < self.epsilon:
            action = random.randint(0, 2)
            deciding_move[action] = 1 # making it 1 as the chosen action
        else:
            # Converting the current state into tensor state with tensor float which helps in using the neural network
            state_tensor = torch.tensor(state, dtype=torch.float)
            # Predicts the Q values for each action in the current state and the action with the highest value is our final move
            prediction = self.model(state_tensor)
            action = torch.argmax(prediction).item()
            deciding_move[action] = 1

        return deciding_move


def train():
    score_list = []
    mean_score_list = []
    total_score = 0
    high_score = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        deciding_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(deciding_move)
        state_new = agent.get_state(game)

        # train the memory that takes in the single experience  
        agent.train_single_experience(state_old, deciding_move, reward, state_new, done)

        # remember the experience
        agent.remember_experience(state_old, deciding_move, reward, state_new, done)

        if done:
            # train the memory batch, plot result
            game.reset()
            agent.n_games += 1
            agent.train_memory_batch()

            # Saving the high score
            if score > high_score:
                high_score = score
                agent.model.save_model()

            print('Game', agent.n_games, 'Score', score, 'High score:', high_score)

            # Plotting the mean score
            score_list.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            mean_score_list.append(mean_score)
            plot(score_list, mean_score_list)


if __name__ == '__main__':
    train()