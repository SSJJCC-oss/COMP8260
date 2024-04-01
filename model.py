import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

class DeepQNetwork(nn.Module):
    # Initializes the deep neural network with input, hidden and output layer sizes
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # Defining two linear layers with the first one having the input size and the hidden size 
        # and the second one with the hidden size and the output size
        self.linearinput = nn.Linear(input_size, hidden_size)
        self.linearoutput = nn.Linear(hidden_size, output_size)

    # Allowing the input data to flow through the network's layers
    def forward(self, x):
        x = F.relu(self.linearinput(x)) # Using the activation function Rectified Linear Unit on the linear input
        x = self.linearoutput(x) # Applying the linear transformation to the output
        return x

    # Saving the model. Helps saving the trained model for further training another time 
    def save_model(self, file_name='model.pth'):
        model_path = './saved_models'
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        file_name = os.path.join(model_path, file_name)
        torch.save(self.state_dict(), file_name)


class QLearningTrainer:
    # Is responsible in training a Q-Learning model
    def __init__(self, model, lr, discount):
        # Initializing different variables in storing the learning rate, discount factor and the model
        # and then calling the optimizer Adam and initializing a mean squared error loss function
        self.lr = lr
        self.discount = discount
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    # Performing a training step
    def train_step(self, state, action, reward, next_state, done):
        # Converting the following parameters to tensor state to be processed by the neural network model
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        # Converting the tensor from one-dimensional to two-dimensional by unsqueeze function
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        predict = self.model(state)

        target = predict.clone() # target value for Q Learning update
        for i in range(len(done)):
            new_Q_value = reward[i]
            if not done[i]:
                new_Q_value = reward[i] + self.discount * torch.max(self.model(next_state[i]))

            target[i][torch.argmax(action[i]).item()] = new_Q_value # Updates the Q-value of the action taken in the current state
    
        self.optimizer.zero_grad()
        loss = self.criterion(target, predict) # Loss between the target Q-values and the current Q-values
        loss.backward()

        self.optimizer.step()