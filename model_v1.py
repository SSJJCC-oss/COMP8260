import torch
import torch.nn as nn           #neural net
import torch.optim as optim     #optimiser

class Linear_QNet(nn.Module): #our learning net used, using functions from nn.module
    def __init__(self, input_size, hidden_size, output_size): # sets the layer sizes for the model we will wnat to use
        super(Linear_QNet, self).__init__() #calls constructor to make our necessary initializations
        self.linear1 = nn.Linear(input_size, hidden_size)       # first layer, input
        self.relu = nn.ReLU()                                   # second layer, hidden
        self.linear2 = nn.Linear(hidden_size, output_size)      # third layer, output
        
    def forward(self, x):   # pushes 'x' through layers
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def predict(self, state): #pushes state through layers, returns output(prediction)
        with torch.no_grad():
            return self.forward(state)

"""
training model function 
model: The neural network model that you are training.
optimizer: The optimization algorithm used for updating the weights of the model.
criterion: The loss function used to evaluate the performance of the model.
state: The current state of the game.
next_state: The state of the game after taking an action.
reward: The reward received after taking an action.
action: The action taken by the agent.
done: A boolean indicating whether the game has ended.
"""

def train_model(model, optimizer, criterion, state, next_state, reward, action, done):
    #converts to workable numbers
    state = torch.tensor(state, dtype=torch.float) 
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.long)
    
    if done: 
        target_reward = reward #game complete
    else:
        with torch.no_grad():
            target_reward = reward + 0.99 * torch.max(model.predict(next_state)) # else is predicted reward, discount factor: 0.99

    #uses the model to predict the rewards for the current state for each action
    predicted_reward = model.predict(state)[torch.arange(0, action.size(0)), action]
    loss = criterion(predicted_reward, target_reward) #compares said reward to desired reward

    optimizer.zero_grad()   # clear gradient
    loss.backward()         # compute loss gradient
    optimizer.step()        # update optimiser  
    
# import torch.nn as nn
# import torch.optim as optim #optimization
# import torch.nn.functional as F #activation function
# import os

# class Linear_QNet(nn.Module):
#     #initialises 3 layers
#     def __init__(self, input_size, hidden_size, output_size):
#         super().__init__()
#         self.linear1 = nn.Linear(input_size, hidden_size)
#         self.linear2 = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         x = F.relu(self.linear1(x)) #relu activation on the first layer
#         x = self.linear2(x) 
#         return x #returns the output of the second layer

#     def save(self, file_name='model.pth'): #saving the file
#         model_folder_path = './model'
#         if not os.path.exists(model_folder_path):
#             os.makedirs(model_folder_path)

#         file_name = os.path.join(model_folder_path, file_name)
#         torch.save(self.state_dict(), file_name)


# class QTrainer:
#     def __init__(self, model, lr, gamma):
#         self.lr = lr
#         self.gamma = gamma
#         self.model = model
#         self.optimizer = optim.Adam(model.parameters(), lr=self.lr) #optimizer Adam
#         self.criterion = nn.MSELoss() #Mean Square Error

#     def train_step(self, state, action, reward, next_state, done):
#         state = torch.tensor(state, dtype=torch.float)
#         next_state = torch.tensor(next_state, dtype=torch.float)
#         action = torch.tensor(action, dtype=torch.long)
#         reward = torch.tensor(reward, dtype=torch.float)

       
