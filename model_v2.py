import torch.nn as nn
import torch.optim as optim #optimization
import torch.nn.functional as F #activation function
import os

 class Linear_QNet(nn.Module):
     #initialises 3 layers
     def __init__(self, input_size, hidden_size, output_size):
         super().__init__()
         self.linear1 = nn.Linear(input_size, hidden_size)
         self.linear2 = nn.Linear(hidden_size, output_size)

     def forward(self, x):
         x = F.relu(self.linear1(x)) #relu activation on the first layer
         x = self.linear2(x) 
         return x #returns the output of the second layer

     def save(self, file_name='model.pth'): #saving the file
         model_folder_path = './model'
         if not os.path.exists(model_folder_path):
             os.makedirs(model_folder_path)

         file_name = os.path.join(model_folder_path, file_name)
         torch.save(self.state_dict(), file_name)
    #prediction function in the agent

 class QTrainer:
     def __init__(self, model, lr, gamma):
         self.lr = lr
         self.gamma = gamma
         self.model = model
         self.optimizer = optim.Adam(model.parameters(), lr=self.lr) #optimizer Adam
         self.criterion = nn.MSELoss() #Mean Square Error

     def train_step(self, state, action, reward, next_state, done):
         state = torch.tensor(state, dtype=torch.float)
         next_state = torch.tensor(next_state, dtype=torch.float)
         action = torch.tensor(action, dtype=torch.long)
         reward = torch.tensor(reward, dtype=torch.float)
         
         if done: 
            target_reward = reward #game complete
         else:
             with torch.no_grad():
                target_reward = reward + 0.99 * torch.max(model.predict(next_state)) # else is predicted reward, discount factor: 0.99


       
