import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

OBS_SPACE = 12
ACTION_SPACE = 3
LEARNING_RATE = 1e-3

class RLAgent(nn.Module):
    def __init__(self, model_folder, **kwargs):
        super(RLAgent, self).__init__()

        self.model_folder = model_folder

        # NN model
        self.linear1 = nn.Linear(OBS_SPACE, 128)
        self.linear2 = nn.Linear(128, 64)
        self.critic_linear3 = nn.Linear(64, 1)
        self.actor_linear3 = nn.Linear(64, ACTION_SPACE)

        self.ac_optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)

        # Load or save intial weigths of NN and optimizer
        if os.path.exists(model_folder+"model_weights/current.pt"):
            self.load_weigths()
        else:
            self.save_initial_weigths()

    def forward(self, state):
        state_torch = Variable(torch.from_numpy(state).float().unsqueeze(0))
        out_linear1 = F.relu(self.linear1(state_torch))
        out_linear2 = F.relu(self.linear2(out_linear1))
        value = self.critic_linear3(out_linear2)
        policy_dist = F.softmax(self.actor_linear3(out_linear2), dim=1)
        return value, policy_dist
    
    def load_weigths(self):
        model_params = torch.load(self.model_folder+"model_weights/current.pt")
        self.load_state_dict(model_params['model_state_dict'])
        self.ac_optimizer.load_state_dict(model_params['optimizer_state_dict'])
        max_reward = model_params['max_reward']

    def save_initial_weigths(self):
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.ac_optimizer.state_dict(),
            'max_reward': -1e8,
            }, self.model_folder+"model_weights/current.pt")

    def update_weights(self, model_path, max_reward):
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.ac_optimizer.state_dict(),
            'max_reward': max_reward,
            }, self.model_folder+model_path)

    def get_max_reward(self):
        model_params = torch.load(self.model_folder+"model_weights/current.pt")
        return model_params['max_reward']