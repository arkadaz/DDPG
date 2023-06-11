import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir="tmp/ddpg"):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + "_ddpg")

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        self.action_value = nn.Linear(self.n_actions, self.fc2_dims)

        self.q = nn.Linear(self.fc2_dims, 1)

        f1 = 1.0/np.sqrt(self.fc1.weight.data.size())[0]
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)

        f2 = 1.0/np.sqrt(self.fc2.weight.data.size())[0]
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)

        f3 = 0.003
        self.q.weight.data.uniform_(-f3, f3)
        self.q.bias.data.uniform_(-f3, f3)

        f4 = 1.0/np.sqrt(self.action_value.weight.data.size())[0]
        self.action_value.weight.data.uniform_(-f4, f4)
        self.action_value.bias.data.uniform_(-f4, f4)

        self.optimizer = optim.AdamW(self.parameters(), lr=beta, weight_decay=0.01)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        print(f"device : {self.device}")


    def forward(self, state, action):
        state_value = self.bn2(self.fc2(F.relu(self.bn1(self.fc1(state)))))
        action_value = self.action_value(action)
        state_action_value = self.q(F.relu(torch.add(state_value, action_value)))
        return state_action_value
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir="tmp/ddpg"):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + "_ddpg")

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        self.mu = nn.Linear(self.fc2_dims, self.n_actions)

        f1 = 1.0/np.sqrt(self.fc1.weight.data.size())[0]
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)

        f2 = 1.0/np.sqrt(self.fc2.weight.data.size())[0]
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)

        f3 = 0.003
        self.mu.weight.data.uniform_(-f3, f3)
        self.mu.bias.data.uniform_(-f3, f3)

        self.optimizer = optim.AdamW(self.parameters(), lr=alpha, weight_decay=0.01)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        print(f"device : {self.device}")


    def forward(self, state):
        return torch.tanh(self.mu(F.relu(self.bn2(self.fc2(F.relu(self.bn1(self.fc1(state))))))))
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))