import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_size, action_size, param_file=None):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, action_size)
        self._reset_params()
        if param_file:
            self.load_params(param_file)
        else:
            self._reset_params()

    def _reset_params(self):
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.constant_(self.fc3.bias, 0)

    def load_params(self, param_file):
        self.load_state_dict(torch.load(param_file))

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    def __init__(self, state_size, action_size, param_file=None):
        super().__init__()
        initial_hidden_units = 1024
        self.fc1 = nn.Linear(state_size, initial_hidden_units)
        self.bn1 = nn.BatchNorm1d(initial_hidden_units)
        self.fc2 = nn.Linear(initial_hidden_units + action_size, initial_hidden_units // 2)
        self.bn2 = nn.BatchNorm1d(initial_hidden_units // 2)
        self.fc3 = nn.Linear(initial_hidden_units // 2, initial_hidden_units // 4)
        self.bn3 = nn.BatchNorm1d(initial_hidden_units // 4)
        self.fc4 = nn.Linear(initial_hidden_units // 4, 1)
        if param_file:
            self.load_params(param_file)
        else:
            self._reset_params()

    def _reset_params(self):
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.kaiming_normal_(self.fc3.weight)
        nn.init.constant_(self.fc3.bias, 0)
        nn.init.xavier_normal_(self.fc4.weight)
        nn.init.constant_(self.fc4.bias, 0)
    
    def load_params(self, param_file):
        self.load_state_dict(torch.load(param_file))

    def forward(self, state, action):
        x = F.relu(self.bn1(self.fc1(state)))
        x = torch.cat((x, action), dim=1)
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        return self.fc4(x)