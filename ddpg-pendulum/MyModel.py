import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super().__init__()
        first_hidden_dims = 512
        n = 2
        self.fc_bns = nn.ModuleList()
        for i in range(n):
            if i == 0:
                in_dims = state_size
                out_dims = first_hidden_dims
            else:
                in_dims = first_hidden_dims // (2**(i - 1))
                out_dims = first_hidden_dims // (2**(i))
            fc = nn.Linear(in_dims, out_dims)
            bn = nn.BatchNorm1d(out_dims)
            fc_bn = nn.ModuleList([fc, bn])
            self.fc_bns.append(fc_bn)
        self.fc = nn.Linear(first_hidden_dims // 2**(n - 1), action_size)
        self._reset_params()

    def _reset_params(self):
        for fc, _ in self.fc_bns:
            nn.init.kaiming_normal_(fc.weight)
            nn.init.constant_(fc.bias, 0)
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, state):
        x = state
        for fc, bn in self.fc_bns:
            x = fc(x)
            x = bn(x)
            x = F.relu(x)
        return 2 * torch.tanh(self.fc(x))


class Critic(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super().__init__()
        first_hidden_dims = 512
        n = 2
        self.fc_bns = nn.ModuleList()
        for i in range(n):
            if i == 0:
                in_dims = state_size + action_size
                out_dims = first_hidden_dims
            else:
                in_dims = first_hidden_dims // (2**(i - 1))
                out_dims = first_hidden_dims // (2**i)
            fc = nn.Linear(in_dims, out_dims)
            bn = nn.BatchNorm1d(out_dims)
            fc_bn = nn.ModuleList([fc, bn])
            self.fc_bns.append(fc_bn)
        self.fc = nn.Linear(first_hidden_dims // (2**(n - 1)), 1)
        self._reset_params()

    def _reset_params(self):
        for fc, _ in self.fc_bns:
            nn.init.kaiming_normal_(fc.weight)
            nn.init.constant_(fc.bias, 0)
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
    
    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        for fc, bn in self.fc_bns:
            x = fc(x)
            x = bn(x)
            x = F.relu(x)
        return self.fc(x)