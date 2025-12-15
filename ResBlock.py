import torch.nn.functional as F
from torch import nn

class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.fc1 = nn.Linear(num_hidden, num_hidden)        
        self.bn1 = nn.BatchNorm1d(num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.bn2 = nn.BatchNorm1d(num_hidden)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.bn2(self.fc2(x))
        x += residual
        x = F.relu(x)
        return x