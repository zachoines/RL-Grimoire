import torch
from torch import nn

class ValueNetwork(nn.Module):
    def __init__(self, in_features : int, hidden_size : int, device : torch.device = torch.device("cpu")):
        super().__init__()
        
        self.value_net = nn.Sequential(
            nn.Linear(in_features, int(hidden_size / 2.0)),
            nn.LeakyReLU(),
            nn.Linear(int(hidden_size / 2.0), hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.device = device
        self.to(self.device)

    def forward(self, x):
        x = self.value_net(x)
        return x