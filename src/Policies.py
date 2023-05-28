import torch
from torch import nn
from torch.nn import functional as F

class DiscreteGradientPolicy(nn.Module):
    def __init__(self, in_features : int, out_features : int, hidden_size : int, device : torch.device = torch.device("cpu")):
        super().__init__()
        hidden_space1 = int(hidden_size / 2)
        hidden_space2 = hidden_size
        self.fc1 = nn.Linear(in_features, hidden_space1)
        self.fc2 = nn.Linear(hidden_space1, hidden_space2)
        self.fc3 = nn.Linear(hidden_space2, out_features)
        self.lrelu = nn.LeakyReLU()
        
        self.device = device
        self.to(self.device)

    def forward(self, x):
        x = self.lrelu(self.fc1(x))
        x = self.lrelu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=-1)
        return x

class GaussianGradientPolicyV2(nn.Module):
    def __init__(self, 
        in_features : int, 
        out_features : int, 
        hidden_size : int,
        device : torch.device = torch.device("cpu")):
        super().__init__()

        # Shared Network
        hidden_space1 = int(hidden_size / 2)
        hidden_space2 = hidden_size
        self.shared_net = nn.Sequential(
            nn.Linear(in_features, hidden_space1),
            nn.LeakyReLU(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.LeakyReLU(),
        )

        # Policy Mean
        self.mean = nn.Sequential(
            nn.Linear(hidden_space2, out_features),
            nn.Tanh()
        )

        # Policy Std Dev
        self.std = nn.Sequential(
            nn.Linear(hidden_space2, out_features),
            nn.Softplus()
        )

        self.eps = 0.0001
        self.device = device
        self.to(self.device)

    def forward(self, state):
        shared_features = self.shared_net(state)
        means = self.mean(shared_features)
        stds = self.std(shared_features) + self.eps
        return means, stds

class GaussianGradientPolicy(nn.Module):
    def __init__(self, 
        in_features : int, 
        out_features : int, 
        hidden_size : int,
        device : torch.device = torch.device("cpu")):
        super().__init__()

        # Shared Network
        hidden_space1 = int(hidden_size / 2)
        hidden_space2 = hidden_size
        self.shared_net = nn.Sequential(
            nn.Linear(in_features, hidden_space1),
            nn.Tanh(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.Tanh(),
        )

        # Policy Mean
        self.mean = nn.Sequential(
            nn.Linear(hidden_space2, out_features)
        )

        # Policy Std Dev
        self.std = nn.Sequential(
            nn.Linear(hidden_space2, out_features)
        )

        self.eps = 1e-6
        self.device = device
        self.to(self.device)

    def forward(self, state):
        shared_features = self.shared_net(state)
        means = self.mean(shared_features)
        stds = torch.log(
            1 + torch.exp(self.std(shared_features))
        )
        return means, stds
        
        