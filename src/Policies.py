import torch
from torch import nn
from torch.nn import functional as F
from Networks import ResidualBlock, ResidualBlockLarge

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

class GaussianGradientPolicy(nn.Module):
    def __init__(self, 
        in_features : int, 
        out_features : int, 
        hidden_size : int,
        log_std_min: float = -20,
        log_std_max: float = 2,
        min_std_value: float = 1e-3,  # New parameter
        device : torch.device = torch.device("cpu")):
        super().__init__()

        # Shared Network
        self.shared_net = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
        )

        self.mean = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_size // 2, out_features),
        )

        self.log_std = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_size // 2, out_features),
        )

        self.apply(self.init_weights) 
        self.eps = 1e-8
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.min_std_value = min_std_value
        self.device = device
        self.to(self.device)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, state):
        shared_features = self.shared_net(state.to(self.device))
        means = torch.tanh(self.mean(shared_features)) 
        # log_stds = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (torch.tanh(self.log_std(shared_features)) + 1.0)
        # stds = torch.exp(log_stds)
        # stds = torch.clamp(stds, min=self.min_std_value)  # Clamping std values

        stds = F.softplus(self.log_std(shared_features)) + .001
        return means, stds

class GaussianGradientPolicyV3(nn.Module):
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 hidden_size: int,
                 device: torch.device = torch.device("cpu")):
        super().__init__()

        # Shared Network
        self.shared_net = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        self.residual_net = nn.Sequential(
            nn.Linear(in_features, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 2),
            nn.ReLU(),
        )

        self.mean = nn.Linear(hidden_size + (hidden_size // 2), out_features)
        self.std = nn.Linear(hidden_size + (hidden_size // 2), out_features)

        self.apply(self.init_weights)  # Xavier initialization
        self.eps = 1e-8
        self.std_min = 0.1
        self.std_max = 20.0
        self.device = device
        self.to(self.device)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, state):
        shared_features = self.shared_net(state.to(self.device))
        residual_features = self.residual_net(state.to(self.device))

        combined_features = torch.cat((shared_features, residual_features), dim=-1)

        means = torch.tanh(self.mean(combined_features))
        stds = F.softplus(self.std(combined_features)) + self.eps
        stds = torch.clamp(stds, min=self.std_min) #, max=self.std_max)

        return means, stds
  


class GaussianGradientPolicyV4(nn.Module):
    def __init__(self, 
        in_features : int, 
        out_features : int, 
        hidden_size : int,
        device : torch.device = torch.device("cpu")):
        super().__init__()

        # Shared Network
        self.shared_net = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(),
            ResidualBlockLarge(hidden_size),
            ResidualBlockLarge(hidden_size),
        )
        self.mean = nn.Linear(hidden_size, out_features)
        self.std = nn.Linear(hidden_size, out_features)

        self.apply(self.init_weights)  # Xavier initialization
        self.eps = 1e-8
        self.device = device
        self.to(self.device)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, state):
        shared_features = self.shared_net(state)
        means = torch.tanh(self.mean(shared_features)) 
        stds = F.softplus(self.std(shared_features)) + self.eps
        return means, stds
    
class GaussianGradientPolicyV5(nn.Module):
    def __init__(self, 
        in_features : int, 
        out_features : int, 
        hidden_size : int,
        device : torch.device = torch.device("cpu")):
        super().__init__()

        # Shared Network
        self.shared_net = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(),
            ResidualBlock(hidden_size)
        )
        self.mean = nn.Linear(hidden_size, out_features)
        self.std = nn.Linear(hidden_size, out_features)

        self.apply(self.init_weights)
        self.eps = 1e-8
        self.device = device
        self.to(self.device)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, state):
        shared_features = self.shared_net(state)
        means = torch.tanh(self.mean(shared_features)) 
        stds = F.softplus(self.std(shared_features)) + self.eps
        return means, stds
        
        