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

class GaussianGradientPolicy(nn.Module):
    def __init__(self, 
        in_features : int, 
        out_features : int, 
        hidden_size : int,
        device : torch.device = torch.device("cpu")):
        super().__init__()

        self.shared_net = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.LeakyReLU(),
        )

        self.mean = nn.Sequential(
            nn.Linear(hidden_size, out_features),
            nn.Tanh()
        )

        self.log_std = nn.Sequential(
            nn.Linear(hidden_size, out_features),
            nn.Softplus()
        )

        self.apply(self.init_weights) 
        self.eps = 1e-8
        self.min_std_value = 1e-5
        self.device = device
        self.to(self.device)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight, a=0.01)
            # m.weight.data *= 0.1
            m.bias.data.fill_(0.0)

    def forward(self, state):
        shared_features = self.shared_net(state.to(self.device))
        means = self.mean(shared_features)
        stds = self.log_std(shared_features) + 0.001
        # stds = torch.clamp(stds, min=self.min_std_value)
        return means, stds