import torch
from torch import nn

class ResidualBlockLarge(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )

    def forward(self, x):
        return x + self.block(x)
    

class ResidualBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return x + self.block(x)


class ValueNetwork(nn.Module):
    def __init__(self, in_features : int, hidden_size : int, device : torch.device = torch.device("cpu")):
        super().__init__()
        
        self.value_net = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.apply(self.init_weights) 
        self.device = device
        self.to(self.device)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        x = self.value_net(x)
        return x


class ValueNetworkResidual(nn.Module):
    def __init__(self, in_features: int, hidden_size: int, device: torch.device = torch.device("cpu")):
        super().__init__()

        self.hidden_layers = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU()
        )

        self.residual_layers = nn.Sequential(
            nn.Linear(in_features, hidden_size // 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 2),
            nn.LeakyReLU(),
        )

        self.output_layer = nn.Linear(hidden_size + (hidden_size // 2), 1)

        self.apply(self.init_weights)
        self.device = device
        self.to(self.device)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        hidden_features = self.hidden_layers(x)
        residual_features = self.residual_layers(x)

        combined_features = torch.cat((hidden_features, residual_features), dim=-1)
        output = self.output_layer(combined_features)

        return output
    


class ValueNetworkV2(nn.Module):
    def __init__(self, in_features : int, hidden_size : int, device : torch.device = torch.device("cpu")):
        super().__init__()

        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(in_features, hidden_size)
        self.bn1 = nn.LayerNorm(hidden_size)
        self.relu = nn.LeakyReLU()
        self.residual1 = ResidualBlockLarge(hidden_size)
        self.residual2 = ResidualBlockLarge(hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

        self.apply(self.init_weights) 
        self.device = device
        self.to(self.device)
    
    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        batch_size, num_envs, _ = x.size()
        x = self.fc1(x.view(-1, x.size(-1)))  # Flatten last two dimensions
        x = x.view(batch_size, num_envs, -1)  # Reshape back to original size
        x = self.bn1(x)
        x = self.relu(x)
        x = x.view(-1, self.hidden_size)  # Flatten last two dimensions for Residual blocks and final Linear layer
        x = self.residual1(x)
        x = self.residual2(x)
        x = self.fc2(x)
        return x.view(batch_size, num_envs, 1)  # Reshape back to original size
    

class ValueNetworkv3(nn.Module):
    def __init__(self, in_features : int, hidden_size : int, device : torch.device = torch.device("cpu")):
        super().__init__()

        self.value_net = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(),
            ResidualBlock(hidden_size),
            nn.Linear(hidden_size, 1)
        )

        self.apply(self.init_weights) 
        self.device = device
        self.to(self.device)
    
    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        orig_shape = x.shape
        x = x.view(-1, orig_shape[-1])
        x = self.value_net(x)
        x = x.view(orig_shape[:-1])
        return x


