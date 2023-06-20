import torch
from torch import nn
from torch.nn import functional as F
from Networks import StatefulTransformer
from typing import Tuple

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
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, out_features),
            nn.Tanh()
        )

        self.std = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
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
            m.weight.data *= 0.1
            m.bias.data.fill_(0.0)

    def forward(self, state):
        shared_features = self.shared_net(state.to(self.device))
        means = self.mean(shared_features)
        stds = self.std(shared_features)
        stds = torch.clamp(stds, min=self.min_std_value)
        return means, stds
    

class GaussianGradientTransformerPolicy(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_size: int,
        stack_size: int = 16,
        num_layers: int = 8,
        nhead: int = 4,
        device: torch.device = torch.device("cpu")
    ):
        super().__init__()

        self.in_features = in_features
        self.stack_size = stack_size
        self.hidden_size = hidden_size
        self.state_size = self.in_features // self.stack_size

        # Embedding layer to map the state to an embedding dimension
        self.embedding = nn.Linear(self.state_size, hidden_size).to(device)

        # Shared transformer encoder network
        self.shared_net = nn.Transformer(
            d_model=hidden_size,
            nhead=nhead,
            num_encoder_layers=num_layers,
            batch_first=True
        ).to(device)

        # Mean network
        self.mean = nn.Sequential(
            nn.Linear(hidden_size, out_features),
            nn.Tanh()
        ).to(device)

        # Standard deviation network
        self.std = nn.Sequential(
            nn.Linear(hidden_size, out_features),
            nn.Softplus()
        ).to(device)

        self.apply(self.init_weights)
        self.eps = 1e-8
        self.min_std_value = 1e-5
        self.device = device

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight, a=0.01)
            m.bias.data.fill_(0.0)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the policy network.

        Args:
            state (torch.Tensor): Input state tensor of shape (num_envs, state_size * stack_size).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of mean and standard deviation tensors.
                - mean: Tensor of shape (num_envs, out_features).
                - std: Tensor of shape (num_envs, out_features).
        """

        if len(state.size()) == 2:
            state = state.unsqueeze(1)  # Add a singleton dimension for num_samples

        num_envs, num_samples, _ = state.size()

        # Reshape the state to (num_envs, stack_size, state_size)
        state = state.reshape((num_envs, num_samples, self.stack_size, self.state_size)).to(self.device)

        # Pass the state through the embedding layer
        embedded_state = F.leaky_relu(self.embedding(state))

        # Reshape the embedded state to (num_envs * num_samples, stack_size, hidden_size)
        embedded_state = embedded_state.reshape((num_envs * num_samples, self.stack_size, self.hidden_size))

        # Pass the embedded state through the shared transformer encoder network
        shared_features = self.shared_net(embedded_state, embedded_state)

        # Reshape back to (num_envs, num_samples, hidden_size) or (num_envs, hidden_size) in the single sample case
        shared_features = shared_features.reshape((num_envs, num_samples, self.stack_size, self.hidden_size)).to(self.device)

        # Apply mean pooling along the spatial dimensions
        pooled_features = torch.mean(shared_features, dim=2)

        # Reshape back to (num_envs, num_samples, hidden_size) or (num_envs, hidden_size) in the single sample case
        pooled_features = pooled_features.reshape((num_envs, num_samples, self.hidden_size))

        # Compute the mean and standard deviation
        means = self.mean(pooled_features)
        stds = self.std(pooled_features)
        stds = torch.clamp(stds, min=self.min_std_value)

        return means, stds

class GaussianGradientPolicyRecurrent(nn.Module):
    def __init__(
            self,
            in_features : int, 
            out_features : int, 
            hidden_size : int,
            device : torch.device = torch.device("cpu")
        ):
        super().__init__()

        self.mean = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, out_features),
            nn.Tanh()
        )

        self.std = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, out_features),
            nn.Softplus()
        )
        
        self.transformer = StatefulTransformer(in_features, hidden_size, hidden_size, device=device)

        self.apply(self.init_weights) 
        self.eps = 1e-8
        self.min_std_value = 1e-5
        self.device = device
        self.to(self.device)
        

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight, a=0.01)
            m.weight.data *= 0.1
            m.bias.data.fill_(0.0)

    def forward(self, x, dones=None):
        x = torch.unsqueeze(x, dim=1)
        if dones == None:
            dones = torch.zeros((*x.shape[:-1], 1))
        x = self.transformer(x, dones==1)
        x = torch.squeeze(x)
        x = F.leaky_relu(x)
        means = self.mean(x)
        stds = self.std(x)
        stds = torch.clamp(stds, min=self.min_std_value)
        return means, stds, self.transformer.get_memory()

    def reset_memory(self):
        self.transformer.reset_memory()

    def zero_grad(self):
        super().zero_grad()
        self.reset_memory()