import torch
from torch import nn
from torch.nn import functional as F
from Utilities import to_tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math

class DuelingNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_size, num_actions, device):
        super(DuelingNetwork, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU()
        )
        
        self.advantage = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions)
        )
        
        self.value = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.apply(self.init_weights) 
        self.device = device
        self.to(self.device)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.kaiming_uniform_(m.weight)
            m.bias.data.fill_(0.0)
        
    def forward(self, x):
        x = self.feature(x)
        advantage = self.advantage(x)
        value     = self.value(x)
        return value, advantage
    

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
            nn.init.kaiming_normal_(m.weight, a=0.01)
            m.weight.data *= 0.1
            m.bias.data.fill_(0.0)

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
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1)
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
            nn.init.kaiming_normal_(m.weight, a=0.01)
            # m.weight.data *= 0.1
            m.bias.data.fill_(0.0)

    def forward(self, x):
        hidden_features = self.hidden_layers(x)
        residual_features = self.residual_layers(x)

        combined_features = torch.cat((hidden_features, residual_features), dim=-1)
        output = self.output_layer(combined_features)

        return output

class ValueNetworkTransformer(nn.Module):
    def __init__(
        self,
        in_features: int,
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

        # Linear layer for value computation
        self.linear = nn.Linear(hidden_size, 1).to(device)

        self.apply(self.init_weights)
        self.device = device

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight, a=0.01)
            m.bias.data.fill_(0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the value network.

        Args:
            state (torch.Tensor): Input state tensor of shape (num_envs, state_size * stack_size).

        Returns:
            torch.Tensor: Value tensor of shape (num_envs,).
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

        # Split the computation into smaller chunks
        chunk_size = 1024
        chunks = torch.split(embedded_state, chunk_size, dim=0)
        outputs = []
        for chunk in chunks:
            result = self.shared_net(chunk, chunk)
            outputs.append(result)

        # Combine the outputs from all chunks
        shared_features = torch.cat(outputs, dim=0)

        # Reshape back to (num_envs, num_samples, stack_size, hidden_size)
        shared_features = shared_features.reshape((num_envs, num_samples, self.stack_size, self.hidden_size)).to(self.device)

        # Apply mean pooling along the spatial dimensions
        pooled_features = torch.mean(shared_features, dim=2)

        # Reshape back to (num_envs, num_samples, hidden_size) or (num_envs, hidden_size) in the single sample case
        pooled_features = pooled_features.reshape((num_envs, num_samples, self.hidden_size))

        # Compute value
        value = self.linear(pooled_features)

        return value

class ValueNetworkRecurrent(nn.Module):
    def __init__(self, in_features: int, hidden_size: int, device: torch.device = torch.device("cpu")):
        super().__init__()

        self.transformer = StatefulTransformer(in_features, hidden_size, hidden_size, device=device)
        self.value_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.apply(self.init_weights)
        self.device = device
        self.to(self.device)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight, a=0.01)
            m.weight.data *= 0.1
            m.bias.data.fill_(0.0)

    def forward(self, x, dones=None):
        x = self.transformer(x, dones==1)
        x = torch.squeeze(x)
        x = F.leaky_relu(x)
        x = self.value_net(x)
        return x

    def reset_memory(self):
        self.transformer.reset_memory()

    def zero_grad(self):
        super().zero_grad()
        self.reset_memory()

class StatefulTransformer(nn.Module):
    def __init__(self, in_features: int, hidden_size: int, output_size: int, device: torch.device, max_memory: int = 100):
        super().__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.device = device
        self.max_memory = max_memory

        self.embedding = nn.Linear(in_features, hidden_size).to(device)
        self.transformer = TransformerEncoder(
            TransformerEncoderLayer(d_model=hidden_size, nhead=8, dim_feedforward=hidden_size, batch_first=True),
            num_layers=1
        ).to(device)
        self.out = nn.Linear(hidden_size, output_size).to(device)

        # Initial hidden state for each environment
        self.memory = {}

    def update_memory(self, env_id, sequence):
        """Update the memory of a trajectory by appending new states and keeping only the most recent states."""
        if env_id in self.memory:
            # Append new sequence to memory
            updated_memory = torch.cat([self.memory[env_id], sequence], dim=0)
            # Keep only the most recent states up to max_memory
            num_extra_states = max(0, updated_memory.shape[0] - self.max_memory)
            updated_memory = updated_memory[num_extra_states:]
        else:
            updated_memory = sequence

        self.memory[env_id] = updated_memory

    def forward(self, x, dones, mode='train'):
        """Process input sequences through the transformer and return the output for each step of each sequence."""
        x = x.to(self.device)
        dones = dones.to(self.device)
        x_embedded = self.embedding(x)

        batch_size, seq_len, _ = x.size()

        outputs = []
        for i in range(batch_size):
            env_id = i  # Assuming the environment id is the index in the batch
            traj_starts = [0] + (dones[i].nonzero(as_tuple=True)[0] + 1).tolist()
            if traj_starts[-1] != seq_len:
                traj_starts.append(seq_len)
            env_outputs = []
            for start, end in zip(traj_starts[:-1], traj_starts[1:]):
                sequence = x_embedded[i, start:end]
                if env_id not in self.memory or (mode == 'eval' and dones[i, start]) or (mode == 'train' and dones[i, end-1]):
                    # Start a new trajectory
                    self.memory[env_id] = sequence
                else:
                    # Continue an existing trajectory
                    self.update_memory(env_id, sequence)

                # Process the trajectory with the transformer
                output = self.transformer(self.memory[env_id].unsqueeze(0))

                # Append the corresponding output for this trajectory
                env_outputs.append(self.out(output[:, -sequence.size(0):, :]))

            # Concatenate all outputs for this environment
            outputs.append(torch.cat(env_outputs, dim=1))

        return torch.stack(outputs, dim=0)

    def reset_memory(self):
        """Reset the memory of all trajectories."""
        self.memory = {}

    def get_memory(self):
        """Return a copy of the memory dictionary."""
        return self.memory.copy()

    def set_memory(self, memory):
        """Set the memory dictionary to a given value."""
        self.memory = memory

