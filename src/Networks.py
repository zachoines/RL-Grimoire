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
        stack_size: int = 8, 
        num_layers: int = 2, 
        nhead: int = 2, 
        device: torch.device = torch.device("cpu")
    ):
        super().__init__()

        self.in_features = in_features
        self.stack_size = stack_size
        self.hidden_size = hidden_size
        self.state_size = self.in_features // self.stack_size
        self.device = device

        # Embedding layer for older states
        self.embedding = nn.Linear(self.state_size, hidden_size).to(device)

        self.positional_encodings = self.create_positional_encodings(stack_size - 1, hidden_size).to(device)

        # Transformer encoder layers
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, batch_first=True).to(device)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers).to(device)

        # Linear layer for recent state
        self.linear = nn.Linear(self.state_size, hidden_size).to(device)

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1)
        ).to(device)

        # self.apply(self.init_weights)
        self.to(self.device)

    @staticmethod
    def create_positional_encodings(seq_len: int, d_model: int):
        """Creates positional encodings for the Transformer model.

        Args:
            seq_len: The sequence length.
            d_model: The dimension of the embeddings (i.e., model dimension).
            batch_size: The batch size.

        Returns:
            A tensor containing the positional encodings.
        """
        pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pos_enc = pos * div_term
        pos_enc = torch.stack([torch.sin(pos_enc), torch.cos(pos_enc)], dim=-1).view(seq_len, d_model)
        return pos_enc

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight, a=0.01)
            m.bias.data.fill_(0.0)

    def forward(self, state: torch.Tensor):
        """
        Forward pass of the value network.

        Args:
            state (torch.Tensor): Input state tensor of shape (num_envs, num_samples, stack_size * state_size).

        Returns:
            torch.Tensor: Value predictions tensor of shape (num_envs * num_samples, 1).
        """
        if len(state.size()) == 2:
            state = state.unsqueeze(1)

        num_envs, num_samples, _ = state.size()

        # Reshape the state tensor to separate the states in the stack
        state = state.reshape(num_envs * num_samples, self.stack_size, self.state_size)
        recent_state, older_states = state[:, -1, :], state[:, :-1, :]

        # Move tensors to the specified device
        older_states = older_states.to(self.device)
        recent_state = recent_state.to(self.device)

        # Embed the older states and reshape them back into sequence form
        embedded_states = self.embedding(older_states)

        # Add positional encodings to the older states
        embedded_states += self.positional_encodings

        # # Create a mask to ignore zero-padded states in the transformer encoder
        # mask = torch.all(older_states == 0, dim=2)

        # # Create the key_padding_mask by inverting the mask and converting it to a byte tensor
        # key_padding_mask = mask.to(torch.bool).to(self.device)

        # # Pass the older states through the transformer encoder only if there are non-zero states
        # if key_padding_mask.any():
        #     transformer_output = self.transformer_encoder(embedded_states, src_key_padding_mask=key_padding_mask)
        # else:
        #     transformer_output = torch.zeros_like(embedded_states)  # Output zeros if all older states are zero

        # # Handle NaN values in the transformer_output (When older_states are all empty)
        # transformer_output = torch.where(torch.isnan(transformer_output), torch.zeros_like(transformer_output), transformer_output)

        # Pass the older states through the transformer encoder
        transformer_output = self.transformer_encoder(embedded_states)

        # Process the most recent state through the linear layer
        linear_output = self.linear(recent_state)

        # Apply max pooling to the transformer output
        pooled_output = torch.mean(transformer_output, dim=1)

        # Combine the pooled transformer output and linear output
        combined_output = F.leaky_relu(torch.cat((pooled_output, linear_output), dim=-1))

        # Compute the value predictions
        value = self.value_net(combined_output).view(num_envs, num_samples, 1)

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

