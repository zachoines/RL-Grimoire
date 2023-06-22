import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple
import math

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
        stack_size: int = 8,
        num_layers: int = 1,
        nhead: int = 1,
        device: torch.device = torch.device("cpu")
    ):
        super().__init__()
        self.out_features = out_features
        self.in_features = in_features  # Size of the input features
        self.stack_size = stack_size  # Number of past states to stack
        self.hidden_size = hidden_size  # Size of the hidden layer
        self.state_size = self.in_features // self.stack_size  # Size of a single state
        self.device = device  # Device to use for computations

        self.embedding = nn.Linear(self.state_size, hidden_size).to(device)

        self.positional_encodings = self.create_positional_encodings(stack_size - 1, hidden_size).to(device)

        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, batch_first=True, dropout=0).to(device)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers).to(device)

        self.linear = nn.Linear(self.state_size, hidden_size).to(device)

        self.mean = nn.Sequential(
            nn.Linear(hidden_size * 2, out_features),  # Multiply by 2 because we concatenate transformer and linear outputs
            nn.Tanh()
        ).to(device)

        self.std = nn.Sequential(
            nn.Linear(hidden_size * 2, out_features),  # Multiply by 2 because we concatenate transformer and linear outputs
            nn.Softplus()
        ).to(device)

        self.apply(self.init_weights)
        self.eps = 1e-8
        self.min_std_value = 1e-5

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
        # Weight initialization function to be applied to each layer
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight, a=0.01)
            m.bias.data.fill_(0.0)

    def forward(self, state: torch.Tensor):
        """
        Forward pass of the policy network.

        Args:
            state (torch.Tensor): Input state tensor of shape (num_envs, num_samples, stack_size * state_size).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of mean and standard deviation tensors.
                - mean: Tensor of shape (num_envs, num_samples, out_features).
                - std: Tensor of shape (num_envs, num_samples, out_features).
        """
        # Add a singleton dimension for num_samples if necessary
        if len(state.size()) == 2:
            state = state.unsqueeze(1)

        num_envs, num_samples, _ = state.size()

        # Reshape the state tensor to separate the states in the stack
        state = state.reshape(num_envs * num_samples, self.stack_size, self.state_size)

        # Separate the most recent state from the older states
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
        # transformer_output = F.leaky_relu(torch.where(torch.isnan(transformer_output), torch.zeros_like(transformer_output), transformer_output))

        # Pass the older states through the transformer encoder
        transformer_output = self.transformer_encoder(embedded_states)

        # Process the most recent state through the linear layer
        linear_output = self.linear(recent_state)

        # Combine the outputs of the transformer encoder and the linear layer
        # pooled_output = torch.mean(transformer_output, dim=1)
        combined_output = F.leaky_relu(torch.cat((transformer_output[:, -1, :], linear_output), dim=-1))

        # Compute the mean and standard deviation
        means = self.mean(combined_output).view(num_envs, num_samples, self.out_features)
        stds = self.std(combined_output).view(num_envs, num_samples, self.out_features)
        stds = torch.clamp(stds, min=self.min_std_value)  # Clamp the standard deviation to avoid values too close to 0

        return means, stds