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
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(),
        )

        self.mean = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LeakyReLU(),
            nn.Linear(hidden_size // 4, out_features),
            nn.Tanh()
        )

        self.std = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LeakyReLU(),
            nn.Linear(hidden_size // 4, out_features),
            nn.Softplus()
        )

        # self.apply(self.init_weights) 
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
        stack_size: int = 32,
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

        # Embedding layer for states
        self.embedding = nn.Linear(self.state_size, hidden_size).to(device)

        # Positional encoding
        self.position_embedding = nn.Embedding(self.stack_size, hidden_size).to(device)
        self.positional_encodings = self.create_positional_encodings(stack_size, hidden_size).to(device)

        # Transformer Encoder Layers
        encoder_layers = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=nhead, batch_first=True, dropout=0).to(device)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers).to(device)

        # Linear layer for attention scores
        self.attention_linear = nn.Linear(hidden_size, 1).to(device)

        # Mean and std layers
        self.mean = nn.Linear(hidden_size, out_features).to(device)
        self.std = nn.Linear(hidden_size, out_features).to(device)

        self.eps = 1e-8
        self.min_std_value = 1e-4

        self.to(self.device)

    @staticmethod
    def create_positional_encodings(seq_len: int, d_model: int):
        """Creates positional encodings for the Transformer model.

        Args:
            seq_len: The sequence length.
            d_model: The dimension of the embeddings (i.e., model dimension).

        Returns:
            A tensor containing the positional encodings.
        """
        pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pos_enc = torch.zeros(seq_len, d_model)
        pos_enc[:, 0::2] = torch.sin(pos * div_term)
        if d_model % 2 == 1:
            # For odd dimension models, compute one extra term for the cosine function
            pos_enc[:, 1::2] = torch.cos(pos * div_term[:-1])
        else:
            pos_enc[:, 1::2] = torch.cos(pos * div_term)

        return pos_enc

    def forward(self, state: torch.Tensor):
        # Add a singleton dimension for num_samples if necessary
        if len(state.size()) == 2:
            state = state.unsqueeze(1)

        num_envs, num_samples, _ = state.size()

        # Reshape the state tensor to separate the states in the stack
        state = state.reshape(num_envs * num_samples, self.stack_size, self.state_size).to(self.device)

        # Pass the states through the embedding layer
        embedded_states = self.embedding(state)

        # Alternative method for positional encodings
        embedded_states = self.positional_encodings + embedded_states

        # Create and add positional embeddings to states
        # positions = torch.arange(self.stack_size, device=self.device).expand(num_envs * num_samples, self.stack_size)
        # embedded_states = embedded_states + self.position_embedding(positions)

        # Pass the embedded states through the transformer encoder
        transformer_output = F.leaky_relu(self.transformer_encoder(embedded_states))

        # Compute attention scores for the transformer outputs
        attention_scores = F.softmax(self.attention_linear(transformer_output), dim=1)

        # Apply attention pooling over the transformer output
        transformer_output = torch.sum(transformer_output * attention_scores, dim=1)

        # Compute the mean and standard deviation
        means = F.tanh(self.mean(transformer_output).view(num_envs, num_samples, self.out_features))
        stds = F.softplus(self.std(transformer_output).view(num_envs, num_samples, self.out_features))

        # Clamp the standard deviation to avoid values too close to 0
        stds = torch.clamp(stds, min=self.min_std_value)

        return means, stds
