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
        stack_size: int = 32, 
        num_layers: int = 1, 
        nhead: int = 1, 
        device: torch.device = torch.device("cpu")
    ):
        super().__init__()

        self.in_features = in_features  # Size of the input features
        self.stack_size = stack_size  # Number of past states to stack
        self.hidden_size = hidden_size  # Size of the hidden layer
        self.state_size = self.in_features // self.stack_size  # Size of a single state
        self.device = device  # Device to use for computations

        # Embedding layer for states
        self.embedding = nn.Linear(self.state_size, hidden_size).to(device)

        # Positional embedding layer
        self.position_embedding = nn.Embedding(stack_size, hidden_size).to(device)
        self.positional_encodings = self.create_positional_encodings(stack_size, hidden_size).to(device)

        # Transformer Encoder Layers
        encoder_layers = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=nhead, batch_first=True, dropout=0).to(device)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers).to(device)

        # Linear layer for computing attention scores
        self.attention_linear = nn.Linear(hidden_size, 1).to(device)

        # Value layer
        self.value_output = nn.Linear(hidden_size, 1).to(device)

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
        # embedded_states = self.positional_encodings + embedded_states

        # Create and add positional embeddings to states
        positions = torch.arange(self.stack_size, device=self.device).expand(num_envs * num_samples, self.stack_size)
        embedded_states = embedded_states + self.position_embedding(positions)

        # Pass the embedded states through the transformer encoder
        transformer_output = F.leaky_relu(self.transformer_encoder(embedded_states))

        # Compute attention scores for the transformer outputs
        attention_scores = F.softmax(self.attention_linear(transformer_output), dim=1)

        # Apply attention pooling over the transformer output
        transformer_output = torch.sum(transformer_output * attention_scores, dim=1)

        # Compute the value predictions
        value = self.value_output(transformer_output).view(num_envs, num_samples, 1)

        return value
