import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from typing import Tuple
import math
import torch.distributions as distributions

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
    
class ValueNetworkTransformerV1(nn.Module):
    def __init__(
        self, 
        in_features: int, 
        hidden_size: int, 
        stack_size: int = 16, 
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
        embedded_states_with_positions = self.positional_encodings.detach() + embedded_states

        # Create and add positional embeddings to states
        # positions = torch.arange(self.stack_size, device=self.device).expand(num_envs * num_samples, self.stack_size)
        # embedded_states_with_positions = embedded_states + self.position_embedding(positions)

        # Pass the embedded states through the transformer encoder
        transformer_output = F.leaky_relu(self.transformer_encoder(embedded_states_with_positions))

        # Compute attention scores for the transformer outputs
        attention_scores = F.softmax(self.attention_linear(transformer_output), dim=1)

        # Apply attention pooling over the transformer output
        transformer_output = torch.sum(transformer_output * attention_scores, dim=1)

        # Compute the value predictions
        value = self.value_output(transformer_output).view(num_envs, num_samples, 1)

        return value

class ValueNetworkTransformer(nn.Module):
    def __init__(
        self, 
        in_features: int, 
        hidden_size: int, 
        stack_size: int = 16, 
        num_layers: int = 1, 
        nhead: int = 1, 
        device: torch.device = torch.device("cpu")
    ):
        super().__init__()

        self.in_features = in_features
        self.stack_size = stack_size
        self.hidden_size = hidden_size
        self.state_size = self.in_features // self.stack_size
        self.device = device

        # Embedding layer for states
        self.embedding = nn.Linear(self.state_size, hidden_size).to(device)

        # Positional embedding layer
        self.position_embedding = nn.Embedding(stack_size, hidden_size).to(device)
        self.positional_encodings = self.create_positional_encodings(stack_size, hidden_size).to(device)

        # Transformer Encoder Layers
        encoder_layers = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=nhead, batch_first=True, dropout=0).to(device)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers).to(device)

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
            pos_enc[:, 1::2] = torch.cos(pos * div_term[:-1])
        else:
            pos_enc[:, 1::2] = torch.cos(pos * div_term)

        return pos_enc

    def forward(self, state: torch.Tensor):
        # Add a singleton dimension for num_samples if necessary
        if len(state.size()) == 2:
            state = state.unsqueeze(1)

        num_envs, num_samples, _ = state.size()
        state = state.reshape(num_envs * num_samples, self.stack_size, self.state_size).to(self.device)

        # Pass the states through the embedding layer
        embedded_states = self.embedding(state)

        # Add positional encodings to the embedded states
        embedded_states_with_positions = self.positional_encodings.detach() + embedded_states

        # Pass the embedded states through the transformer encoder
        transformer_output = F.leaky_relu(self.transformer_encoder(embedded_states_with_positions))

        # Select only the last output of the transformer (last time step)
        last_output = transformer_output[:, -1, :]

        # Compute the value predictions
        value = self.value_output(last_output).view(num_envs, num_samples, 1)

        return value

class ValueNetworkGRU(nn.Module):
    def __init__(
            self,
            in_features: int,
            hidden_size: int,
            num_layers: int = 2,
            device: torch.device = torch.device("cpu")
    ):
        super().__init__()

        self.shared_net = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.LeakyReLU(),
        )

        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=False
        )

        self.value_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.device = device
        self.to(self.device)

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.hidden = None  # Hidden state
        self.prev_hidden = None  # Previous hidden state

        # self.init_weights()

    def init_hidden(self, batch_size: int):
        """Initialize the hidden state for a new batch."""
        self.hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device).contiguous()

    def get_hidden(self):
        """Return the current hidden state."""
        return self.hidden

    def get_prev_hidden(self):
        """Return the previous hidden state."""
        return self.prev_hidden.contiguous()

    def set_hidden(self, hidden):
        """Set the hidden state to a specific value."""
        self.hidden = hidden

    def init_weights(self):
        """Initialize the network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.GRU):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0.0)

    def forward(self, x, input_hidden=None, dones=None):
        seq_length, batch_size, *_ = x.size() 

        # Enforce internal hidden state
        if self.hidden is None or self.hidden.size(2) != batch_size:
            self.init_hidden(batch_size)

        # Pass through shared net 
        shared_features = self.shared_net(x.to(self.device))

        # Process each sequence step, taking dones into consideration
        gru_outputs = []

        if dones is not None: # TODO: check if "and torch.any(dones):" works
            for t in range(seq_length):
                
                # Set the hidden state
                self.hidden = input_hidden[t, :] if input_hidden is not None else self.hidden  
                
                # Reset hidden state for environments that are done
                mask = dones[t].to(dtype=torch.bool, device=self.device)
                self.hidden[:, mask, :] = 0.0

                self.prev_hidden = self.hidden
                gru_output, self.hidden = self.gru(shared_features[t].unsqueeze(0), self.hidden.clone())
                gru_outputs.append(gru_output)
            gru_outputs = torch.stack(gru_outputs)
        else:
            self.hidden = input_hidden if input_hidden is not None else self.hidden
            self.prev_hidden = self.hidden
            gru_outputs, self.hidden = self.gru(shared_features, self.hidden)

        value = self.value_net(gru_outputs)
        return value, self.hidden

class ValueNetworkLSTM(nn.Module):
    def __init__(
            self,
            in_features: int,
            hidden_size: int,
            num_layers: int = 1,
            device: torch.device = torch.device("cpu")
    ):
        super().__init__()

        # Shared network layer
        self.shared_net = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.LeakyReLU(),
        )

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=False
        )

        # Value network layers
        self.value_net = nn.Sequential(
            # nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.device = device
        self.to(self.device)

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.hidden = None  # Hidden state
        self.cell = None  # Cell state
        self.prev_hidden = None  # Previous hidden state
        self.prev_cell = None  # Previous cell state

    def init_hidden(self, batch_size: int):
        """Initialize the hidden state and cell state for a new batch."""
        self.hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device).contiguous()
        self.cell = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device).contiguous()

    def get_hidden(self):
        """Return the current hidden and cell states concatenated."""
        return torch.cat((self.hidden, self.cell), dim=0) # type: ignore

    def get_prev_hidden(self):
        """Return the previous hidden and cell states concatenated."""
        return torch.cat((self.prev_hidden, self.prev_cell), dim=0) # type: ignore

    def set_hidden(self, hidden):
        """Set the hidden state and cell state to a specific value."""
        self.hidden, self.cell = torch.split(hidden.clone(), hidden.size(0) // 2, dim=0)

    def forward(self, x, input_hidden=None, dones=None):
        seq_length, batch_size, *_ = x.size() 

        # Enforce internal hidden state
        if self.hidden is None or self.hidden.size(1) != batch_size or self.cell is None or self.cell.size(1) != batch_size:
            self.init_hidden(batch_size)

        # Pass through shared net 
        shared_features = self.shared_net(x.to(self.device))

        # Process each sequence step, taking dones into consideration
        lstm_outputs = []

        for t in range(seq_length):
            
            # Set the hidden and cell states
            if input_hidden is not None:
                self.set_hidden(input_hidden[t, :])
            
            # Reset hidden and cell states for environments that are done
            if dones is not None:
                mask = dones[t].to(dtype=torch.bool, device=self.device)
                self.hidden[:, mask, :] = 0.0 # type: ignore
                self.cell[:, mask, :] = 0.0 # type: ignore

            self.prev_hidden = self.hidden
            self.prev_cell = self.cell
            lstm_output, (self.hidden, self.cell) = self.lstm(shared_features[t].unsqueeze(0), (self.hidden.detach(), self.cell.detach())) # type: ignore
            lstm_outputs.append(lstm_output)
        lstm_outputs = torch.cat(lstm_outputs, dim=0)

        value = self.value_net(lstm_outputs)
        return value, self.get_hidden()

class ICM(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int, state_feature_dim: int, device: torch.device = torch.device("cpu")):
        """
        Initialize an Intrinsic Curiosity Module.

        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            hidden_size (int): The size of the hidden layers in the forward and inverse models.
            state_feature_dim (int): The output dimension of the forward model.
            device (torch.device): The device to run the network on.
        """
        super(ICM, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.state_feature_dim = state_feature_dim

        # Feature Network
        self.feature_network = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, state_feature_dim),
            nn.ReLU()
        ).to(self.device)

        # Forward Model
        self.forward_model = nn.Sequential(
            nn.Linear(state_feature_dim + (action_dim * 2), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, state_feature_dim)
        ).to(self.device)

        # Inverse Model
        self.inverse_model_mu = nn.Sequential(
            nn.Linear(state_feature_dim * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh()
        ).to(self.device)
        self.inverse_model_log_std = nn.Sequential(
            nn.Linear(state_feature_dim * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Softplus()
        ).to(self.device)

    def forward(self, states_plus_one: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor, n: float = 0.5, beta: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state_plus_one = states_plus_one.to(self.device)
        loc = loc.to(self.device)
        scale = scale.to(self.device)
        
        # Feature Network
        state_plus_one_features = self.feature_network(state_plus_one)

        state_features = state_plus_one_features[:-1]  # All states except the last one
        next_state_features = state_plus_one_features[1:]  # All states except the first one

        # Forward Model
        x = torch.cat([state_features, loc, scale], dim=2)
        predicted_next_state_feature = self.forward_model(x)

        # Inverse Model
        x = torch.cat([state_features, predicted_next_state_feature], dim=2)
        predicted_loc = self.inverse_model_mu(x)
        predicted_scale = self.inverse_model_log_std(x)

        forward_loss_per_state = F.mse_loss(predicted_next_state_feature, next_state_features.detach(), reduction='none')
        forward_loss = beta * forward_loss_per_state.mean()
        inverse_loss = (1 - beta) * (F.mse_loss(predicted_loc, loc.detach()) + F.mse_loss(predicted_scale, scale.detach()))

        # Intrinsic reward is the forward loss for each state
        intrinsic_reward = n * forward_loss_per_state.mean(-1).squeeze(-1).detach()

        # return the losses, the intrinsic reward
        return forward_loss, inverse_loss, intrinsic_reward
   
class ICMRecurrent(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int, state_feature_dim: int, num_layers = 1, device: torch.device = torch.device("cpu")):
        """
        Initialize an Intrinsic Curiosity Module with Recurrent Networks.

        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            hidden_size (int): The size of the hidden layers in the forward and inverse models.
            state_feature_dim (int): The output dimension of the forward model.
            num_layers (int): The number of layers in the LSTM.
            device (torch.device): The device to run the network on.
        """
        super(ICMRecurrent, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.state_feature_dim = state_feature_dim
        self.hidden = None
        self.cell = None

        # Feature Network
        self.feature_network = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        ).to(self.device)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=state_feature_dim,  # output of LSTM is state_feature_dim
            batch_first=False,
            num_layers=num_layers
        ).to(self.device)

        # Forward Model
        self.forward_model = nn.Sequential(
            nn.Linear(state_feature_dim + (action_dim * 2), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, state_feature_dim)
        ).to(self.device)

        # Inverse Model
        self.inverse_model_mu = nn.Sequential(
            nn.Linear(state_feature_dim * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh()
        ).to(self.device)
        self.inverse_model_log_std = nn.Sequential(
            nn.Linear(state_feature_dim * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Softplus()
        ).to(self.device)

    def init_hidden(self, batch_size):
        self.hidden = torch.zeros(self.num_layers, batch_size, self.state_feature_dim).to(self.device)
        self.cell = torch.zeros(self.num_layers, batch_size, self.state_feature_dim).to(self.device)

    def get_hidden(self, batch_size=None):
        if self.hidden is None:
            self.init_hidden(batch_size)

        """Return the current hidden and cell states concatenated."""
        return torch.cat((self.hidden, self.cell), dim=0) # type: ignore

    def set_hidden(self, hidden):
        """Set the hidden state and cell state to a specific value."""
        self.hidden, self.cell = torch.split(hidden.clone().detach(), hidden.size(0) // 2, dim=0)

    def forward(self, states_plus_one: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor, dones_plus_one=None, input_hidden=None, n: float = 0.5, beta: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run the ICMRecurrent.

        Args:
            states_plus_one (torch.Tensor): The states plus one next state.
            loc (torch.Tensor): The loc of the action distribution.
            scale (torch.Tensor): The scale of the action distribution.
            dones_plus_one (torch.Tensor): Done flags indicating the end of episodes, plus one done for the next state.
            input_hidden (torch.Tensor): The input hidden state for LSTM.
            n (float): scaler for intrinsic rewards.
            beta (float): Weighing term that balances forward and inverse loss.

        Returns:
            torch.Tensor: The forward model loss.
            torch.Tensor: The inverse model loss.
            torch.Tensor: The intrinsic reward.
            torch.Tensor: The current LSTM hidden state.
        """
        seq_length, batch_size, *_ = states_plus_one.size() 

        if self.hidden is None or self.hidden.size(1) != batch_size or self.cell is None or self.cell.size(1) != batch_size:
            self.init_hidden(batch_size)

        state_plus_one = states_plus_one.to(self.device)
        loc = loc.to(self.device)
        scale = scale.to(self.device)
        
        # Feature Network
        state_plus_one_features = self.feature_network(state_plus_one)

        # LSTM for the feature network
        lstm_outputs = []
        hidden_outputs = torch.zeros(seq_length, self.num_layers * 2, batch_size, self.state_feature_dim).to(self.device)  # Initialize tensor for hidden states
        for t in range(seq_length):
            if input_hidden is not None:
                self.set_hidden(input_hidden[t, :])
            
            if dones_plus_one is not None:
                mask = dones_plus_one[t].to(dtype=torch.bool, device=self.device)
                self.hidden[:, mask, :] = 0.0 # type: ignore
                self.cell[:, mask, :] = 0.0 # type: ignore
            
            hidden_outputs[t] = self.get_hidden()  # Store hidden states in tensor
            lstm_output, (self.hidden, self.cell) = self.lstm(state_plus_one_features[t].unsqueeze(0), (self.hidden.detach(), self.cell.detach())) # type: ignore
            lstm_outputs.append(lstm_output)
            
        state_plus_one_features = torch.cat(lstm_outputs, dim=0)

        state_features = state_plus_one_features[:-1]  # All states except the last one
        next_state_features = state_plus_one_features[1:]  # All states except the first one

        # Forward Model
        x = torch.cat([state_features, loc, scale], dim=2)
        predicted_next_state_feature = self.forward_model(x)

        # Inverse Model
        x = torch.cat([state_features, predicted_next_state_feature], dim=2)
        predicted_loc = self.inverse_model_mu(x)
        predicted_scale = self.inverse_model_log_std(x)

        forward_loss_per_state = F.mse_loss(predicted_next_state_feature, next_state_features.detach(), reduction='none')
        forward_loss = beta * forward_loss_per_state.mean()
        inverse_loss = (1 - beta) * (F.mse_loss(predicted_loc, loc.detach()) + F.mse_loss(predicted_scale, scale.detach()))

        # Intrinsic reward is the forward loss for each state
        intrinsic_reward = n * forward_loss_per_state.mean(-1).squeeze(-1).detach()

        # return the losses, the intrinsic reward and the current LSTM hidden state
        return forward_loss, inverse_loss, intrinsic_reward, hidden_outputs

class RND(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, feature_dim: int, device: torch.device = torch.device("cpu")):
        super(RND, self).__init__()
        self.device = device

        # Predictor Network
        self.predictor_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        ).to(self.device)

        # Target Network
        self.target_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        ).to(self.device)
        # Initialize the target network with random weights and set it to not trainable
        self.target_network.requires_grad_(False)
        for param in self.target_network.parameters():
            param.requires_grad = False

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Run the RND.

        Args:
            states (torch.Tensor): The states.

        Returns:
            torch.Tensor: The intrinsic reward.
        """
        states = states.to(self.device)
        
        target_features = self.target_network(states)
        predicted_features = self.predictor_network(states)

        # Intrinsic reward is the MSE between the target and predicted features
        intrinsic_reward = F.mse_loss(predicted_features, target_features.detach(), reduction='none').mean(dim=-1)

        return intrinsic_reward