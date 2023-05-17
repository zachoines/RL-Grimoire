from gymnasium.spaces import Discrete, MultiDiscrete, Box, Space
import numpy as np

# Torch imports
import torch
from typing import Dict, Iterator
from torch import Tensor
from torch.optim import Optimizer
from torch.distributions import Normal
import torch.nn.functional as F

# Local imports
from Datasets import ExperienceBuffer, Transition
from Hyperparams import Hyperparams
from Policies import DiscreteGradientPolicy, GaussianGradientPolicy
from Networks import ValueNetwork

class Agent:
    def __init__(
            self,
            hidden_size: int,
            observation_space: Space,
            action_space: Space,
            device: torch.device = torch.device("cpu")
        ):

        self.hidden_size = hidden_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device
        self.eps = 1e-8
    
    def save(self, location: str)->None:
        torch.save(self.state_dict(), location)

    def load(self, location: str)->None:
        raise NotImplementedError
    
    def state_dict(self)-> Dict[str,Dict]:
        raise NotImplementedError

    def get_actions(self, state: np.ndarray, eval=False)->Tensor:
        raise NotImplementedError
    
    def learn(self, exp_buffer: ExperienceBuffer, hyperparams: Hyperparams, optimizers: Dict[str,Optimizer])->Dict[str, Tensor]:
        raise NotImplementedError
    
    def parameter_dict(self)->Dict[str, Iterator]:
        raise NotImplementedError
    
    def is_discrete(self):
        return self.action_space.__class__ == Discrete or self.is_multi_discrete()
    
    def is_continous(self):
        return self.action_space.__class__ == Box
    
    def is_multi_discrete(self):
        return self.action_space.__class__ == MultiDiscrete


class REINFORCE(Agent):
    def __init__(
            self,
            hidden_size,
            observation_space,
            action_space,
            device = torch.device("cpu")
        ):

        super().__init__(
            hidden_size,
            observation_space,
            action_space,
            device
        )

        self.state_size = self.observation_space.shape[-1] # type: ignore
        if self.is_discrete():
            if self.is_multi_discrete():
                self.num_actions = self.action_space.nvec[0] # type: ignore
            else:
                self.num_actions = self.action_space.n # type: ignore
            self.policy = DiscreteGradientPolicy(self.state_size, self.num_actions, hidden_size, device)
            self.value = ValueNetwork(self.state_size, hidden_size, device)
        elif self.is_continous():
            self.num_actions = self.action_space.shape[-1] # type: ignore
            self.action_min = float(self.action_space.low_repr) # type: ignore
            self.action_max = float(self.action_space.high_repr) # type: ignore
            self.policy = GaussianGradientPolicy(self.state_size, self.num_actions, hidden_size, device=device)
            self.value = ValueNetwork(self.state_size, hidden_size, device)
        else:
            raise NotImplementedError
        
    def get_actions(self, state: np.ndarray, eval=False)->Tensor:
        if self.is_discrete():
            if eval:
                state_tensor = torch.tensor(state, device=self.device, dtype=torch.float32, requires_grad=False)
                action_probs = self.policy(state_tensor)
                return torch.argmax(action_probs)
            else:
                state_tensor = torch.tensor(state, device=self.device, dtype=torch.float32, requires_grad=False)
                probas = self.policy(state_tensor)
                return torch.multinomial(probas, 1).int()
        elif self.is_continous():
            state_tensor = torch.tensor(state, device=self.device, dtype=torch.float32)
            mean, std = self.policy(state_tensor)

            if eval:
                return mean
            else:
                normal = torch.distributions.Normal(mean + self.eps, std + self.eps) 
                return normal.sample()
                
        else:
            raise NotImplementedError
        
    
    def learn(self, exp_buffer: ExperienceBuffer, hyperparams: Hyperparams, optimizers: Dict[str,Optimizer])->dict[str, Tensor]:
        batch = None
        if len(exp_buffer) >= hyperparams.batch_size:
            batch = exp_buffer.sample(hyperparams.batch_size, remove=True)
        else:
            return {}
        
        # Reshape batch to gathered lists 
        states, actions, _, rewards, dones = map(np.stack, zip(*batch))
        returns = np.array(self.calc_returns(rewards, dones, hyperparams), dtype=np.float32)

        # Reshape data
        num_samples = hyperparams.num_envs * hyperparams.batch_size
        states = states.reshape(num_samples, -1)
        actions = actions.reshape(num_samples, -1)
        returns = returns.reshape(num_samples, -1)

        # Shuffle the batch
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        states = torch.tensor(states[indices], device=self.device, dtype=torch.float32)
        actions = torch.tensor(actions[indices], device=self.device, dtype=torch.float32 if self.is_continous() else torch.int64)
        returns = torch.tensor(returns[indices], device=self.device, dtype=torch.float32)
        mean = returns.mean()
        std = returns.std()
        returns_scaled = (returns - mean) / (std + self.eps)

        if self.is_discrete():
            
            # Log probs and advantage
            advantages = returns_scaled - self.value(states).detach()
            probs = self.policy(states)
            log_probs = torch.log(probs + self.eps)
            action_log_probs = log_probs.gather(1, actions)

            # Policy loss and entropy
            entropy = -torch.sum(probs * log_probs, dim=-1, keepdim=True)
            pg_loss = -action_log_probs * advantages
            loss = torch.mean(pg_loss - (hyperparams.entropy_coefficient * entropy))

            # Value loss
            values = self.value(states)
            value_loss = F.smooth_l1_loss(returns_scaled, values)

            # Optimize the models
            optimizers['policy'].zero_grad()
            loss.backward()
            optimizers['policy'].step()

            optimizers['value'].zero_grad()
            value_loss.backward()
            optimizers['value'].step()
            
            return {
                "Policy loss": pg_loss.mean(),
                "Value loss": value_loss.mean(),
                "Entropy": entropy.mean(),
                "Discounted returns": returns.mean()
            }
        elif self.is_continous():
        
            # Guassian Policy log probabilities
            loc, scale = self.policy(states)
            dist = Normal(loc, scale)
            log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)

            # Policy loss and entropy
            entropy = dist.entropy().sum(dim=-1, keepdim=True)
            advantages = returns_scaled - self.value(states).detach()
            pg_loss = -log_probs * advantages
            loss = torch.mean(pg_loss - (hyperparams.entropy_coefficient * entropy))

            # Value loss
            value_loss = torch.nn.functional.mse_loss(self.value(states), returns_scaled)

            # Optimize the models
            optimizers['policy'].zero_grad()
            loss.backward()
            optimizers['policy'].step()

            optimizers['value'].zero_grad()
            value_loss.backward()
            optimizers['value'].step()

            return {
                "Policy loss": pg_loss.mean(),
                "Value loss": value_loss.mean(),
                "Entropy": entropy.mean(),
                "Discounted returns": returns.mean()
            }
        else:
            raise NotImplementedError
        
    def calc_returns(self, rewards, dones, hyperparams: Hyperparams)->np.ndarray:
        # Compute returns by going backwards and iteratively summing discounted rewards
        # discounts = [pow(hyperparams.gamma, i) for i in range(len(rewards))]
        running_returns = np.zeros(hyperparams.num_envs, dtype=np.float32)
        returns = np.zeros_like(rewards)
        for i in range(hyperparams.batch_size - 1, -1, -1):
            running_returns = rewards[i] + (1 - dones[i]) * hyperparams.gamma * running_returns
            returns[i] = running_returns

        return returns
    
    def rescaleAction(self, action, min, max):
        return torch.clip((min + (0.5 * (action + 1.0) * (max - min))), min, max)
    
    def parameter_dict(self) -> Dict[str,Iterator]:
        return {
            "policy": self.policy.parameters(),
            "value": self.value.parameters()
        }
    
    def state_dict(self)-> Dict[str,Dict]:
        return {
            "policy": self.policy.state_dict()
        }
    
    def load(self, location: str)->None:
        state_dicts = torch.load(location)
        self.policy.load_state_dict(state_dicts["policy"])
