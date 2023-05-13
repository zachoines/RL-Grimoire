from gymnasium.spaces import Discrete, Space
import numpy as np

# Torch imports
import torch
from typing import Dict, Iterator
from torch import Tensor
from torch.optim import Optimizer

# Local imports
from Datasets import ExperienceBuffer, Transition
from Hyperparams import Hyperparams
from Policies import DiscreteGradientPolicy
from Utilities import multinomial_select

class Agent:
    def __init__(
            self,
            state_size: int,
            hidden_size: int,
            num_actions: int,
            action_type: Space,
            device: torch.device = torch.device("cpu")
        ):
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        self.action_type = action_type
        self.device = device

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

class REINFORCE(Agent):
    def __init__(
            self,
            state_size,
            hidden_size,
            num_actions,
            action_type,
            device = torch.device("cpu")
        ):

        super().__init__(
            state_size,
            hidden_size,
            num_actions,
            action_type,
            device
        )

        if self.action_type.__class__ == Discrete:
            self.policy = DiscreteGradientPolicy(state_size, num_actions, hidden_size, device=device)
            
        else:
            raise NotImplementedError
    

    def get_actions(self, state: np.ndarray, eval=False)->Tensor:
        if self.action_type.__class__ == Discrete:
            if eval:
                state_tensor = torch.tensor(state, device=self.device, dtype=torch.float32, requires_grad=False)
                action_probs = self.policy(state_tensor)
                return torch.argmax(action_probs)
            else:
                state_tensor = torch.tensor(state, device=self.device, dtype=torch.float32, requires_grad=False)
                return multinomial_select(self.policy(state_tensor))
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
        actions = torch.tensor(actions[indices], device=self.device, dtype=torch.int64)
        returns = torch.tensor(returns[indices], device=self.device, dtype=torch.float32)
        
        # Now determine our policy loss
        probs = self.policy(states)
        log_probs = torch.log(probs + 1e-6)
        action_log_probs = log_probs.gather(1, actions)
        entropy = -torch.sum(probs * log_probs, dim=-1, keepdim=True)

        pg_loss = -action_log_probs * returns
        loss = torch.mean(pg_loss - (hyperparams.entropy_coefficient * entropy))

        # Optimize the model
        optimizers['policy'].zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad.clip_grad_norm_(self.policy.parameters(), 2.0)
        optimizers['policy'].step()
        
        return { 
            "policy_loss": pg_loss.mean(),
            "entropy": entropy.mean(),
            "returns": returns.mean()
        }
        
    def calc_returns(self, rewards, dones, hyperparams: Hyperparams)->np.ndarray:
        # Compute returns by going backwards and iteratively summing discounted rewards
        discounts = [pow(hyperparams.gamma, i) for i in range(len(rewards))]
        running_returns = np.zeros(hyperparams.num_envs, dtype=np.float32)
        returns = np.zeros_like(rewards)
        for i in range(hyperparams.batch_size - 1, -1, -1):
            running_returns = rewards[i] + (1 - dones[i]) * discounts[i] * running_returns
            returns[i] = running_returns

        return returns
    
    def parameter_dict(self) -> Dict[str,Iterator]:
        return {
            "policy": self.policy.parameters()
        }
    
    def state_dict(self)-> Dict[str,Dict]:
        return {
            "policy": self.policy.state_dict()
        }
    
    def load(self, location: str)->None:
        state_dicts = torch.load(location)
        self.policy.load_state_dict(state_dicts["policy"])