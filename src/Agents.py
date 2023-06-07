from gymnasium.spaces import Discrete, MultiDiscrete, Box, Space
import numpy as np

# Torch imports
import torch
from typing import Dict, Union
from torch import Tensor
from torch.optim import Optimizer, AdamW
from torch.distributions import Normal
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.utils.clip_grad import clip_grad_norm_

# Local imports
from Datasets import Transition
from Configurations import *
from Policies import *
from Networks import *
from Utilities import to_tensor

class Agent:
    def __init__(
            self,
            observation_space: Space,
            action_space: Space,
            hyperparams: AgentParams = AgentParams(), 
            device: torch.device = torch.device("cpu")
        ):

        self.hidden_size = hyperparams.hidden_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device
        self.eps = 1e-8
        self.max_grad_norm = 1.0
        self.hyperparams = hyperparams
        self.optimizers = {}
    
    def save(self, location: str)->None:
        torch.save(self.state_dict(), location)

    def load(self, location: str)->None:
        raise NotImplementedError
    
    def state_dict(self)-> Dict[str,Dict]:
        raise NotImplementedError

    def get_actions(self, state: torch.Tensor, eval=False)->tuple[Tensor, Tensor]:
        raise NotImplementedError
    
    def get_optimizers(self) -> Dict[str, Optimizer]:
        raise NotImplementedError
    
    def is_discrete(self) -> bool:
        return self.action_space.__class__ == Discrete or self.is_multi_discrete()
    
    def is_continous(self) -> bool:
        return self.action_space.__class__ == Box
    
    def is_multi_discrete(self) -> bool:
        return self.action_space.__class__ == MultiDiscrete
    
    def rescaleAction(self, action : Tensor, min : float, max: float) -> torch.Tensor:
        return min + (0.5 * (action + 1.0) * (max - min))
    
    def calc_returns(self, rewards: np.ndarray, dones: np.ndarray, num_envs: int, batch_size: int)->np.ndarray:
        running_returns = np.zeros(num_envs, dtype=np.float32)
        returns = np.zeros_like(rewards)
        for i in range(batch_size - 1, -1, -1):
            running_returns = rewards[i] + (1 - dones[i]) * self.hyperparams.gamma * running_returns
            returns[i] = running_returns

        return returns
    
    def learn(self, batch: list[Transition], num_envs: int, batch_size: int)->dict[str, Tensor]:
        raise NotImplementedError


class PPO2(Agent):
    def __init__(
            self,
            observation_space: Space,
            action_space: Space,
            hyperparams: PPO2Params, 
            device = torch.device("cpu")
        ):

        super().__init__(
            observation_space,
            action_space,
            hyperparams,
            device
        )

        self.hyperparams = hyperparams
        self.state_size = self.observation_space.shape[-1] # type: ignore
        if self.is_continous():
            self.num_actions = self.action_space.shape[-1] # type: ignore
            self.action_min = float(self.action_space.low_repr) # type: ignore
            self.action_max = float(self.action_space.high_repr) # type: ignore
            self.actor = GaussianGradientPolicyV3(self.state_size, self.num_actions, self.hidden_size, device=device)
            self.critic = ValueNetworkV1(self.state_size, self.hidden_size, device)
            self.target_critic = ValueNetworkV1(self.state_size, self.hidden_size, device)

            # make target network initially the same parameters
            self.target_critic.load_state_dict(self.critic.state_dict())
            self.max_grad_norm = .50
            
        else:
            raise NotImplementedError
        
        self.optimizers = self.get_optimizers()
        
    def get_actions(self, state: torch.Tensor, eval=False)->tuple[Tensor, Tensor]:
        if self.is_continous():
            mean, std = self.actor(state)
            # mean = self.rescaleAction(mean, self.action_min, self.action_max)
            normal = torch.distributions.Normal(mean, std) 

            if eval:
                return mean, normal.log_prob(mean).sum(dim=-1)
            else:
                action_sample = normal.sample()
                return action_sample, normal.log_prob(action_sample).sum(dim=-1)
        else:
            raise NotImplementedError

    def learn(self, batch: list[Transition], num_envs: int, batch_size: int, num_rounds: int = 5, mini_batch_size: int = 32) -> dict[str, Tensor]:
        # Reshape batch to gathered lists
        states, actions, next_states, rewards, dones, other = map(torch.stack, zip(*batch))

        # Reshape and send to device
        states = states.to(device=self.device, dtype=torch.float32).view((num_envs, batch_size, -1))
        actions = actions.to(device=self.device, dtype=torch.float32).view((num_envs, batch_size, -1))
        next_states = next_states.to(device=self.device, dtype=torch.float32).view((num_envs, batch_size, -1))
        rewards = rewards.to(device=self.device, dtype=torch.float32).view((num_envs, batch_size, -1))
        dones = dones.to(device=self.device, dtype=torch.float32).view((num_envs, batch_size, -1))
        prev_log_probs = other.to(device=self.device, dtype=torch.float32).view((num_envs, batch_size)).detach()

        # Compute values for states and next states
        with torch.no_grad():
            values = self.critic(states.view(-1, states.size(-1)))
            next_values = self.critic(next_states.view(-1, next_states.size(-1))) 
            # self.target_critic(next_states.view(-1, next_states.size(-1)))

        # Reshape back to original dimensions for G_t and deltas computation
        values = values.view(num_envs, batch_size, -1)
        next_values = next_values.view(num_envs, batch_size, -1)

        # Calculate advantages with gae
        advantages_all = torch.zeros_like(values)
        advantages = torch.zeros((num_envs, 1), device=self.device)

        for t in reversed(range(batch_size)):
            # Non-terminal mask for this timestep
            non_terminal = 1.0 - dones[:, t]

            # Adjust next_values for terminal states
            next_values_adjusted: torch.Tensor
            if t == batch_size - 1:
                next_values_adjusted = dones[:, t] * next_values[:, t] 
            else:
                next_values_adjusted = dones[:, t] * next_values[:, t] + (1.0 - dones[:, t]) * next_values[:, t+1]

            # Compute the delta for this timestep
            delta = rewards[:, t] + self.hyperparams.gamma * non_terminal * next_values_adjusted - values[:, t]

            # Update advantages with delta and discounted, lambda-scaled future advantage
            advantages = delta + self.hyperparams.gamma * self.hyperparams.gae_lambda * non_terminal * advantages

            # Store this timestep's advantages
            advantages_all[:, t] = advantages

      
        # Flatten states, actions, log probabilities, advantages, and targets
        states = states.view(-1, states.size(-1))
        actions = actions.view(-1, actions.size(-1))
        prev_log_probs = prev_log_probs.view(-1)
        advantages_all = advantages_all.view(-1)
        
        # Normalize advantages and calculate targets (while shifting their mean towards actual returns)
        advantages_all = ((advantages_all - advantages_all.mean()) / (advantages_all.std() + 1e-8)).view(-1).detach()
        targets = (advantages_all + values.view(-1)).detach() # normally its: targets = G_t.view(-1)

        # Initialize loss and entropy accumulators
        total_loss_actor, total_loss_critic, total_entropy = 0, 0, 0

        # Perform multiple rounds of learning
        for _ in range(num_rounds):
            # Shuffle data for stochastic gradient descent
            rand_ids = torch.randperm(states.size(0))

            # Process each mini-batch
            for start_idx in range(0, states.size(0), mini_batch_size):
                # Mini-batch indices
                ids = rand_ids[start_idx:start_idx + mini_batch_size]

                # Extract mini-batch data
                mb_states = states[ids]
                mb_actions = actions[ids]
                mb_old_log_probs = prev_log_probs[ids].detach()
                mb_advantages = advantages_all[ids].detach()
                mb_targets = torch.unsqueeze(targets[ids], dim=-1).detach()

                # Compute policy distribution parameters
                loc, scale = self.actor(mb_states)
                dist = Normal(loc, scale)
                log_probs = dist.log_prob(mb_actions).sum(dim=-1)

                # Compute entropy bonus
                entropy = dist.entropy().mean()

                # Compute the policy loss using the PPO clipped objective
                ratio = torch.exp(log_probs - mb_old_log_probs)
                loss_policy = -torch.min(
                    ratio * mb_advantages,
                    torch.clamp(ratio, 1.0 - self.hyperparams.clip, 1.0 + self.hyperparams.clip) * mb_advantages
                ).mean() - self.hyperparams.entropy_coefficient * entropy

                # Compute the value loss
                loss_value = F.smooth_l1_loss(self.critic(mb_states), mb_targets)

                # Backpropagate and update actor parameters
                self.optimizers['actor'].zero_grad() # type: ignore
                loss_policy.backward()
                clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.optimizers['actor'].step()
                
                # Backpropagate and update critic parameters
                self.optimizers['critic'].zero_grad() # type: ignore
                loss_value.backward()
                clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optimizers['critic'].step()

                # Accumulate losses and entropy
                total_loss_actor += loss_policy.item()
                total_loss_critic += loss_value.item()
                total_entropy += entropy.item()

                # Step learning rate schedulers
                self.optimizers['critic_scheduler'].step()
                self.optimizers['actor_scheduler'].step()

                # Update target network
                # for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                #     target_param.data.copy_(self.hyperparams.tau * param.data + (1 - self.hyperparams.tau) * target_param.data)


        # Compute average losses and entropy
        total_loss_actor /= (num_rounds * states.size(0) / mini_batch_size)
        total_loss_critic /= (num_rounds * states.size(0) / mini_batch_size)
        total_entropy /= (num_rounds * states.size(0) / mini_batch_size)

        return {
            "Actor loss": to_tensor(total_loss_actor),
            "Critic loss": to_tensor(total_loss_critic),
            "Entropy": to_tensor(total_entropy),
            "Train Rewards": rewards.mean()
        }

    def create_lr_lambda(self, initial_lr: float, final_lr: float, constant_steps: int, max_steps: int):
        return lambda step: (
            initial_lr if step < constant_steps 
            else initial_lr - ((min(step, max_steps) - constant_steps) / (max_steps - constant_steps)) * (initial_lr - final_lr) 
            if step < max_steps 
            else final_lr
        )

    def get_optimizers(self) -> Dict[str, Optimizer | lr_scheduler.LRScheduler]:
        actor_optimizer = AdamW(self.actor.parameters(), lr=self.hyperparams.policy_learning_rate)
        critic_optimizer = AdamW(self.critic.parameters(), lr=self.hyperparams.value_learning_rate)
        actor_scheduler = torch.optim.lr_scheduler.LambdaLR(
            actor_optimizer, 
            lr_lambda=self.create_lr_lambda(
                1.0,  # Maximum multiplicative factor
                1.0 / 10.0,  # Minimum multiplicative factor
                10000,
                1000000
            )
        )
        critic_scheduler = torch.optim.lr_scheduler.LambdaLR(
            critic_optimizer, 
            lr_lambda=self.create_lr_lambda(
                1.0,  # Maximum multiplicative factor
                1.0 / 10.0,  # Minimum multiplicative factor
                10000,
                1000000
            )
        )

        return {
            "actor": AdamW(self.actor.parameters(), lr=self.hyperparams.policy_learning_rate),
            "critic": AdamW(self.critic.parameters(), lr=self.hyperparams.value_learning_rate),
            "actor_scheduler": actor_scheduler,
            "critic_scheduler": critic_scheduler
        }
    
    def state_dict(self)-> Dict[str,Dict]:
        return {
            "actor": self.actor.state_dict()
        }
    
    def load(self, location: str)->None:
        state_dicts = torch.load(location)
        self.actor.load_state_dict(state_dicts["actor"])



class PPO(Agent):
    def __init__(
            self,
        observation_space: Space,
            action_space: Space,
            hyperparams: PPOParams, 
            device = torch.device("cpu")
        ):

        super().__init__(
            observation_space,
            action_space,
            hyperparams,
            device
        )

        self.hyperparams = hyperparams
        self.state_size = self.observation_space.shape[-1] # type: ignore
        if self.is_continous():
            self.num_actions = self.action_space.shape[-1] # type: ignore
            self.action_min = float(self.action_space.low_repr) # type: ignore
            self.action_max = float(self.action_space.high_repr) # type: ignore
            self.actor = GaussianGradientPolicyV2(self.state_size, self.num_actions, self.hidden_size, device=device)
            self.critic = ValueNetwork(self.state_size, self.hidden_size, device)
            self.target_critic = ValueNetwork(self.state_size, self.hidden_size, device)

            # make target network initially the same parameters
            self.target_critic.load_state_dict(self.critic.state_dict())
            
        else:
            raise NotImplementedError
        
        self.optimizers = self.get_optimizers()
        
    def get_actions(self, state: torch.Tensor, eval=False)->tuple[Tensor, Tensor]:
        if self.is_continous():
            mean, std = self.actor(state)
            mean = self.rescaleAction(mean, self.action_min, self.action_max)

            if eval:
                return mean, torch.concat((mean, std), dim=-1)
            else:
                normal = torch.distributions.Normal(mean, std) 
                return normal.sample(), torch.concat((mean, std), dim=-1)
        else:
            raise NotImplementedError
        
    
    def learn(self, batch: list[Transition], num_envs: int, batch_size: int)->dict[str, Tensor]:
    
        # Reshape batch to gathered lists 
        states, actions, next_states, rewards, dones, other = map(torch.stack, zip(*batch))

        # Reshape data
        num_samples = batch_size
        states = states.reshape(num_samples, -1).to(device=self.device, dtype=torch.float32)
        actions = actions.reshape(num_samples, -1).to(device=self.device, dtype=torch.float32 if self.is_continous() else torch.int64)
        next_states = next_states.reshape(num_samples, -1).to(device=self.device, dtype=torch.float32)
        rewards = rewards.reshape(num_samples, -1).to(device=self.device, dtype=torch.float32)
        dones = dones.reshape(num_samples, -1).to(device=self.device, dtype=torch.float32)
        other = other.reshape(num_samples, -1).to(device=self.device, dtype=torch.float32)
        # rewards_normalized = (rewards - rewards.mean()) / (rewards.std() + self.eps)

        if self.is_continous():
            state_values = self.critic(states)

            with torch.no_grad():
                next_state_values = self.target_critic(next_states)
                target = rewards + (self.hyperparams.gamma * (1.0 - dones) * next_state_values)
                advantages = target - state_values

            # Critic loss
            critic_loss = F.smooth_l1_loss(state_values, target)

            # Policy loss and entropy
            new_loc, new_scale = self.actor(states)
            new_dist = Normal(self.rescaleAction(new_loc, self.action_min, self.action_max), new_scale)
            new_log_probs = new_dist.log_prob(actions).sum(dim=-1, keepdim=True)

            prev_loc, prev_scale = other.chunk(2, dim=-1)
            prev_dist = Normal(self.rescaleAction(prev_loc, self.action_min, self.action_max), prev_scale)
            prev_log_probs = prev_dist.log_prob(actions).sum(dim=-1, keepdim=True)

            rho = torch.exp(new_log_probs - prev_log_probs)
            policy_loss = -torch.min(
                rho * advantages,
                rho.clip(1 - self.hyperparams.clip, 1 + self.hyperparams.clip) * advantages
            )
            
            entropy = new_dist.entropy().sum(dim=-1, keepdim=True)
            loss = (policy_loss - self.hyperparams.entropy_coefficient * entropy).mean()

            # Optimize the models
            self.optimizers['actor'].zero_grad()
            loss.backward()
            clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.optimizers['actor'].step()

            self.optimizers['critic'].zero_grad()
            critic_loss.backward()
            clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizers['critic'].step()

            # Update target network
            for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                target_param.data.copy_(self.hyperparams.tau * param.data + (1 - self.hyperparams.tau) * target_param.data)

            return {
                "Actor loss": policy_loss.mean(),
                "Critic loss": critic_loss.mean(),
                "Entropy": entropy.mean(),
                "Train Rewards": rewards.mean()
            }
        else:
            raise NotImplementedError
    
    def get_optimizers(self) -> Dict[str, Optimizer]:
        return {
            "actor": AdamW(self.actor.parameters(), lr=self.hyperparams.policy_learning_rate),
            "critic": AdamW(self.critic.parameters(), lr=self.hyperparams.value_learning_rate),
        }
    
    def state_dict(self)-> Dict[str,Dict]:
        return {
            "actor": self.actor.state_dict()
        }
    
    def load(self, location: str)->None:
        state_dicts = torch.load(location)
        self.actor.load_state_dict(state_dicts["actor"])


class A2C(Agent):
    def __init__(
            self,
            observation_space,
            action_space,
            hyperparams: A2CParams, 
            device = torch.device("cpu")
        ):

        super().__init__(
            observation_space,
            action_space,
            hyperparams,
            device = device
        )

        self.hyperparams = hyperparams
        self.state_size = self.observation_space.shape[-1] # type: ignore
        if self.is_continous():
            self.num_actions = self.action_space.shape[-1] # type: ignore
            self.action_min = float(self.action_space.low_repr) # type: ignore
            self.action_max = float(self.action_space.high_repr) # type: ignore
            self.actor = GaussianGradientPolicyV2(self.state_size, self.num_actions, self.hidden_size, device=device)
            self.critic = ValueNetwork(self.state_size, self.hidden_size, device)
            self.target_critic = ValueNetwork(self.state_size, self.hidden_size, device)

            # make target network initially the same parameters
            self.target_critic.load_state_dict(self.critic.state_dict())
            
        else:
            raise NotImplementedError
        
        self.optimizers = self.get_optimizers()
        
    def get_actions(self, state: torch.Tensor, eval=False)->tuple[Tensor, Tensor]:
        if self.is_continous():
            mean, std = self.actor(state)
            mean = self.rescaleAction(mean, self.action_min, self.action_max)

            if eval:
                return mean, torch.concat((mean, std), dim=-1)
            else:
                normal = torch.distributions.Normal(mean, std) 
                return normal.sample(), torch.concat((mean, std), dim=-1)
        else:
            raise NotImplementedError
        
    
    def learn(self, batch: list[Transition], num_envs: int, batch_size: int)->dict[str, Tensor]:
        
        # Reshape batch to gathered lists 
        states, actions, next_states, rewards, dones, other = map(torch.stack, zip(*batch))

        # Reshape data
        states = states.reshape(batch_size, -1).to(device=self.device, dtype=torch.float32)
        actions = actions.reshape(batch_size, -1).to(device=self.device, dtype=torch.float32 if self.is_continous() else torch.int64)
        next_states = next_states.reshape(batch_size, -1).to(device=self.device, dtype=torch.float32)
        rewards = rewards.reshape(batch_size, -1).to(device=self.device, dtype=torch.float32)
        dones = dones.reshape(batch_size, -1).to(device=self.device, dtype=torch.float32)
        other = other.reshape(batch_size, -1).to(device=self.device, dtype=torch.float32)
        rewards_normalized = (rewards - rewards.mean()) / (rewards.std() + self.eps)

        if self.is_continous():
            state_values = self.critic(states)

            with torch.no_grad():
                next_state_values = self.target_critic(next_states)
                target = rewards_normalized + (self.hyperparams.gamma * (1.0 - dones) * next_state_values)
                advantages = target - state_values

            # Critic loss
            critic_loss = F.mse_loss(state_values, target)

            # Policy loss and entropy
            loc, scale = self.actor(states)
            dist = Normal(self.rescaleAction(loc, self.action_min, self.action_max), scale)
            log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
            entropy = dist.entropy().sum(dim=-1, keepdim=True)
            pg_loss = -log_probs * advantages # Negative because of SGA (not SGD)
            loss = (pg_loss - self.hyperparams.entropy_coefficient * entropy).mean()

            # Optimize the models
            self.optimizers['actor'].zero_grad()
            loss.backward()
            self.optimizers['actor'].step()

            self.optimizers['critic'].zero_grad()
            critic_loss.backward()
            self.optimizers['critic'].step()

            # Update target network
            for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                target_param.data.copy_(self.hyperparams.tau * param.data + (1 - self.hyperparams.tau) * target_param.data)

            return {
                "Actor loss": pg_loss.mean(),
                "Critic loss": critic_loss.mean(),
                "Entropy": entropy.mean(),
                "Train Rewards": rewards.mean()
            }
        else:
            raise NotImplementedError
    
    def get_optimizers(self) -> Dict[str, Optimizer]:
        return {
            "actor": AdamW(self.actor.parameters(), lr=self.hyperparams.policy_learning_rate),
            "critic": AdamW(self.critic.parameters(), lr=self.hyperparams.value_learning_rate),
        }
    
    def state_dict(self)-> Dict[str,Dict]:
        return {
            "actor": self.actor.state_dict()
        }
    
    def load(self, location: str)->None:
        state_dicts = torch.load(location)
        self.actor.load_state_dict(state_dicts["actor"])


class REINFORCE(Agent):
    def __init__(
            self,
            observation_space,
            action_space,
            hyperparams: REINFORCEParams, 
            device = torch.device("cpu")
        ):

        super().__init__(
            observation_space,
            action_space,
            hyperparams,
            device
        )

        self.state_size = self.observation_space.shape[-1] # type: ignore
        if self.is_discrete():
            if self.is_multi_discrete():
                self.num_actions = self.action_space.nvec[0] # type: ignore
            else:
                self.num_actions = self.action_space.n # type: ignore
            self.policy = DiscreteGradientPolicy(self.state_size, self.num_actions, self.hidden_size, device)
            self.value = ValueNetwork(self.state_size, self.hidden_size, device)
        elif self.is_continous():
            self.num_actions = self.action_space.shape[-1] # type: ignore
            self.action_min = float(self.action_space.low_repr) # type: ignore
            self.action_max = float(self.action_space.high_repr) # type: ignore
            self.policy = GaussianGradientPolicy(self.state_size, self.num_actions, self.hidden_size, device=device)
            self.value = ValueNetwork(self.state_size, self.hidden_size, device)
        else:
            raise NotImplementedError
        

        self.optimizers = self.get_optimizers()
        
    def get_actions(self, state: torch.Tensor, eval=False)->tuple[Tensor, Tensor]:
        if self.is_discrete():
            if eval:
                action_probs = self.policy(state)
                return torch.argmax(action_probs), action_probs
            else:
                probas = self.policy(state)
                return torch.multinomial(probas, 1).int(), probas
        elif self.is_continous():
            mean, std = self.policy(state)

            if eval:
                return mean, torch.concat((mean, std), dim=-1)
            else:
                normal = torch.distributions.Normal(mean + self.eps, std + self.eps) 
                return normal.sample(), torch.concat((mean, std), dim=-1)
        else:
            raise NotImplementedError
        
    
    def learn(self, batch: list[Transition], num_envs: int, batch_size: int)->dict[str, Tensor]:
        
        # Reshape batch to gathered lists 
        states, actions, _, rewards, dones, _ = map(np.stack, zip(*batch))
        dones.astype(np.float32)
        returns = np.array(self.calc_returns(rewards, dones, num_envs, batch_size), dtype=np.float32)

        # Reshape data
        num_samples = batch_size * num_envs
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
            loss = torch.mean(pg_loss - (self.hyperparams.entropy_coefficient * entropy))

            # Value loss
            values = self.value(states)
            value_loss = F.smooth_l1_loss(returns_scaled, values)

            # Optimize the models
            self.optimizers['policy'].zero_grad()
            loss.backward()
            self.optimizers['policy'].step()

            self.optimizers['value'].zero_grad()
            value_loss.backward()
            self.optimizers['value'].step()
            
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
            loss = torch.mean(pg_loss - (self.hyperparams.entropy_coefficient * entropy))

            # Value loss
            value_loss = torch.nn.functional.mse_loss(self.value(states), returns_scaled)

            # Optimize the models
            self.optimizers['policy'].zero_grad()
            loss.backward()
            self.optimizers['policy'].step()

            self.optimizers['value'].zero_grad()
            value_loss.backward()
            self.optimizers['value'].step()

            return {
                "Policy loss": pg_loss.mean(),
                "Value loss": value_loss.mean(),
                "Entropy": entropy.mean(),
                "Discounted returns": returns.mean()
            }
        else:
            raise NotImplementedError
    
    def get_optimizers(self) -> Dict[str, Optimizer]:
        return {
            "policy": AdamW(self.policy.parameters(), lr=self.hyperparams.policy_learning_rate),
            "value": AdamW(self.value.parameters(), lr=self.hyperparams.value_learning_rate),
        }
    
    def state_dict(self)-> Dict[str,Dict]:
        return {
            "policy": self.policy.state_dict()
        }
    
    def load(self, location: str)->None:
        state_dicts = torch.load(location)
        self.policy.load_state_dict(state_dicts["policy"])
