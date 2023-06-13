from gymnasium.spaces import Discrete, MultiDiscrete, Box, Space
import numpy as np

# Torch imports
import torch
from typing import Dict, Union
from torch import Tensor
from torch.optim import Optimizer, AdamW, Adam 
from torch.distributions import Normal
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.utils.clip_grad import clip_grad_norm_
from itertools import chain

# Local imports
from Datasets import Transition
from Configurations import *
from Policies import *
from Networks import *
from Utilities import to_tensor, RewardNormalizer

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
            self.actor = GaussianGradientPolicy(
                self.state_size, 
                self.num_actions, 
                64,
                device=device
            )
            self.critic = ValueNetwork(
                self.state_size, 
                self.hidden_size, 
                device
            )
            self.old_critic = ValueNetwork(
                self.state_size, 
                self.hidden_size, 
                device
            )

            # make old critic network initially the same parameters
            self.old_critic.load_state_dict(self.critic.state_dict())
            
        else:
            raise NotImplementedError
        
        self.optimizers = self.get_optimizers()
        self.update_count = 0

        self.reward_normalizer = RewardNormalizer()
        self.advantage_normalizer = RewardNormalizer()

    def get_actions(self, state: torch.Tensor, eval=False)->tuple[Tensor, Tensor]:
        if self.is_continous():
            mean, std = self.actor(state)
            normal = torch.distributions.Normal(mean, std) 

            if eval:
                action = mean
            else:
                action = normal.sample()

            # Clip the action to the valid range
            action = action.clamp(-1.0, 1.0)
    
            return action, normal.log_prob(action).sum(dim=-1)
        else:
            raise NotImplementedError
        
    def compute_gae_and_targets_truncated_time_horizon(self, rewards, dones, truncs, values, next_values, batch_size, time_horizon, gamma=0.99, lambda_=0.95):

        """
            Wraps compute_gae_and_targets() method. Mean't to compute targets and adantages when batch sizes are very large.
            This prevents advantages from exploding in size. Say if you have a single environment with 
            a batch size of 2048,  then this function will chunk that batch into several "time_horizon" sized mini-batches, 
            computing GAE on each of those mini-batches. These are then concatenated and returned. This 
        """

        # Split batch into smaller batches
        num_mini_batches = batch_size // time_horizon
        targets_all = []
        advantages_all = []
        
        for i in range(num_mini_batches):
            start = i * time_horizon
            end = (i + 1) * time_horizon
            mb_rewards = rewards[:, start:end]
            mb_dones = dones[:, start:end]
            mb_truncs = truncs[:, start:end]
            mb_values = values[:, start:end]
            mb_next_values = next_values[:, start:end]
            
            targets, advantages = self.compute_gae_and_targets(
                mb_rewards, 
                mb_dones, 
                mb_truncs, 
                mb_values, 
                mb_next_values,
                time_horizon, 
                gamma,
                lambda_
            )
            
            targets_all.append(targets)
            advantages_all.append(advantages)

        targets_all = torch.cat(targets_all, dim=1)
        advantages_all = torch.cat(advantages_all, dim=1)

        return targets_all, advantages_all

        
    def compute_gae_and_targets(self, rewards, dones, truncs, values, next_values, batch_size, gamma=0.99, lambda_=0.95):
        """
        Compute GAE and bootstrapped targets for PPO.

        :param rewards: (torch.Tensor) Rewards.
        :param dones: (torch.Tensor) Done flags.
        :param truncs: (torch.Tensor) Truncation flags.
        :param values: (torch.Tensor) State values.
        :param next_values: (torch.Tensor) Next state values.
        :param num_envs: (int) Number of environments.
        :param batch_size: (int) Batch size.
        :param gamma: (float) Discount factor.
        :param lambda_: (float) GAE smoothing factor.
        :param ema_decay: (float) Decay factor for EMA calculation.
        :return: Bootstrapped targets and advantages.

        The λ parameter in the Generalized Advantage Estimation (GAE) algorithm acts as a trade-off between bias and variance in the advantage estimation.
        When λ is close to 0, the advantage estimate is more biased, but it has less variance. It would rely more on the current reward and less on future rewards. This could be useful when your reward signal is very noisy because it reduces the impact of that noise on the advantage estimate.
        On the other hand, when λ is close to 1, the advantage estimate has more variance but is less biased. It will take into account a longer sequence of future rewards.
        """
        advantages = torch.zeros_like(rewards)
        last_gae_lam = 0

        for t in reversed(range(batch_size)):
            non_terminal = 1.0 - torch.clamp(dones[:, t] - truncs[:, t], 0.0, 1.0)
            delta = (rewards[:, t] + (gamma * next_values[:, t] * non_terminal)) - values[:, t]
            last_gae_lam = delta + (gamma * lambda_ * non_terminal * last_gae_lam)
            advantages[:, t] = last_gae_lam * non_terminal

        # Compute bootstrapped targets by adding unnormalized advantages to values
        targets = values + advantages

        # Normalize the advantages for the policy update
        advantages = self.advantage_normalizer.update(advantages)
        # advantages = (advantages - advantages.mean()) / (advantages.std() + self.eps)

        return targets, advantages
    

    def learn(self, batch: list[Transition], num_envs: int, batch_size: int, num_rounds: int = 4, mini_batch_size: int = 32) -> dict[str, Tensor]:
        self.update_count += 1

        # Reshape batch to gathered lists
        states, actions, next_states, rewards, dones, truncs, other = map(torch.stack, zip(*batch))

        # Reshape and send to device
        states = states.permute((1, 0, 2)).to(device=self.device, dtype=torch.float32).contiguous()
        next_states = next_states.permute((1, 0, 2)).to(device=self.device, dtype=torch.float32).contiguous()
        actions = actions.permute((1, 0, 2)).to(device=self.device, dtype=torch.float32).contiguous()
        dones = dones.permute((1, 0)).to(device=self.device, dtype=torch.float32).contiguous()
        truncs = truncs.permute((1, 0)).to(device=self.device, dtype=torch.float32).contiguous()
        prev_log_probs = other.permute((1, 0)).to(device=self.device, dtype=torch.float32).contiguous()
        rewards = rewards.permute((1, 0)).to(device=self.device, dtype=torch.float32).contiguous()

        # Update running mean and variance and normalize rewards
        if self.hyperparams.use_moving_average_reward:
            rewards = self.reward_normalizer.update(rewards)

        # Compute values for states and next states
        with torch.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)
            # old_values = self.old_critic(next_states)

            # Calculate advantages with GAE
            targets, advantages_all = self.compute_gae_and_targets(
                rewards.unsqueeze(-1), 
                dones.unsqueeze(-1), 
                truncs.unsqueeze(-1), 
                values, 
                next_values, 
                batch_size, 
                self.hyperparams.gamma, 
                self.hyperparams.gae_lambda
            )

        # Flatten states, actions, log probabilities, advantages, and targets
        states = states.view(-1, states.size(-1))
        values = values.view(-1)
        # old_values = old_values.view(-1, 1)
        prev_log_probs = prev_log_probs.view(-1)
        advantages_all = advantages_all.view(-1)
        actions = actions.view(-1, actions.size(-1))
        targets = targets.view(-1)

        # Initialize loss and entropy accumulators
        total_loss_combined, total_loss_actor, total_loss_critic, total_entropy = to_tensor(0, requires_grad=False), to_tensor(0, requires_grad=False), to_tensor(0, requires_grad=False), to_tensor(0, requires_grad=False)

        # Perform multiple rounds of learning
        for _ in range(num_rounds):
            
            # Generate a random order of mini-batches
            mini_batch_start_indices = list(range(0, states.size(0), mini_batch_size))
            np.random.shuffle(mini_batch_start_indices)
            
            # Process each mini-batch
            for start_idx in mini_batch_start_indices:
                
                # Mini-batch indices
                end_idx = min(start_idx + mini_batch_size, states.size(0))
                ids = slice(start_idx, end_idx)  # using a slice instead of a list of indices

                # Extract mini-batch data
                with torch.no_grad():
                    # mb_old_values = old_values[ids]
                    mb_states = states[ids]
                    mb_actions = actions[ids]
                    mb_old_log_probs = prev_log_probs[ids]
                    mb_advantages = advantages_all[ids]
                    mb_targets = torch.unsqueeze(targets[ids], dim=-1)

                # Compute policy distribution parameters
                loc, scale = self.actor(mb_states)
                dist = Normal(loc, scale)
                log_probs = dist.log_prob(mb_actions).sum(dim=-1)

                # Compute entropy bonus
                entropy = dist.entropy().mean(dim=-1)

                # Compute the policy loss using the PPO clipped objective
                ratio = torch.exp(log_probs - mb_old_log_probs)
                loss_policy = (-torch.min(
                    ratio * mb_advantages,
                    ratio.clip(1.0 - self.hyperparams.clip, 1.0 + self.hyperparams.clip) * mb_advantages
                ) - (self.hyperparams.entropy_coefficient * entropy)).mean()

                # Compute the value loss with clipping
                # clipped_values = mb_old_values + (predicted_values - mb_old_values).clamp(-self.hyperparams.clipped_value_loss_eps, self.hyperparams.clipped_value_loss_eps)
                # loss_value1 = F.mse_loss(predicted_values, mb_targets)
                # loss_value2 = F.mse_loss(clipped_values, mb_targets)
                # loss_value = torch.min(loss_value1, loss_value2)
                loss_value = F.mse_loss(self.critic(mb_states), mb_targets)

                if self.hyperparams.combined_optimizer:
                    # Combine the losses
                    total_loss = self.hyperparams.policy_loss_weight * loss_policy + self.hyperparams.value_loss_weight * loss_value

                    # Backpropagate the total loss
                    total_loss.backward()

                    # Perform a single optimization step, clip gradients before step
                    clip_grad_norm_(self.optimizers["combined"].param_groups[0]['params'], self.hyperparams.max_grad_norm)
                    self.optimizers["combined"].step()

                    # Clear the gradients
                    self.optimizers["combined"].zero_grad()
                
                else:
                    # Optimize the models
                    loss_policy.backward()
                    loss_value.backward()
                    
                    clip_grad_norm_(self.optimizers["actor"].param_groups[0]['params'], self.max_grad_norm)
                    clip_grad_norm_(self.optimizers["critic"].param_groups[0]['params'], self.max_grad_norm)
                    
                    self.optimizers['actor'].step()
                    self.optimizers['critic'].step()
                    
                    self.optimizers['critic'].zero_grad()
                    self.optimizers['actor'].zero_grad()
                

                # Accumulate losses and entropy
                with torch.no_grad():
                    total_loss_combined += (loss_policy + loss_value).cpu()
                    total_loss_actor += loss_policy.cpu()
                    total_loss_critic += loss_value.cpu()
                    total_entropy += entropy.mean().cpu()

        # Update the old critic network by copying the current critic network weights
        # if self.update_count % 5 == 0:
        #     self.old_critic.load_state_dict(self.critic.state_dict())

        # Compute average losses and entropy
        total_loss_actor /= (num_rounds * states.size(0) / mini_batch_size)
        total_loss_critic /= (num_rounds * states.size(0) / mini_batch_size)
        total_entropy /= (num_rounds * states.size(0) / mini_batch_size)
        total_loss_combined /= (num_rounds * states.size(0) / mini_batch_size)
        return {
            "Actor loss": total_loss_actor,
            "Critic loss": total_loss_critic,
            "Total loss": total_loss_combined,
            "Entropy": total_entropy,
            "Advantages" : advantages_all.mean(),
            "Train Rewards": rewards.mean(),
            "Total Batch Rewards": rewards.sum()
        }

    def create_lr_lambda(self, initial_lr: float, final_lr: float, constant_steps: int, max_steps: int):
        return lambda step: (
            initial_lr if step < constant_steps 
            else initial_lr - ((min(step, max_steps) - constant_steps) / (max_steps - constant_steps)) * (initial_lr - final_lr) 
            if step < max_steps 
            else final_lr
        )

    def get_optimizers(self) -> Dict[str, Optimizer | lr_scheduler.LRScheduler]:

        if self.hyperparams.combined_optimizer:
        
            # self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            #     self.optimizer, 
            #     lr_lambda=self.create_lr_lambda(
            #         1.0,  # Maximum multiplicative factor
            #         1.0 / 20.0,  # Minimum multiplicative factor
            #         5000,
            #         100000
            #     )
            # )

            return {
                "combined": AdamW(
                    chain(self.actor.parameters(), self.critic.parameters()), 
                    lr=self.hyperparams.policy_learning_rate
                )
            }
        
        else:
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
            self.actor = GaussianGradientPolicy(self.state_size, self.num_actions, self.hidden_size, device=device)
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
            # mean = self.rescaleAction(mean, self.action_min, self.action_max)

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
            self.actor = GaussianGradientPolicy(self.state_size, self.num_actions, self.hidden_size, device=device)
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
