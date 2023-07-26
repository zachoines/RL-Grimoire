from gymnasium.spaces import Discrete, MultiDiscrete, Box, Space
import numpy as np

# Torch imports
import torch
from typing import Dict, List
from torch import Tensor
from torch.optim import Optimizer, AdamW
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
from Utilities import to_tensor, winsorize, SlidingWindowNormalizer, RunningMeanStdNormalizer

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

    def get_actions(self, state: torch.Tensor, eval: bool = False, **kwargs)->tuple[Tensor, Tensor]:
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
    
    def save_train_state(self):
        return None

    def restore_train_state(self, state):
        pass
    
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
            non_terminal = 1.0 - torch.clamp(dones[t, :] - truncs[t, :], 0.0, 1.0)
            delta = (rewards[t, :] + (gamma * next_values[t, :] * non_terminal)) - values[t, :]
            last_gae_lam = delta + (gamma * lambda_ * non_terminal * last_gae_lam)
            advantages[t, :] = last_gae_lam * non_terminal

        # Compute bootstrapped targets by adding unnormalized advantages to values 
        targets = values + advantages
        return targets, advantages
    
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

            self.actor = GaussianGradientTransformerPolicyV1(
                self.state_size, 
                self.num_actions, 
                self.hidden_size,
                device=device
            )
            
            self.critic = ValueNetworkTransformerV1(
                self.state_size, 
                self.hidden_size, 
                device=device
            )

            if self.hyperparams.value_loss_clipping:
                self.old_critic = ValueNetworkTransformer(
                    self.state_size, 
                    self.hidden_size, 
                    device=device
                )

                # make old critic network initially the same parameters
                self.old_critic.load_state_dict(self.critic.state_dict())
                
        else:
            raise NotImplementedError
        
        # Optimizers and schedulers
        self.optimizers = None
        self.optimizers = self.get_optimizers()
        self.update_count = 0
        
        # Normaliers
        self.reward_normalizer = Normalizer(device=device)

    def get_actions(self, state: torch.Tensor, eval=False, **kwargs)->tuple[Tensor, Tensor]:
        if self.is_continous():
            mean, std = self.actor(state)
            mean, std = torch.squeeze(mean), torch.squeeze(std)
            normal = torch.distributions.Normal(mean, std) 

            if eval:
                action = mean
            else:
                action = normal.sample()
            action = action.clip(self.action_min, self.action_max)
            return action, normal.log_prob(action).sum(dim=-1)
        else:
            raise NotImplementedError
          
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
        return targets, advantages

    def learn(self, batch: list[Transition], num_envs: int, batch_size: int) -> dict[str, Tensor]:
        self.update_count += 1
        num_rounds = self.hyperparams.num_rounds
        mini_batch_size = self.hyperparams.mini_batch_size

        # Reshape batch to gathered lists
        states, actions, next_states, rewards, dones, truncs, prev_log_probs = map(torch.stack, zip(*batch))

        # Reshape and send to device
        states = states.permute((1, 0, 2)).to(device=self.device, dtype=torch.float32).contiguous()
        next_states = next_states.permute((1, 0, 2)).to(device=self.device, dtype=torch.float32).contiguous()
        actions = actions.permute((1, 0, 2)).to(device=self.device, dtype=torch.float32).contiguous()
        dones = dones.permute((1, 0)).to(device=self.device, dtype=torch.float32).contiguous()
        truncs = truncs.permute((1, 0)).to(device=self.device, dtype=torch.float32).contiguous()
        prev_log_probs = prev_log_probs.permute((1, 0)).to(device=self.device, dtype=torch.float32).contiguous()
        rewards = rewards.permute((1, 0)).to(device=self.device, dtype=torch.float32).contiguous()
        batch_rewards = rewards.clone()

        # Update running mean and variance and normalize rewards
        if self.hyperparams.use_moving_average_reward:
            rewards = self.reward_normalizer.update(rewards)

        # Compute values for states and next states
        with torch.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)
            old_values = self.old_critic(next_states) if self.hyperparams.value_loss_clipping else None

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

        # Initialize loss and entropy accumulators
        total_loss_combined, total_loss_actor, total_loss_critic, total_entropy = to_tensor(0, requires_grad=False), to_tensor(0, requires_grad=False), to_tensor(0, requires_grad=False), to_tensor(0, requires_grad=False)

        # Perform multiple rounds of learning
        for _ in range(num_rounds):
            
            # Generate a random order of mini-batches
            mini_batch_start_indices = list(range(0, states.size(1), mini_batch_size))
            np.random.shuffle(mini_batch_start_indices)
            
            # Process each mini-batch
            for start_idx in mini_batch_start_indices:
                
                # Mini-batch indices
                end_idx = min(start_idx + mini_batch_size, states.size(1))
                ids = slice(start_idx, end_idx)  # using a slice instead of a list of indices

                # Extract mini-batch data
                with torch.no_grad():
                    mb_old_values = old_values[:, ids, :] if self.hyperparams.value_loss_clipping else None
                    mb_states = states[:, ids, :]
                    mb_actions = actions[:, ids, :]
                    mb_old_log_probs = prev_log_probs[:, ids]
                    mb_advantages = torch.squeeze(advantages_all[:, ids, :])
                    mb_targets = targets[:, ids, :]

                # Compute policy distribution parameters
                loc, scale = self.actor(mb_states)
                dist = Normal(loc, scale)
                log_probs = dist.log_prob(mb_actions).sum(dim=-1)

                # Compute entropy bonus
                entropy = torch.mean(dist.entropy())

                # Compute the policy loss using the PPO clipped objective
                ratio = torch.exp(log_probs - mb_old_log_probs)
                loss_policy = -torch.mean(
                    torch.min(
                        ratio * mb_advantages,
                        ratio.clip(1.0 - self.hyperparams.clip, 1.0 + self.hyperparams.clip) * mb_advantages
                    )
                )

                # Compute the value loss with clipping
                predicted_values = self.critic(mb_states)
                loss_value: torch.Tensor
                if self.hyperparams.value_loss_clipping:
                    clipped_values = mb_old_values + (predicted_values - mb_old_values).clamp(-self.hyperparams.clipped_value_loss_eps, self.hyperparams.clipped_value_loss_eps)
                    loss_value1 = torch.square(predicted_values - mb_targets)
                    loss_value2 = torch.square(clipped_values - mb_targets)
                    loss_value = 0.5 * torch.max(loss_value1, loss_value2).mean()
                else:
                    loss_value = F.smooth_l1_loss(predicted_values, mb_targets)

                if self.hyperparams.combined_optimizer:
                    
                    # Combine the losses 
                    total_loss = ((self.hyperparams.policy_loss_weight * loss_policy) - (self.hyperparams.entropy_coefficient * entropy)) + (self.hyperparams.value_loss_weight * loss_value)

                    # Backpropagate the total loss
                    total_loss.backward()

                    # Perform a single optimization step, clip gradients before step
                    clip_grad_norm_(self.optimizers["combined"].param_groups[0]['params'], self.hyperparams.max_grad_norm)
                    self.optimizers["combined"].step()
                    if self.hyperparams.use_lr_scheduler:
                        self.optimizers["combined_scheduler"].step()

                    # Clear the gradients
                    self.optimizers["combined"].zero_grad()

                    # Accumulate losses and entropy
                    with torch.no_grad():
                        total_loss_combined += total_loss.cpu()
                        total_loss_actor += loss_policy.cpu()
                        total_loss_critic += loss_value.cpu()
                        total_entropy += entropy.mean().cpu()
                
                else:
                    loss_policy_with_entropy_bonus = loss_policy - (self.hyperparams.entropy_coefficient * entropy.detach())

                    # Optimize the models
                    loss_value.backward()
                    loss_policy_with_entropy_bonus.backward()
                    
                    clip_grad_norm_(self.optimizers["actor"].param_groups[0]['params'], self.max_grad_norm)
                    clip_grad_norm_(self.optimizers["critic"].param_groups[0]['params'], self.max_grad_norm)
                    
                    self.optimizers['actor'].step()
                    self.optimizers['critic'].step()
                    
                    if self.hyperparams.use_lr_scheduler:
                        self.optimizers["actor_scheduler"].step()
                        self.optimizers["critic_scheduler"].step()
                    
                    self.optimizers['critic'].zero_grad()
                    self.optimizers['actor'].zero_grad()
                
                    # Accumulate losses and entropy
                    with torch.no_grad():
                        total_loss_combined += (loss_policy + loss_value).cpu()
                        total_loss_actor += loss_policy.cpu()
                        total_loss_critic += loss_value.cpu()
                        total_entropy += entropy.mean().cpu()

        # Update the old critic network by copying the current critic network weights
        if self.hyperparams.value_loss_clipping:
            self.old_critic.load_state_dict(self.critic.state_dict())

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
            "Total Batch Rewards": batch_rewards.sum(),
            **(
                { 
                    "Combined RL Scheduler": to_tensor(self.optimizers["combined_scheduler"].get_last_lr()[0]) 
                } 
                if self.hyperparams.combined_optimizer else 
                {
                    "Actor RL Scheduler" : to_tensor(self.optimizers["actor_scheduler"].get_last_lr()[0]),
                    "Critic RL Scheduler" : to_tensor(self.optimizers["critic_scheduler"].get_last_lr()[0])
                }
            ),
        }

    def create_lr_lambda(self):
        initial_lr = self.hyperparams.lr_scheduler_max_factor
        final_lr = self.hyperparams.lr_scheduler_min_factor
        constant_steps = self.hyperparams.lr_scheduler_constant_steps
        max_steps = self.hyperparams.lr_scheduler_max_steps
        return lambda step: (
            initial_lr if step < constant_steps 
            else initial_lr - ((min(step, max_steps) - constant_steps) / (max_steps - constant_steps)) * (initial_lr - final_lr) 
            if step < max_steps 
            else final_lr
        )

    def get_optimizers(self) -> Dict[str, Optimizer | lr_scheduler.LRScheduler]:

        if self.optimizers == None:
            if self.hyperparams.combined_optimizer:
                optimizer = AdamW(
                    chain(self.actor.parameters(), self.critic.parameters()), 
                    lr=self.hyperparams.policy_learning_rate
                )

                return {
                    "combined": optimizer,
                    "combined_scheduler": torch.optim.lr_scheduler.LambdaLR(
                        optimizer, 
                        lr_lambda=self.create_lr_lambda()
                    )
                }
            else:
                actor_optimizer = AdamW(self.actor.parameters(), lr=self.hyperparams.policy_learning_rate)
                critic_optimizer = AdamW(self.critic.parameters(), lr=self.hyperparams.value_learning_rate)

                return {
                    "actor": actor_optimizer,
                    "critic": critic_optimizer,
                    "actor_scheduler": torch.optim.lr_scheduler.LambdaLR(
                        actor_optimizer, 
                        lr_lambda=self.create_lr_lambda()
                    ),
                    "critic_scheduler": torch.optim.lr_scheduler.LambdaLR(
                        critic_optimizer, 
                        lr_lambda=self.create_lr_lambda()
                    )
                }
        else:
            return self.optimizers     

    def state_dict(self)-> Dict[str,Dict]:
        return {
            "actor": self.actor.state_dict()
        }
    
    def load(self, location: str)->None:
        state_dicts = torch.load(location)
        self.actor.load_state_dict(state_dicts["actor"])

class PPO2Recurrent(Agent):
    def __init__(
            self,
            observation_space: Space,
            action_space: Space,
            hyperparams: PPO2RecurrentParams, 
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

            self.actor = GaussianGradientLSTMPolicy(
                self.state_size, 
                self.num_actions, 
                self.hidden_size,
                device=device
            )
            
            self.critic = ValueNetworkLSTM(
                self.state_size, 
                self.hidden_size, 
                device=device
            )

            if self.hyperparams.value_loss_clipping:
                self.old_critic = ValueNetworkLSTM(
                    self.state_size, 
                    self.hidden_size, 
                    device=device
                )

                self.old_critic.load_state_dict(self.critic.state_dict())
            
            if self.hyperparams.icm_module.enabled:
                self.ICM = ICMRecurrent(
                    self.state_size, 
                    self.num_actions, 
                    self.hyperparams.icm_module.hidden_size, 
                    self.hyperparams.icm_module.state_feature_size, 
                    device = self.device
                )
                
        else:
            raise NotImplementedError
        
        # RND network module
        self.rnd_module = False

        if self.rnd_module:
            self.RND = RND(
                self.state_size,
                128,
                64,
                self.device
            )
        
        # Optimizers and schedulers
        self.optimizers = None
        self.optimizers = self.get_optimizers()
        self.update_count = 0
        
        # Normaliers
        # self.reward_normalizer = SlidingWindowNormalizer(device=device)
        self.external_reward_normalizer = RunningMeanStdNormalizer(device=device)
        self.intrinsic_reward_normalizer = RunningMeanStdNormalizer(device=device)

    def save_train_state(self):
        return self.actor.get_hidden(), self.critic.get_hidden()
    
    def restore_train_state(self, state):
        actor_hidden, critic_hidden = state
        self.actor.set_hidden(actor_hidden)
        self.critic.set_hidden(critic_hidden)

    def get_actions(self, state: torch.Tensor, dones: torch.Tensor = torch.tensor([]), eval=False, **kwargs)->tuple[Tensor]:
        if self.is_continous():
            mean, std, _ = self.actor(state.unsqueeze(0), dones=dones)
            value, _ = self.critic(state.unsqueeze(0), dones=dones)
            critic_hidden, policy_hidden = self.critic.get_prev_hidden(), self.actor.get_prev_hidden()
            mean, std = torch.squeeze(mean), torch.squeeze(std)
            value = torch.squeeze(value)
            normal = torch.distributions.Normal(mean, std) 

            if eval:
                action = mean
            else:
                action = normal.sample()
            action = action.clip(self.action_min, self.action_max)
            return action, normal.log_prob(action).sum(dim=-1), policy_hidden, critic_hidden, mean, std  # type: ignore
        else:
            raise NotImplementedError

    def compute_clipped_value_loss(self, old_values, values, returns, clip_epsilon=0.2):
        """
        Compute the value loss for PPO with clipping.

        :param old_values: The old value predictions.
        :param values: The new value predictions.
        :param returns: The returns (advantages + values).
        :param clip_epsilon: The epsilon for clipping the value loss.
        :return: The value loss.
        """

        # Detach old_values so they're treated as constants
        old_values = old_values.detach()

        # Calculate the value difference
        value_diff = values - old_values

        # Clipped values
        abs_old_values = torch.abs(old_values)
        clipped_values = old_values + torch.clamp(value_diff, -clip_epsilon * abs_old_values, clip_epsilon * abs_old_values)

        # Unclipped loss
        loss_unclipped = torch.square(returns - values)

        # Clipped loss
        loss_clipped = torch.square(returns - clipped_values)

        # Take the minimum of the clipped and unclipped losses
        loss = torch.min(loss_unclipped, loss_clipped).mean()

        return loss

    def learn(self, batch: List[Transition], num_envs: int, batch_size: int) -> Dict[str, torch.Tensor]:
        self.update_count += 1
        num_rounds = self.hyperparams.num_rounds
        mini_batch_size = self.hyperparams.mini_batch_size

        with torch.no_grad():
            # Reshape batch to gathered lists
            states, actions, next_states, rewards, dones, truncs, prev_log_probs, policy_hidden, critic_hidden, locs, scales = map(
                torch.stack, zip(*batch)
            )

            total_batch_rewards = rewards.sum()
            states_plus_one = torch.cat((states, next_states[-1:, :]), dim=0)
            dones_plus_one = torch.cat((dones, torch.zeros_like(dones[-1:, :])), dim = 0)
    

        # Run the ICM module
        icm_intrinsic_reward: torch.Tensor
        icm_hidden_outputs: torch.Tensor
        if self.hyperparams.icm_module.enabled:
            # _, _, icm_intrinsic_reward = self.ICM(
            #     states_plus_one, 
            #     locs, 
            #     scales,
            #     n = self.hyperparams.icm_module.n,
            #     beta = self.hyperparams.icm_module.beta
            # )
            _, _, icm_intrinsic_reward, icm_hidden_outputs = self.ICM(
                states_plus_one, 
                locs, 
                scales,
                n = self.hyperparams.icm_module.n,
                beta = self.hyperparams.icm_module.beta
            )

        rnd_intrinsic_reward = None
        if self.rnd_module:
            rnd_intrinsic_reward = self.RND(states)
        
        with torch.no_grad():

            # Update running mean and variance and normalize rewards.
            if self.hyperparams.use_moving_average_reward:
                rewards = self.external_reward_normalizer.update(rewards)
            
            if self.hyperparams.icm_module.enabled:
                icm_intrinsic_reward = self.intrinsic_reward_normalizer.update(icm_intrinsic_reward) # type: ignore
                rewards = rewards + icm_intrinsic_reward

            if self.rnd_module:
                rewards = rewards + rnd_intrinsic_reward

            # Compute values for states and next states
            self.critic.set_hidden(critic_hidden[0, :])
            values_plus_one, _ = self.critic(states_plus_one, dones=dones_plus_one)
            values = values_plus_one[:-1]
            next_values = values_plus_one[1:]
            old_values = values.clone()

            # Calculate advantages with GAE
            targets, advantages_all = self.compute_gae_and_targets(
                rewards.unsqueeze(-1).clone(),
                dones.unsqueeze(-1).clone(), # No cloning causes error on MPS (bug in pytorch for OSX)
                truncs.unsqueeze(-1).clone(),
                values,
                next_values,
                batch_size,
                self.hyperparams.gamma,
                self.hyperparams.gae_lambda,
            )

        # Initialize loss and entropy accumulators
        total_loss_combined, total_loss_actor, total_loss_critic, total_entropy, total_rnd_loss, total_icm_forward_loss, total_icm_inverse_loss = [], [], [], [], [], [], []

        # Perform multiple rounds of learning
        for _ in range(num_rounds):

            # Generate a random order of mini-batches
            mini_batch_start_indices = list(range(0, states.size(0), mini_batch_size))
            np.random.shuffle(mini_batch_start_indices)

            # Process each mini-batch
            for start_idx in mini_batch_start_indices:

                # Mini-batch indices
                end_idx = min(start_idx + mini_batch_size, states.size(0))
                ids = slice(start_idx, end_idx)

                # Extract mini-batch data
                with torch.no_grad():
                    mb_icm_hidden_outputs = icm_hidden_outputs[ids] # type: ignore
                    mb_loc = locs[ids]
                    mb_scale = scales[ids]
                    mb_states = states[ids]
                    mb_actions = actions[ids]
                    mb_prev_log_probs = prev_log_probs[ids]
                    mb_advantages = advantages_all[ids]
                    mb_targets = targets[ids]
                    mb_policy_hidden = policy_hidden[ids]
                    mb_critic_hidden = critic_hidden[ids]
                    mb_dones = dones_plus_one[ids]
                    mb_old_values = old_values[ids]

                # Compute policy distribution parameters
                loc, scale, _ = self.actor(mb_states, mb_policy_hidden, dones=mb_dones)
                dist = Normal(loc, scale)
                log_probs = dist.log_prob(mb_actions).sum(dim=-1)

                # Compute entropy bonus
                entropy = dist.entropy().mean(dim=-1)

                # Compute the policy loss using the PPO clipped objective
                ratio = torch.exp(log_probs - mb_prev_log_probs)
                loss_policy = -torch.mean(
                        torch.min(
                            ratio * torch.squeeze(mb_advantages),
                            ratio.clip(1.0 - self.hyperparams.clip, 1.0 + self.hyperparams.clip) * torch.squeeze(mb_advantages),
                        ) - (self.hyperparams.entropy_coefficient * entropy)
                    )

                # Compute the value loss with clipping
                loss_value: torch.Tensor
                predicted_values, _ = self.critic(mb_states, mb_critic_hidden, dones=mb_dones)
                if self.hyperparams.value_loss_clipping:
                    loss_value = self.compute_clipped_value_loss(mb_old_values, predicted_values, mb_targets, self.hyperparams.clipped_value_loss_eps)
                else:
                    loss_value = F.smooth_l1_loss(predicted_values, mb_targets)

                if self.rnd_module:
                    rnd_loss = F.mse_loss(self.RND.predictor_network(mb_states), self.RND.target_network(mb_states).detach())
                    rnd_loss.backward()
                    clip_grad_norm_(self.optimizers["RND"].param_groups[0]["params"], self.max_grad_norm) # type: ignore
                    self.optimizers["RND"].step() # type: ignore
                    if self.hyperparams.use_lr_scheduler:
                        self.optimizers["RND_scheduler"].step() # type: ignore
                    self.optimizers["RND"].zero_grad() # type: ignore
                    total_rnd_loss.append(rnd_loss.cpu())

                if self.hyperparams.icm_module.enabled:
                    # forward_loss, inverse_loss, _ = self.ICM(
                    #     torch.cat((mb_states, mb_states[-1:, :]), dim=0), 
                    #     mb_loc, 
                    #     mb_scale,
                    #     n = self.hyperparams.icm_module.n,
                    #     beta = self.hyperparams.icm_module.beta
                    # )

                    forward_loss, inverse_loss, _, _ = self.ICM(
                        torch.cat((mb_states, mb_states[-1:, :]), dim=0), 
                        mb_loc, 
                        mb_scale,
                        dones_plus_one = torch.cat((mb_dones, mb_dones[-1:, :]), dim=0),
                        input_hidden = torch.cat((mb_icm_hidden_outputs, mb_icm_hidden_outputs[-1:, :]), dim=0),
                        n = self.hyperparams.icm_module.n,
                        beta = self.hyperparams.icm_module.beta
                    )

                    total_icm_forward_loss.append(forward_loss.cpu())
                    total_icm_inverse_loss.append(inverse_loss.cpu())

                    (forward_loss + inverse_loss).backward()
                    clip_grad_norm_(self.optimizers["ICM"].param_groups[0]["params"], self.max_grad_norm) # type: ignore
                    self.optimizers["ICM"].step() # type: ignore
                    if self.hyperparams.use_lr_scheduler:
                        self.optimizers["ICM_scheduler"].step() # type: ignore 
                    self.optimizers["ICM"].zero_grad() # type: ignore
               
                if self.hyperparams.combined_optimizer:

                    # Combine the losses
                    total_loss = (
                        (self.hyperparams.policy_loss_weight * loss_policy)
                        - (self.hyperparams.entropy_coefficient * entropy)
                        + (self.hyperparams.value_loss_weight * loss_value)
                    )

                    # Backpropagate the total loss
                    total_loss.backward()

                    # Perform a single optimization step, clip gradients before step
                    clip_grad_norm_(self.optimizers["combined"].param_groups[0]["params"], self.hyperparams.max_grad_norm) # type: ignore
                    self.optimizers["combined"].step() # type: ignore
                    if self.hyperparams.use_lr_scheduler:
                        self.optimizers["combined_scheduler"].step() # type: ignore

                    # Clear the gradients
                    self.optimizers["combined"].zero_grad() # type: ignore

                    # Accumulate losses and entropy
                    total_loss_combined += total_loss.cpu()
                    total_loss_actor += loss_policy.cpu()
                    total_loss_critic += loss_value.cpu()
                    total_entropy += entropy.mean().cpu()

                else:
                    # Optimize the models
                    loss_value.backward()
                    loss_policy.backward()
                        

                    clip_grad_norm_(self.optimizers["actor"].param_groups[0]["params"], self.max_grad_norm) # type: ignore
                    clip_grad_norm_(self.optimizers["critic"].param_groups[0]["params"], self.max_grad_norm) # type: ignore
                    
                    self.optimizers["actor"].step() # type: ignore
                    self.optimizers["critic"].step() # type: ignore

                    if self.hyperparams.use_lr_scheduler:
                        self.optimizers["actor_scheduler"].step() # type: ignore
                        self.optimizers["critic_scheduler"].step() # type: ignore

                    self.optimizers["critic"].zero_grad() # type: ignore
                    self.optimizers["actor"].zero_grad() # type: ignore
                    
                    # Accumulate losses and entropy
                    total_loss_combined.append((loss_policy + loss_value).cpu())
                    total_loss_actor.append(loss_policy.cpu())
                    total_loss_critic.append(loss_value.cpu())
                    total_entropy.append(entropy.mean().cpu())

        # Compute average losses and entropy
        total_loss_actor = torch.stack(total_loss_actor).mean()
        total_loss_critic = torch.stack(total_loss_critic).mean()
        total_entropy = torch.stack(total_entropy).mean()
        total_loss_combined = torch.stack(total_loss_combined).mean()
        if self.rnd_module:
            total_rnd_loss = torch.stack(total_rnd_loss).mean()

        if self.hyperparams.icm_module.enabled:
            total_icm_forward_loss = torch.stack(total_icm_forward_loss).mean()
            total_icm_inverse_loss = torch.stack(total_icm_inverse_loss).mean()
        

        return {
            "Actor loss": total_loss_actor,
            "Critic loss": total_loss_critic,
            "Total loss": total_loss_combined,
            "Entropy": total_entropy,
            "Advantages": advantages_all.mean(),
            "Train Rewards": rewards.mean(),
            "Total Batch Rewards": total_batch_rewards,
            **(
                {
                    "RND Reward" : rnd_intrinsic_reward.mean(), # type: ignore
                    "RND loss" : total_rnd_loss,
                }
                if self.rnd_module
                else {}
            ),
            **(
                {
                    "ICM Rewards": icm_intrinsic_reward.mean(), # type: ignore 
                    "ICM Forward Loss": total_icm_forward_loss, # type: ignore 
                    "ICM Inverse Loss": total_icm_inverse_loss, # type: ignore 
                } 
                if self.hyperparams.icm_module.enabled 
                else {}
            ),
            **(
                {
                    "Combined RL Scheduler": to_tensor(self.optimizers["combined_scheduler"].get_last_lr()[0]) # type: ignore
                }
                if self.hyperparams.combined_optimizer
                else 
                {
                    "Actor RL Scheduler": to_tensor(self.optimizers["actor_scheduler"].get_last_lr()[0]), # type: ignore
                    "Critic RL Scheduler": to_tensor(self.optimizers["critic_scheduler"].get_last_lr()[0]), # type: ignore
                }
            )
        }

    def create_lr_lambda(self):
        initial_lr = self.hyperparams.lr_scheduler_max_factor
        final_lr = self.hyperparams.lr_scheduler_min_factor
        constant_steps = self.hyperparams.lr_scheduler_constant_steps
        max_steps = self.hyperparams.lr_scheduler_max_steps
        return lambda step: (
            initial_lr if step < constant_steps 
            else initial_lr - ((min(step, max_steps) - constant_steps) / (max_steps - constant_steps)) * (initial_lr - final_lr) 
            if step < max_steps 
            else final_lr
        )

    def get_optimizers(self) -> Dict[str, Optimizer | lr_scheduler.LRScheduler]:

        if self.optimizers == None:
            if self.hyperparams.combined_optimizer:
                optimizer = AdamW(
                    chain(self.actor.parameters(), self.critic.parameters()), 
                    lr=self.hyperparams.policy_learning_rate
                )

                return {
                    "combined": optimizer,
                    "combined_scheduler": torch.optim.lr_scheduler.LambdaLR(
                        optimizer, 
                        lr_lambda=self.create_lr_lambda()
                    )
                }
            else:
                actor_optimizer =  AdamW(
                    self.actor.parameters(), 
                    lr=self.hyperparams.policy_learning_rate
                )
                critic_optimizer = AdamW(
                    self.critic.parameters(), 
                    lr=self.hyperparams.value_learning_rate
                )

                if self.rnd_module:
                    RND_optimizer = AdamW(
                        self.RND.parameters(), 
                        lr=self.hyperparams.value_learning_rate
                    )
                if self.hyperparams.icm_module.enabled:
                    icm_optimizer = AdamW(
                        self.ICM.parameters(), 
                        lr=self.hyperparams.icm_module.learning_rate
                    )
                return {
                    "actor": actor_optimizer,
                    "critic": critic_optimizer,
                    "actor_scheduler": torch.optim.lr_scheduler.LambdaLR(
                        actor_optimizer, 
                        lr_lambda=self.create_lr_lambda()
                    ),
                    "critic_scheduler": torch.optim.lr_scheduler.LambdaLR(
                        critic_optimizer, 
                        lr_lambda=self.create_lr_lambda()
                    ),
                    **({
                            "RND": RND_optimizer, # type: ignore 
                            "RND_scheduler": torch.optim.lr_scheduler.LambdaLR(
                                RND_optimizer, # type: ignore 
                                lr_lambda=self.create_lr_lambda()
                            ),
                        } if self.rnd_module else {}
                    ),
                    **({
                            "ICM": icm_optimizer, # type: ignore 
                            "ICM_scheduler": torch.optim.lr_scheduler.LambdaLR(
                                icm_optimizer,  # type: ignore
                                lr_lambda=self.create_lr_lambda()
                            ),
                        } if self.hyperparams.icm_module.enabled else {}
                    ) 
                }
        else:
            return self.optimizers     

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
        
    def get_actions(self, state: torch.Tensor, eval=False, **kwargs)->tuple[Tensor, Tensor]:
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
            self.target_critic.load_state_dict(self.critic.state_dict())

            if self.hyperparams.icm_module.enabled:
                self.ICM = ICM(
                    self.state_size, 
                    self.num_actions, 
                    self.hyperparams.icm_module.hidden_size, 
                    self.hyperparams.icm_module.state_feature_size, 
                    self.device
                )
            
        elif self.is_discrete():
            if self.is_multi_discrete():
                self.num_actions = self.action_space.nvec[0] # type: ignore
            else:
                self.num_actions = self.action_space.n # type: ignore
            self.actor = DiscreteGradientPolicy(self.state_size, self.num_actions, self.hidden_size, device=device)
            self.critic = ValueNetwork(self.state_size, self.hidden_size, device)
            self.target_critic = ValueNetwork(self.state_size, self.hidden_size, device)
            self.target_critic.load_state_dict(self.critic.state_dict())

            if self.hyperparams.icm_module.enabled:
                raise NotImplementedError    
        else:
            raise NotImplementedError
        
        self.optimizers = self.get_optimizers()
        
    def get_actions(self, state: torch.Tensor, eval=False, **kwargs)->tuple[Tensor, Tensor]:
        state.to(device=self.device)

        if self.is_continous():
            mean, std = self.actor(state)
            mean, std = torch.squeeze(mean), torch.squeeze(std)
            normal = torch.distributions.Normal(mean, std) 

            if eval:
                action = mean
            else:
                action = normal.sample()
            # action = action.clip(self.action_min, self.action_max)
            return action, normal.log_prob(action).sum(dim=-1)
        else:
            probs = self.actor(state)
            if eval:
                action = torch.argmax(probs)
                action_probs = torch.distributions.Categorical(probs)
                log_prob = action_probs.log_prob(action)
                return action, log_prob
            else:
               action_probs = torch.distributions.Categorical(probs)
               action = action_probs.sample()
               log_prob = action_probs.log_prob(action)
               return action, log_prob
          
    def learn(self, batch: list[Transition], num_envs: int, batch_size: int)->dict[str, Tensor]:
        
        # Reshape batch to gathered lists 
        states, actions, next_states, rewards, dones, truncs, _ = map(torch.stack, zip(*batch))

        # Reshape data
        with torch.no_grad():
            states = states.to(device=self.device, dtype=torch.float32)
            actions = actions.to(device=self.device, dtype=torch.float32 if self.is_continous() else torch.int64)
            next_states = next_states.to(device=self.device, dtype=torch.float32)
            rewards = rewards.to(device=self.device, dtype=torch.float32)
            dones = dones.to(device=self.device, dtype=torch.float32)
            batch_rewards = rewards.clone()
            values, next_values = self.critic(states), self.target_critic(next_states) # self.critic(next_states)
            
        if self.is_continous():
            
            # Run the ICM module
            if self.hyperparams.icm_module.enabled:
                pred_mean, pred_std, next_state_feature, next_state_feature_pred = self.ICM(states, actions, next_states)
                
                intrinsic_rewards = self.ICM.intrinsic_reward(
                    next_state_feature, 
                    next_state_feature_pred,
                    self.hyperparams.icm_module.n
                ).detach()

                policy_mean, policy_std = self.actor(states)
                forward_loss, inverse_loss = self.ICM.calc_loss(
                    policy_mean, 
                    policy_std,
                    next_state_feature, 
                    next_state_feature_pred, 
                    pred_mean,
                    pred_std,
                    self.hyperparams.icm_module.beta
                )

                rewards = rewards + intrinsic_rewards

            with torch.no_grad():
                # rewards = winsorize(rewards, batch_size, self.device)
                # rewards = rewards.clip(-1.0, 1.0)
                # rewards = (rewards - rewards.mean()) / (rewards.std() + self.eps)
                
                target, advantages = self.compute_gae_and_targets(
                    rewards.unsqueeze(-1).clone(), 
                    dones.unsqueeze(-1).clone(), 
                    truncs.unsqueeze(-1).clone(), 
                    values.clone(), 
                    next_values.clone(), 
                    batch_size
                )
                advantages = (advantages - advantages.mean()) / (advantages.std() + self.eps)

            dist = Normal(*self.actor(states))
            log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
            entropy = dist.entropy().sum(dim=-1, keepdim=True)
            pg_loss = -log_probs * advantages

            # Critic loss
            critic_loss = F.smooth_l1_loss(self.critic(states.detach()), target)

            # Optimize the models
            if self.hyperparams.icm_module.enabled:
                self.optimizers['actor'].zero_grad()
                total_loss = (self.hyperparams.icm_module.alpha * pg_loss.mean()) + forward_loss + inverse_loss
                total_loss.backward() # type: ignore
                clip_grad_norm_(self.optimizers['actor'].param_groups[0]['params'], self.hyperparams.max_grad_norm)
                self.optimizers['actor'].step()
            else:
                self.optimizers['actor'].zero_grad()
                (pg_loss - (self.hyperparams.entropy_coefficient * entropy)).mean().backward()
                clip_grad_norm_(self.optimizers['actor'].param_groups[0]['params'], self.hyperparams.max_grad_norm)
                self.optimizers['actor'].step()

            self.optimizers['critic'].zero_grad()
            critic_loss.backward()
            clip_grad_norm_(self.optimizers['critic'].param_groups[0]['params'], self.hyperparams.max_grad_norm)
            self.optimizers['critic'].step()

            # # Update target network
            # for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            #     target_param.data.copy_(self.hyperparams.tau * param.data + (1 - self.hyperparams.tau) * target_param.data)

            return {
                "Actor loss": pg_loss.mean(),
                "Critic loss": critic_loss.mean(),
                "Entropy": entropy.mean(),
                "Rewards": batch_rewards.mean(),
                **({
                    "ICM Rewards": intrinsic_rewards.mean(), # type: ignore 
                    "ICM Forward Loss": forward_loss.mean(), # type: ignore 
                    "ICM Inverse Loss": inverse_loss.mean(), # type: ignore 
                } if self.hyperparams.icm_module.enabled else {}) 
            }
        else:
            with torch.no_grad():
                target = torch.unsqueeze(rewards, dim=-1) + (self.hyperparams.gamma * (1.0 - torch.unsqueeze(dones, dim=-1)) * next_state_values)
                advantages = target - state_values

            # Critic loss
            critic_loss = F.mse_loss(self.critic(states), torch.unsqueeze(target, dim=-1))

            # Policy loss with entropy bonus
            action_probs = self.actor(states.clone().detach())
            dist = torch.distributions.Categorical(action_probs)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()
            pg_loss = -log_probs * advantages
            loss = (pg_loss - self.hyperparams.entropy_coefficient * entropy).mean()

            # Optimize the models
            self.optimizers['critic'].zero_grad()
            critic_loss.backward()
            self.optimizers['critic'].step()
            
            self.optimizers['actor'].zero_grad()
            loss.backward()
            self.optimizers['actor'].step()

            # Update target network
            for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                target_param.data.copy_(self.hyperparams.tau * param.data + (1 - self.hyperparams.tau) * target_param.data)

            return {
                "Actor loss": pg_loss.mean(),
                "Critic loss": critic_loss,
                "Entropy": entropy.mean(),
                "Rewards": rewards.mean()
            }
    
    def get_optimizers(self) -> Dict[str, Optimizer]:
        return {
            "actor": AdamW(
                chain(self.actor.parameters(), self.ICM.parameters()) if self.hyperparams.icm_module.enabled else self.actor.parameters(), 
                lr=self.hyperparams.policy_learning_rate
            ),
            "critic": AdamW(
                self.critic.parameters(), 
                lr=self.hyperparams.value_learning_rate
            )
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
        
    def get_actions(self, state: torch.Tensor, eval=False, **kwargs)->tuple[Tensor, Tensor]:
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
