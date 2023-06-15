import numpy as np
import gymnasium as gym
from typing import List, Dict

# Torch imports
from torch.utils.tensorboard.writer import SummaryWriter
import torch
from torch import Tensor

# Local imports
from Utilities import RunningMeanStd, to_tensor
from Agents import Agent
from Datasets import ExperienceBuffer, Transition
from Configurations import TrainerParams, EnvParams
from Utilities import Normalizer

class Trainer:
    def __init__(self,
        agent: Agent, 
        env: gym.Env,
        train_params: TrainerParams,
        env_params: EnvParams,
        save_location: str,
        normalizer: RunningMeanStd = RunningMeanStd(),
        device = torch.device("cpu")
    ):
                
        self.agent = agent
        self.exp_buffer = ExperienceBuffer(train_params.replay_buffer_max_size)
        self.env = env
        self.train_params = train_params
        self.env_params = env_params
        self.current_epoch = 0
        self.current_update = 0
        self.current_step = 0
        self.optimizers = {}
        self.save_location = save_location
        self.writer = SummaryWriter()
        self.normalizer = normalizer
        self.state : torch.Tensor
        self.device = device

        self.action_min = float(self.env.action_space.low_repr) # type: ignore
        self.action_max = float(self.env.action_space.high_repr) # type: ignore

        self.reset()

    def reduce_dicts_to_avg(self, list_of_dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        result_dict: Dict[str, torch.Tensor] = {}
        
        if len(list_of_dicts) == 0:
            return result_dict
        
        keys = list_of_dicts[0].keys()
        
        for key in keys:
            values_sum = torch.zeros_like(list_of_dicts[0][key])
            
            for dictionary in list_of_dicts:
                values_sum += dictionary[key]
            
            avg_value = values_sum / len(list_of_dicts)
            result_dict[key] = avg_value
        
        return result_dict

    def model_step(self)->dict[str, Tensor]:
        batch = self.exp_buffer.sample(
            self.train_params.batch_size, 
            remove=self.train_params.replay_buffer_remove_on_sample, 
            shuffle=self.train_params.replay_buffer_shuffle_experiances
        )
        self.current_update += 1
        train_results = []
        for _ in range(self.train_params.updates_per_batch):
            if self.train_params.shuffle_batches:
                np.random.shuffle(batch)
            train_results.append(self.agent.learn(batch, self.env_params.num_envs, self.train_params.batch_size))

        if self.train_params.on_policy_training:
            self.exp_buffer.empty()
        
        return self.reduce_dicts_to_avg(train_results)

    def log_step(self, train_results: dict[str, Tensor])->None:
        for metric, value in train_results.items():
            self.writer.add_scalar(tag=metric, scalar_value=value, global_step=self.current_update)

    def save_model(self)->None:
        self.agent.save(self.save_location)
        self.agent.save(self.save_location + str(self.current_epoch))

    def __iter__(self):
        return self
    
    def reset(self):
        self.state, _ = self.env.reset()
        if self.env_params.env_normalization:
            self.state = self.normalizer.update(to_tensor(self.state, device=self.device))

    def step(self):
        
        if self.train_params.render:
            self.env.render()
        
        with torch.no_grad():
            self.current_step += 1
            if self.state.__class__ == np.ndarray:
                self.state = to_tensor(self.state, device=self.device)
            action, other = self.agent.get_actions(self.state)
            other = other.cpu()
            action = action.cpu()
            action = self.train_params.preprocess_action(action)
            next_state, reward, done, trunc, _ = self.env.step(action)
        
            # Convert to tensor if not
            other = to_tensor(other, device=self.device)
            action = to_tensor(action, device=self.device)
            next_state = to_tensor(next_state, device=self.device)
            reward = to_tensor(reward, device=self.device)
            done = to_tensor(done, device=self.device)
            trunc = to_tensor(trunc, device=self.device)
            self.state = to_tensor(self.state, device=self.device)

            if self.env_params.env_normalization:
                next_state = self.normalizer.update(next_state)
            
            self.writer.add_scalar(tag="Step Rewards", scalar_value=reward.mean(), global_step=self.current_step) # type: ignore
            
            if self.train_params.batch_transitions_by_env_trajectory:
                self.exp_buffer.append([Transition(self.state, action, next_state, reward, done, trunc, other)])
            else:
                self.exp_buffer.append([Transition(s, a, n_s, r, d, t, o) for s, a, n_s, r, d, t, o in zip(self.state, action, next_state, reward, done, trunc, other)]) # type: ignore
            self.state = next_state

    def __next__(self):

        # Stop after reaching max epochs
        self.current_epoch += 1
        if self.current_epoch > self.train_params.num_epochs:
            self.writer.close()
            self.env.close()
            raise StopIteration
        
        # Collect experiances
        for _ in range(self.train_params.batches_per_epoch):
            self.step()

            # Fill up buffer
            while (len(self.exp_buffer) < self.train_params.batch_size): 
                self.step()

            # Update with batch
            self.log_step(self.model_step())

        # Save parameters
        if (self.current_epoch % self.train_params.save_model_frequency) == 0:
            self.save_model()
            
        return self.current_epoch