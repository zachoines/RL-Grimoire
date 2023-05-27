import numpy as np
import gymnasium as gym
from typing import List, Dict

# Torch imports
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard.writer import SummaryWriter
import torch
from torch import Tensor

# Local imports
from Utilities import RunningMeanStd
from Agents import Agent
from Datasets import ExperienceBuffer, Transition
from Configurations import TrainerParams
import importlib


class Trainer:
    def __init__(self,
        agent: Agent, 
        env: gym.Env,
        hyperparams: TrainerParams,
        save_location: str,
        test_callback = None
    ):
                
        self.agent = agent
        self.exp_buffer = ExperienceBuffer(hyperparams.replay_buffer_max_size)
        self.env = env
        self.state, _ = env.reset()
        self.hyperparams = hyperparams
        self.current_epoch = 0
        self.current_update = 0
        self.current_step = 0
        self.optimizers = {}
        self.save_location = save_location
        self.writer = SummaryWriter()
        self.test_callback = test_callback
        self.env_running_mean_std = RunningMeanStd(shape = env.observation_space.shape)
    
        if np.any(hyperparams.env_normalization_weights):
            self.state *= hyperparams.env_normalization_weights

        self.learning_rate_schedulers = {}
        if hyperparams.learningRateScheduler:
            for network_name, optimizer in self.agent.optimizers.items():
                class_ = getattr(importlib.import_module("torch.optim.lr_scheduler"), hyperparams.learningRateSchedulerClass)
                learning_rate_scheduler: lr_scheduler.LRScheduler = class_(optimizer, **hyperparams.learningRateScheduleArgs)
                self.learning_rate_schedulers[network_name] = learning_rate_scheduler


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
            self.hyperparams.batch_size, 
            remove=self.hyperparams.replay_buffer_remove_on_sample, 
            shuffle=self.hyperparams.replay_buffer_shuffle_experiances
        )
        self.current_update += 1
        train_results = []
        for _ in range(self.hyperparams.updates_per_batch):
            if self.hyperparams.shuffle_batches:
                np.random.shuffle(batch)
            train_results.append(self.agent.learn(batch, self.hyperparams.num_envs, self.hyperparams.batch_size))

        if self.hyperparams.learningRateScheduler:
            for _, scheduler in self.learning_rate_schedulers.items():
                scheduler.step()
        
        return self.reduce_dicts_to_avg(train_results)

    def log_step(self, train_results: dict[str, Tensor])->None:
        for metric, value in train_results.items():
            self.writer.add_scalar(tag=metric, scalar_value=value, global_step=self.current_update)

        if self.hyperparams.learningRateScheduler:
            for name, scheduler in self.learning_rate_schedulers.items():
                [value] = scheduler.get_last_lr()
                self.writer.add_scalar(tag=name + " learning rate scheduler", scalar_value=value, global_step=self.current_update)

    def save_model(self)->None:
        self.agent.save(self.save_location)
        self.agent.save(self.save_location + str(self.current_epoch))

    def __iter__(self):
        return self
    
    def step(self):
        
        if self.hyperparams.render:
            self.env.render()
        
        with torch.no_grad():
            self.current_step += 1
            action, other = self.agent.get_actions(self.state)
            other = other.cpu().detach()
            action = action.cpu().numpy().squeeze(-1) if self.hyperparams.squeeze_actions else action.cpu().numpy()
            next_state, reward, done, _, _ = self.env.step(action)
            # self.env_running_mean_std.update(next_state)
            if np.any(self.hyperparams.env_normalization_weights):
                next_state *= self.hyperparams.env_normalization_weights
            self.writer.add_scalar(tag="Step Rewards", scalar_value=reward.mean(), global_step=self.current_step) # type: ignore

            if (self.hyperparams.episode_length % self.current_step) == 0:
                done = np.ones_like(done).astype(bool)
            
            if self.hyperparams.batch_transitions_by_env_trajectory:
                self.exp_buffer.append([Transition(self.state, action, next_state, reward, done, other)])
            else:
                self.exp_buffer.append([Transition(s, a, n_s, r, d, o) for s, a, n_s, r, d, o in zip(self.state, action, next_state, reward, done, other)]) # type: ignore
            self.state = next_state

            if (self.hyperparams.episode_length % self.current_step) == 0:
                self.state, _ = self.env.reset()

    def __next__(self):

        # Stop after reaching max epochs
        self.current_epoch += 1
        if self.current_epoch > self.hyperparams.num_epochs:
            self.writer.close()
            raise StopIteration
        
        # Collect experiances
        for _ in range(self.hyperparams.batches_per_epoch):
            for _ in range(self.hyperparams.batch_size):  
                self.step()

                # Fill up buffer if needed
                while len(self.exp_buffer) < self.hyperparams.replay_buffer_min_size:
                    self.step() 

            # Take an update with batch
            self.log_step(self.model_step())

        # Test the policy
        if (self.current_epoch % self.hyperparams.record_video_frequency) == 0:
            if self.test_callback != None:
                self.test_callback(self.agent)
        
        # Save parameters
        self.save_model()
        return self.current_epoch