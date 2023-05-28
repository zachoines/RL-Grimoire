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
        normalizer: RunningMeanStd,
        test_callback = None
    ):
                
        self.agent = agent
        self.exp_buffer = ExperienceBuffer(hyperparams.replay_buffer_max_size)
        self.env = env
        self.hyperparams = hyperparams
        self.current_epoch = 0
        self.current_update = 0
        self.current_step = 0
        self.optimizers = {}
        self.save_location = save_location
        self.writer = SummaryWriter()
        self.test_callback = test_callback
        self.normalizer = normalizer
        self.state : torch.Tensor

        self.reset()

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

        if self.hyperparams.on_policy_training:
            self.exp_buffer.empty()
        
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
    
    def reset(self):
        self.state, _ = self.env.reset()
        if self.hyperparams.env_normalization:
            self.state = self.normalizer.normalize(self.state)

    def step(self):
        
        if self.hyperparams.render:
            self.env.render()
        
        with torch.no_grad():
            self.current_step += 1
            action, other = self.agent.get_actions(self.state)
            other = other.cpu()
            action = action.cpu().squeeze(-1) if self.hyperparams.squeeze_actions else action.cpu()
            next_state, reward, done, _, _ = self.env.step(action)
            if self.hyperparams.env_normalization:
                next_state = self.normalizer.normalize(next_state)
            
            self.writer.add_scalar(tag="Step Rewards", scalar_value=reward.mean(), global_step=self.current_step) # type: ignore

            # if (self.current_step % self.hyperparams.episode_length ) == 0:
            #     done = torch.ones_like(done, dtype=torch.bool) # type: ignore
            
            if self.hyperparams.batch_transitions_by_env_trajectory:
                self.exp_buffer.append([Transition(self.state, action, next_state, reward, done, other)])
            else:
                self.exp_buffer.append([Transition(s, a, n_s, r, d, o) for s, a, n_s, r, d, o in zip(self.state, action, next_state, reward, done, other)]) # type: ignore
            self.state = next_state

            # if (self.current_step % self.hyperparams.episode_length ) == 0:
            #     self.state, _ = self.env.reset()
            #     if self.hyperparams.env_normalization:
            #         self.state = self.normalizer.normalize(self.state)

    def __next__(self):

        # Stop after reaching max epochs
        self.current_epoch += 1
        if self.current_epoch > self.hyperparams.num_epochs:
            self.writer.close()
            self.env.close()
            raise StopIteration
        
        # Collect experiances
        for _ in range(self.hyperparams.batches_per_epoch):
            self.step()

            # Fill up buffer
            while (len(self.exp_buffer) < self.hyperparams.batch_size): 
                self.step()

            # Update with batch
            self.log_step(self.model_step())

            # for _ in range(self.hyperparams.sam):  
            #     self.step()

            #     # Fill up buffer if needed
            #     while len(self.exp_buffer) < self.hyperparams.replay_buffer_min_size:
            #         self.step() 

            #     # Take an update with batch
            #     if len(self.exp_buffer) >= self.hyperparams.batch_size:
            #         self.log_step(self.model_step())

        # Test the policy
        # if (self.current_epoch % self.hyperparams.record_video_frequency) == 0:
        #     self.reset()
        
        # Save parameters
        self.save_model()
        return self.current_epoch