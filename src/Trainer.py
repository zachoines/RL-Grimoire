from Agents import Agent
from Datasets import ExperienceBuffer, Transition
import gymnasium as gym
from Configurations import TrainerParams
from torch.utils.tensorboard.writer import SummaryWriter
import torch
from torch import Tensor
import numpy as np

class Trainer:
    def __init__(self,
        agent: Agent, 
        env: gym.Env,
        hyperparams: TrainerParams,
        save_location: str
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
        if np.any(hyperparams.env_normalization_weights):
            self.state *= hyperparams.env_normalization_weights


    def model_step(self)->dict[str, Tensor]:
        return self.agent.learn(self.exp_buffer, self.hyperparams.num_envs, self.hyperparams.batch_size)

    def log_step(self, train_results: dict[str, Tensor])->None:
        for metric, value in train_results.items():
            self.writer.add_scalar(tag=metric, scalar_value=value, global_step=self.current_update)

    def save_model(self)->None:
        return self.agent.save(self.save_location)

    def __iter__(self):
        return self

    def __next__(self):

        # Stop when reaching max epochs
        self.current_epoch += 1
        if self.current_epoch > self.hyperparams.num_epochs:
            self.writer.close()
            raise StopIteration
        
        # Collect experiances
        for _ in range(self.hyperparams.steps_per_epoch):
            self.current_step += 1
            if self.hyperparams.render:
                self.env.render()
            
            with torch.no_grad():
                action = self.agent.get_actions(self.state).cpu().numpy() # .squeeze()
                next_state, reward, done, _, _ = self.env.step(action)
                if np.any(self.hyperparams.env_normalization_weights):
                    next_state *= self.hyperparams.env_normalization_weights
                self.writer.add_scalar(tag="Step Rewards", scalar_value=reward.mean(), global_step=self.current_step) # type: ignore
                self.exp_buffer.append([Transition(s, a, n_s, r, d) for s, a, n_s, r, d in zip(self.state, action, next_state, reward, done)]) # type: ignore
                self.state = next_state

            if len(self.exp_buffer) > self.hyperparams.replay_buffer_min_size:
                if (self.current_step % self.hyperparams.update_rate) == 0:
                    self.current_update += 1
                    train_results = self.model_step()
                    self.log_step(train_results)
            
        self.save_model()

        return self.current_epoch