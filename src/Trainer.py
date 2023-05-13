from Agents import Agent
from Datasets import ExperienceBuffer, Transition
import gymnasium as gym
from torch.optim import Optimizer, AdamW
from Hyperparams import Hyperparams
from torch.utils.tensorboard.writer import SummaryWriter
from typing import Dict
from torch import Tensor

class Trainer:
    def __init__(self,
        agent: Agent, 
        exp_buffer: ExperienceBuffer,
        env: gym.Env,
        hyperparams: Hyperparams,
        save_location: str
    ):
        self.agent = agent
        self.exp_buffer = exp_buffer
        self.env = env
        self.state, _ = env.reset()
        self.hyperparams = hyperparams
        self.current_epoch = 0
        self.optimizers = {}
        self.save_location = save_location
        
        for net, params in agent.parameter_dict().items():
            self.optimizers[net] = AdamW(params, lr=hyperparams.policy_learning_rate)

        self.writer = SummaryWriter()
   
    def set_optimizers(self, optimizers: Dict[str, Optimizer])->None:
        self.optimizers = optimizers

    def model_step(self)->dict[str, Tensor]:
        return self.agent.learn(self.exp_buffer, self.hyperparams, self.optimizers)

    def log_step(self, train_results: dict[str, Tensor])->None:
        for metric, value in train_results.items():
            self.writer.add_scalar(tag=metric, scalar_value=value, global_step=self.current_epoch)

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
        for _ in range(self.hyperparams.samples_per_epoch):
            action = self.agent.get_actions(self.state)
            next_state, reward, done, _, _ = self.env.step(action.flatten())
            self.exp_buffer.append([Transition(self.state, action, next_state, reward, done)])
            self.state = next_state

        train_results = self.model_step()
        self.log_step(train_results)
        self.save_model()

        return self.current_epoch