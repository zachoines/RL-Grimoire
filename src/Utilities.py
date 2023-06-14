import torch 
import gymnasium as gym
import numpy as np
import os
import torch
import random
from typing import Any, Union, List

def clear_directories():
    directories = ['videos', 'saved_models']
    # directories = ['videos', 'runs', 'saved_models']

    for directory in directories:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    continue  # Skip Python files
                file_path = os.path.join(root, file)
                os.remove(file_path)

        for root, dirs, _ in os.walk(directory):
            for sub_dir in dirs:
                sub_dir_path = os.path.join(root, sub_dir)
                try:
                    os.rmdir(sub_dir_path)
                except OSError:
                    pass

    print("Directories cleared successfully.")


def set_random_seeds(seed=42):
   os.environ['PYTHONHASHSEED']=str(seed)
   torch.manual_seed(seed)
   np.random.seed(seed)
   random.seed(seed)

def test_policy(env, agent, num_episodes=5, max_steps=1024, normalizor = None,):
    with torch.no_grad():
        for _ in range(num_episodes):
            state, _ = env.reset()
            for _ in range(max_steps):
                if normalizor != None:
                    state = normalizor.normalize(state)
                action, _ = agent.get_actions(state, eval=True)
                action = action.cpu()
                next_state, _, _, _, _ = env.step(action)
                state = next_state

def to_tensor(x: Union[np.ndarray, torch.Tensor, int, float, List], device=torch.device("cpu"), dtype=torch.float32, requires_grad=True):
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, device=device, dtype=dtype, requires_grad=requires_grad)
    elif isinstance(x, torch.Tensor):
        x = x.to(device=device, dtype=dtype)
    elif isinstance(x, (int, float, list)):
        x = torch.tensor(x, device=device, dtype=dtype, requires_grad=requires_grad)
    else:
        raise ValueError("Unsupported data type. Only NumPy arrays, PyTorch tensors, primitives, and lists are supported.")
    return x

class RewardNormalizer:

    """
    The RewardNormalizer adapts to the changing distribution of rewards by using the Adam 
    running average calculation. It updates the running mean and variance based on the 
    incoming rewards, allowing it to respond to changes in the reward distribution over time.
    """

    def __init__(self, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = 0  # Running mean initialization
        self.v = 0  # Running variance initialization
        self.t = 0  # Timestep initialization

    def update(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Update the running mean and variance using rewards and normalize rewards.

        :param rewards: Tensor of rewards with shape (num_envs, num_steps, 1).
        :return: Normalized rewards with the same shape as input.
        """
        self.t += 1  # Increment timestep
        self.m = self.beta1 * self.m + (1 - self.beta1) * torch.mean(rewards)  # Update running mean
        self.v = self.beta2 * self.v + (1 - self.beta2) * torch.var(rewards, unbiased=False)  # Update running variance
        m_hat = self.m / (1 - self.beta1 ** self.t)  # Bias-corrected mean estimate
        v_hat = self.v / (1 - self.beta2 ** self.t)  # Bias-corrected variance estimate
        rewards_norm = (rewards - m_hat) / (torch.sqrt(v_hat) + self.eps)  # Normalize rewards
        return rewards_norm


class RunningMeanStd:

    def __init__(self, epsilon=1e-4, shape=(1, 1), device : torch.device = torch.device("cpu")):
        self.mean = torch.zeros(shape[-1], dtype=torch.float32).to(device)
        self.var = torch.ones(shape[-1], dtype=torch.float32).to(device)
        self.count = epsilon
        self.device = device

    def update(self, x: torch.Tensor):
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        self.mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + torch.square(delta) * self.count * batch_count / tot_count
        self.var = M2 / tot_count
        self.count = tot_count

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return ((x - self.mean) / torch.sqrt(self.var + 1e-8)).to(self.device)
    
    def save(self, loc='./normalizer'):
        torch.save({
            "means": self.mean,
            "vars" : self.var
        }, loc)

    def load(self, loc='./normalizer'):
        data = torch.load(loc)
        self.mean = data["means"]
        self.var = data["vars"]
