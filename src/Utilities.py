import torch 
import gymnasium as gym
import numpy as np
import os
import torch
import random
from typing import Any, Union, List

def clear_directories():
    # directories = ['videos', 'saved_models']
    directories = ['videos', 'runs', 'saved_models']

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

class Normalizer:

    def __init__(self, mean_decay_rate: float = 0.9, variance_decay_rate: float = 0.999, eps: float = 1e-8, lower_percentile: float = 0.01, upper_percentile: float = 0.99, device: torch.device = torch.device('cpu')):
        self.mean_decay_rate = mean_decay_rate
        self.variance_decay_rate = variance_decay_rate
        self.eps = eps
        self.m = torch.tensor(0., device=device)  # Running mean initialization
        self.v = torch.tensor(eps, device=device)  # Running variance initialization, initialized with eps to prevent division by zero
        self.t = torch.tensor(0., device=device)  # Timestep initialization
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.device = device

    def update(self, data: torch.Tensor) -> torch.Tensor:
        """
        Update the running mean and variance using data and normalize data.

        :param data: Tensor of data with any shape.
        :return: Normalized data with the same shape as input.
        """
        
        data = data.to(self.device)  # Ensure data is on the correct device
        data_flattened = data.view(-1)  # Flatten the data tensor
        batch_size = data_flattened.shape[0]
        
        # Winsorize the data
        if self.device.type == 'cpu' or self.device.type.startswith('cuda'):
            lower = data_flattened.kthvalue(int(self.lower_percentile * batch_size)).values
            upper = data_flattened.kthvalue(int(self.upper_percentile * batch_size)).values
        else:
            sorted_data, _ = torch.sort(data_flattened)
            lower = sorted_data[int(self.lower_percentile * batch_size)]
            upper = sorted_data[int(self.upper_percentile * batch_size)]
        data_flattened = torch.clamp(data_flattened, lower, upper)

        if self.t == 0:
            # If this is the first batch, initialize the running mean and variance with the sample mean and variance
            self.m = data_flattened.mean()
            self.v = data_flattened.var(unbiased=False)
        else:
            # For subsequent batches, update the running mean and variance as before
            self.m = self.mean_decay_rate * self.m + (1 - self.mean_decay_rate) * data_flattened.mean()  # Update running mean
            var_update = self.variance_decay_rate * self.v + (1 - self.variance_decay_rate) * data_flattened.var(unbiased=False)  # Compute running variance update
            self.v = torch.max(var_update, to_tensor(self.eps, device=self.device, dtype=torch.float32))  # Update running variance, ensuring it's never less than eps

        self.t += batch_size  # Increment timestep by batch size

        m_hat = self.m / (1 - self.mean_decay_rate ** self.t)  # Bias-corrected mean estimate
        v_hat = self.v / (1 - self.variance_decay_rate ** self.t)  # Bias-corrected variance estimate
        data_norm = (data - m_hat) / (torch.sqrt(v_hat) + self.eps)  # Normalize data
        return data_norm


class RunningMeanStd:

    def __init__(self, epsilon=1e-4, shape=(1, 1), device : torch.device = torch.device("cpu")):
        self.mean = torch.zeros(shape[-1], dtype=torch.float32).to(device)
        self.var = torch.ones(shape[-1], dtype=torch.float32).to(device)
        self.count = epsilon
        self.device = device

    def update(self, x: torch.Tensor):
        batch_count = x.shape[0]
        if batch_count == 1:
            return x
        
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

        return ((x - self.mean) / torch.sqrt(self.var + 1e-8)).to(self.device)

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
