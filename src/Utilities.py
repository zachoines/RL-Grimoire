import torch 
import gymnasium as gym
import numpy as np
import os
import torch
import random
from typing import Any, Union, List

def clear_directories(directories: List[str]):
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

def requires_grad(x: torch.Tensor):
    x_grad = x.clone()
    x_grad.requires_grad_(True)
    return x_grad

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

def winsorize(data: torch.Tensor, batch_size: int, device: torch.device, lower_percentile: float = 0.005, upper_percentile: float = 0.995):
    data = data.to(device=device)
    if device.type == 'cpu' or device.type.startswith('cuda'):
        lower = data.kthvalue(int(lower_percentile * batch_size)).values
        upper = data.kthvalue(int(upper_percentile * batch_size)).values
    else: # No support yet for MPS devices
        sorted_data, _ = torch.sort(data)
        lower = sorted_data[int(lower_percentile * batch_size)]
        upper = sorted_data[int(upper_percentile * batch_size)]
    return torch.clamp(data, lower, upper)

class YeoJohnsonTransform:
    def __init__(self, lmbda1=1.0, lmbda2=1.0):
        self.lmbda1 = lmbda1
        self.lmbda2 = lmbda2

    def __call__(self, x):
        out = torch.zeros_like(x)
        pos = x >= 0  # Boolean tensor where positive values are True
        neg = x < 0   # Boolean tensor where negative values are True

        # Apply Yeo-Johnson transformation for positive and negative parts separately
        out[pos] = ((x[pos] + 1).pow(self.lmbda1) - 1) / self.lmbda1 if self.lmbda1 != 0 else torch.log(x[pos] + 1)
        out[neg] = -((-x[neg] + 1).pow(2 - self.lmbda2) - 1) / (2 - self.lmbda2) if self.lmbda2 != 2 else -torch.log(-x[neg] + 1)

        return out

class AdamNormalizer:

    def __init__(self, mean_decay_rate: float = 0.8, variance_decay_rate: float = 0.9, eps: float = 1e-8, device: torch.device = torch.device('cpu')):
        self.mean_decay_rate = mean_decay_rate
        self.variance_decay_rate = variance_decay_rate
        self.eps = eps
        self.m = torch.tensor(0., device=device)  # Running mean initialization
        self.v = torch.tensor(eps, device=device)  # Running variance initialization, initialized with eps to prevent division by zero
        self.t = torch.tensor(0., device=device)  # Timestep initialization
        self.device = device

    def update(self, data: torch.Tensor, log_rewards: bool=False) -> torch.Tensor:
        """
        Update the running mean and variance using data and normalize data.

        :param data: Tensor of data with any shape.
        :return: Normalized data with the same shape as input.
        """
        
        data = data.to(self.device)  # Ensure data is on the correct device
        if log_rewards:
            data = torch.where(data >= 0, torch.log(data + 1e-6), -torch.log(-data + 1e-6))
        data_flattened = data.view(-1)  # Flatten the data tensor
        batch_size = data_flattened.shape[0]
        
        # Winsorize the data
        data_flattened = winsorize(data, batch_size, self.device)

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
  
class SlidingWindowNormalizer:
    def __init__(self, window_size=2024, device=torch.device("cpu")):
        self.window_size = window_size
        self.device = device
        self.rewards = None
        self.index = 0

    def update(self, rewards):
        num_steps, num_envs = rewards.size()
        if self.rewards is None:
            self.rewards = torch.zeros((self.window_size, num_envs), device=self.device)

        for i in range(num_steps):
            self.rewards[self.index] = rewards[i]
            self.index = (self.index + 1) % self.window_size

        mean = self.rewards.mean(dim=0, keepdim=True)
        std = self.rewards.std(dim=0, keepdim=True)
        normalized_rewards = (rewards - mean) / (std + 1e-8)
        return normalized_rewards
    
class RunningMeanStdNormalizer:
    def __init__(self, epsilon: float = 1e-4, device: torch.device = torch.device("cpu")):
        """
        Initialize a Running Mean Standard Deviation Normalizer.

        Args:
            epsilon (float): A small constant to prevent division by zero.
            device (torch.device): The device to run the calculations on.
        """
        # Initialize the mean and variance to None. They will be initialized later based on the shape of the input data.
        self.mean = None
        self.var = None

        # Initialize the count to a small value to prevent division by zero.
        self.count = epsilon

        # Store the device to run the calculations on.
        self.device = device

    def update(self, x: torch.Tensor):
        """
        Normalize a tensor with the running mean and standard deviation.

        Args:
            x (torch.Tensor): The tensor to normalize.

        Returns:
            torch.Tensor: The normalized tensor.
        """
        # If the mean and variance are not initialized, initialize them based on the shape of the input data.
        if self.mean is None or self.var is None:
            self.mean = torch.zeros(x.shape[1:]).to(self.device)  # Initialize the mean to zeros.
            self.var = torch.ones(x.shape[1:]).to(self.device)  # Initialize the variance to ones.

        # Calculate the mean and variance of the new data.
        batch_mean = torch.mean(x, dim=0)  # Mean of the new data.
        batch_var = torch.var(x, dim=0)  # Variance of the new data.
        batch_count = x.size(0)  # Number of new data points.

        # Calculate the delta between the new mean and the current mean.
        delta = batch_mean - self.mean

        # Calculate the total count of data points seen so far.
        tot_count = self.count + batch_count

        # Update the mean based on the new data.
        self.mean = self.mean + delta * batch_count / tot_count

        # Calculate the sum of the variances of the old data and the new data.
        m_a = self.var * self.count  # Variance of the old data times the number of old data points.
        m_b = batch_var * batch_count  # Variance of the new data times the number of new data points.

        # Calculate the sum of the squares of the differences between the means of the old data and the new data.
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count

        # Update the variance based on the new data.
        self.var = M2 / tot_count

        # Update the total count of data points seen so far.
        self.count = tot_count

        # Normalize the input tensor using the updated mean and variance and return it.
        return (x - self.mean) / torch.sqrt(self.var + 1e-8)


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

class VPID:
    def __init__(self, target, max_velocity, max_acceleration, p, i, d, windup_gaurd=1.0, device=torch.device("cpu")):
        self.target = torch.tensor(target, device=device)
        self.P = p
        self.I = i
        self.D = d
        self.integral = torch.tensor(0., device=device)
        self.previous_error = torch.tensor(0., device=device)
        self.previous_velocity = torch.tensor(0., device=device)
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.windup_gaurd = windup_gaurd
        self.device = device

    def reset(self):
        self.integral = torch.tensor(0., device=self.device)
        self.previous_error = torch.tensor(0., device=self.device)
        self.previous_velocity = torch.tensor(0., device=self.device)

    def updateTarget(self, target):
        self.target = torch.tensor(target, device=self.device)

    def update(self, current, ts, reverse=False):
        error = current - self.target if reverse else self.target - current
        self.integral += error * ts

        # Clip integral
        if abs(self.integral.item()) > self.windup_gaurd:
            self.integral = torch.tensor(self.windup_gaurd * torch.sign(self.integral), device=self.device)

        error_derivative = (self.previous_error - error) / ts

        # Determine new control velocity
        new_velocity = (self.P * error) + (self.D * error_derivative) + (self.I * self.integral)

        # Clip velocity
        if abs(new_velocity.item()) > self.max_velocity:
            new_velocity = torch.tensor(self.max_velocity * torch.sign(new_velocity), device=self.device)

        # Clip acceleration
        a = (new_velocity - self.previous_velocity) / ts

        if abs(a.item()) > self.max_acceleration:
            a = torch.tensor(self.max_acceleration * torch.sign(a), device=self.device)
        new_velocity = self.previous_velocity + a * ts

        self.previous_velocity = new_velocity
        self.previous_error = error

        return new_velocity.item()
    
class PID:
    def __init__(self, target: torch.Tensor, p: torch.Tensor, i: torch.Tensor, d: torch.Tensor, windup_guard: float = 1.0, device: torch.device = torch.device("cpu")):
        """
        Initialize the PID controller with target value, gains, and windup guard.

        :param target: The target value as a torch.Tensor.
        :param p: The proportional gain as a torch.Tensor.
        :param i: The integral gain as a torch.Tensor.
        :param d: The derivative gain as a torch.Tensor.
        :param windup_guard: The windup guard value to limit the integral term (default: 1.0).
        :param device: The device on which the computations will be performed (default: 'cpu').
        """
        self.target = target.to(device)
        self.P = p.to(device)
        self.I = i.to(device)
        self.D = d.to(device)
        self.integral = torch.tensor(0., device=device)
        self.previous_error = torch.tensor(0., device=device)
        self.windup_guard = windup_guard
        self.device = device

    def reset(self):
        """
        Reset the integral and previous error to zero.
        """
        self.integral = torch.tensor(0., device=self.device)
        self.previous_error = torch.tensor(0., device=self.device)

    def update(self, current: torch.Tensor) -> torch.Tensor:
        """
        Update the PID controller, apply anti-windup mechanism, and calculate the control output.

        :param current: The current value as a torch.Tensor.
        :return: The control output as a torch.Tensor.
        """
        error = self.target - current
        self.integral += error

        # Apply windup guard to limit integral term
        self.integral = torch.clamp(self.integral, -self.windup_guard, self.windup_guard)

        # Compute error derivative
        error_derivative = error - self.previous_error

        # Determine control output
        control_output = (self.P * error) + (self.D * error_derivative) + (self.I * self.integral)

        self.previous_error = error

        return torch.tanh(control_output)
