import torch 
import gymnasium as gym
import numpy as np
import os
import torch
import random

from typing import Any


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

def to_tensor(x: Any, device = torch.device("cpu"), dtype=torch.float32, requires_grad=True):
    return torch.tensor(x, device=device, dtype=dtype, requires_grad=requires_grad)

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
