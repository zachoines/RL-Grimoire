import torch 
import gymnasium as gym
from gymnasium.wrappers import RecordVideo # type: ignore
import numpy as np
import os
import torch
import random


def set_random_seeds(seed=42):
   os.environ['PYTHONHASHSEED']=str(seed)
   torch.manual_seed(seed)
   np.random.seed(seed)
   random.seed(seed)

def test_policy(env, agent, num_episodes=5, max_steps=1024, normalization_weights=np.array([]), video=True):
    if video:
        env = RecordVideo(env, 'videos', episode_trigger=lambda e: True)
    with torch.no_grad():
        for _ in range(num_episodes):
            state, _ = env.reset()
            for _ in range(max_steps):
                if not video:
                    env.render()
                if np.any(normalization_weights):
                    state *= normalization_weights
                action, _ = agent.get_actions(state, eval=True)
                action = action.cpu().numpy()
                next_state, _, _, _, _ = env.step(action)
                state = next_state

class RunningMeanStd:

    def __init__(self, epsilon=1e-4, shape=(), device : torch.device = torch.device("cpu")):
        self.mean = torch.zeros(shape, dtype=torch.float32).to(device)
        self.var = torch.ones(shape, dtype=torch.float32).to(device)
        self.count = epsilon

    def update(self, x):
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

    def normalize(self,x):
        return (x - self.mean) / torch.sqrt(self.var + 1e-8)