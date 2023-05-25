import torch 
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