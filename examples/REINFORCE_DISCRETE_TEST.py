import torch
import numpy as np
import gymnasium as gym
import multiprocessing
from tqdm import tqdm

from Hyperparams import MountainCarHyperparams, Hyperparams
from Agents import REINFORCE
from Trainer import Trainer
import random

def test_policy(env, agent, num_episodes):

    with torch.no_grad():
        for _ in range(num_episodes):
            state, _ = env.reset()
            done = False
            steps = 0
            while not done:
                steps += 1
                env.render()
                action = agent.get_actions(state, eval=True).cpu().numpy()
                next_state, _, done, _, _ = env.step(action)
                state = next_state

    
if __name__ == "__main__":
    # Set the Random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    multiprocessing.freeze_support()
    hyperparams = MountainCarHyperparams()
    device = torch.device(
        "mps" if torch.has_mps else "cpu" or # MACOS
        "cuda" if torch.has_cuda else 
        "cpu"
    )

    env = None
    if hyperparams.num_envs == 1:
        env = gym.make(hyperparams.env_name, render_mode="human")
    else:
        env = gym.vector.make(hyperparams.env_name, num_envs=hyperparams.num_envs)

    action_space = env.action_space
    observation_space = env.observation_space
    agent = REINFORCE(hidden_size=hyperparams.hidden_size, observation_space=observation_space, action_space = action_space, device=device)
    trainer = Trainer(agent, env, hyperparams, save_location=hyperparams.save_location)

    # Train the environment
    pbar = tqdm(total=hyperparams.num_epochs)
    for epoch in trainer:
        pbar.update(1)
    
    pbar.close()
    env.close()
    
    # Test the environment
    env = gym.make(hyperparams.env_name, render_mode="human")
    action_space = env.action_space
    observation_space = env.observation_space
    agent = REINFORCE(hidden_size=hyperparams.hidden_size, observation_space = observation_space, action_space = action_space, device=device)
    agent.load(hyperparams.save_location)
    test_policy(env, agent, 100)