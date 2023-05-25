import torch
import numpy as np
import gymnasium as gym
import multiprocessing
from tqdm import tqdm
import random

from TrainPresets import REINFORCEHalfCheetahConfig as Config
from Agents import REINFORCE as Agent
from Trainer import Trainer
from Utilities import test_policy, set_random_seeds

    
if __name__ == "__main__":

    # Misc
    multiprocessing.freeze_support()
    set_random_seeds()

    # Hyperparams
    config = Config()
    device = torch.device(
        "mps" if torch.has_mps else "cpu" or # MACOS
        "cuda" if torch.has_cuda else 
        "cpu"
    )

    # Train the environment
    env = gym.vector.make(config.trainer_params.env_name, num_envs=config.trainer_params.num_envs)
    agent = Agent(observation_space=env.observation_space, action_space = env.action_space, hyperparams=config.agent_params, device=device)
    trainer = Trainer(agent, env, config.trainer_params, save_location=config.trainer_params.save_location)
    pbar = tqdm(total=config.trainer_params.num_epochs)
    for epoch in trainer:
        pbar.update(1)
    
    pbar.close()
    env.close()
    
    # Test the environment
    env = gym.make(config.trainer_params.env_name, render_mode="rgb_array")
    agent = Agent(observation_space = env.observation_space, action_space = env.action_space, hyperparams=config.agent_params, device=device)
    agent.load(config.trainer_params.save_location)
    test_policy(env, agent, normalization_weights=config.trainer_params.env_normalization_weights)