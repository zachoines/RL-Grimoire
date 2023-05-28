import sys
import os

# Get the current directory and append the 'src' folder path to sys.path
# Note: May still need to run: export PYTHONPATH="RL-Grimoire/src"
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.append(src_path)

import torch
import gymnasium as gym
import numpy as np
from tqdm import tqdm

from TrainPresets import PPOBraxAntConfig as Config
from Agents import PPO as Agent
from Trainer import Trainer
from Utilities import set_random_seeds, RunningMeanStd
from brax_to_gymnasium import JaxToTorchWrapper, VectorGymWrapper
from datetime import datetime


# Brax related environments
import brax
import brax.v1.envs as envs

if __name__ == "__main__":

    # Misc
    set_random_seeds()

    # Hyperparams
    config = Config()
    device = torch.device(
        "mps" if torch.has_mps else "cpu" or # MACOS
        "cuda" if torch.has_cuda else 
        "cpu"
    )

    # Create the environment ß
    current_datetime = datetime.now()
    current_date_string = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

    env = envs.create(
        config.trainer_params.env_name, 
        batch_size=config.trainer_params.num_envs,
        episode_length=config.trainer_params.episode_length,
        use_contact_forces=True,
        terminate_when_unhealthy = True,
        healthy_reward = 0.5,
        ctrl_cost_weight=1.0,
        contact_cost_weight=1e-3
    )
    env = VectorGymWrapper(
        env,
        record = True, 
        record_location = 'videos/',
        record_name_prefix = f"{config.trainer_params.env_name}_{current_date_string}",
        recording_save_frequeny = 512
    )
    env = JaxToTorchWrapper(env, device)
    running_mean_std_recorder = RunningMeanStd(shape=env.observation_space.shape, device=device)
    state, _ = env.reset()
    with torch.no_grad():
        for _ in range(256):
            action = env.unwrapped.action_space.sample()
            next_state, _, _, _, _ = env.step(action)
            running_mean_std_recorder.update(next_state)
            state = next_state  
    env.close()
    
    # Train agent
    agent = Agent(observation_space=env.observation_space, action_space = env.action_space, hyperparams=config.agent_params, device=device)
    trainer = Trainer(
        agent,
        env,
        config.trainer_params,
        save_location=config.trainer_params.save_location,
        normalizer=running_mean_std_recorder
    )
    pbar = tqdm(total=config.trainer_params.num_epochs)
    for epoch in trainer:
        pbar.update(1)


    running_mean_std_recorder.save(loc=config.trainer_params.save_location + "NormStates")
    pbar.close()
    env.close()