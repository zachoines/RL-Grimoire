import sys
import os
import argparse
import importlib
from tqdm import tqdm

# Make accessible 'src' and its modules
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.append(src_path)

from src.TrainPresets import *
from src.TrainPresets import Config
from src.Trainer import Trainer
from src.Utilities import set_random_seeds, RunningMeanStd, test_policy
from src.wrappers import RecordVideoWrapper

import torch
import gymnasium as gym

# Setup arguements
parser = argparse.ArgumentParser(description='Description of your program.')
parser.add_argument('-c', '--config', type=str, help='Train configuration class', required=True)

# Register environments here
gym.register(
    id="brax-ant",
    entry_point="src.brax_to_gymnasium:convert_brax_to_gym",
    kwargs={
        "name": "ant",
        "episode_length": 1024,
        "healthy_reward": 1.0,
        "ctrl_cost_weight": 0.5,
        "contact_cost_weight": 1e-4,
        "use_contact_forces": True,
        "terminate_when_unhealthy": True
    }
)

gym.register(
    id="brax-hopper",
    entry_point="src.brax_to_gymnasium:convert_brax_to_gym",
    kwargs={
        "name": "hopper",
        "episode_length": 1024,
        "forward_reward_weight": 1.0,
        "ctrl_cost_weight": 1e-3,
        "healthy_reward": 1.0,
        "terminate_when_unhealthy": True
    }
)

if __name__ == "__main__":
    # Misc setup
    set_random_seeds()
    args = parser.parse_args()
    device = torch.device(
        "mps" if torch.has_mps else (# MACOS
        "cuda" if torch.has_cuda else 
        "cpu")
    )

    # Load train hyperparams
    config_class_ = getattr(importlib.import_module("src.TrainPresets"), args.config)
    config: Config = config_class_()

    # Load environment
    env: gym.Env
    if not config.env_params.vector_env:
        env: gym.Env = gym.make(config.env_params.env_name, **config.env_params.misc_arguments)
    else:
        env: gym.Env = gym.vector.make(
            id = config.env_params.env_name, 
            num_envs=config.env_params.num_envs,
            wrappers=[
                lambda env, env_id=i: RecordVideoWrapper(env, recording_length=128, enabled=(env_id==0)) for i in range(config.env_params.num_envs)
            ],
            **config.env_params.misc_arguments
        )

    # Load agent
    agent_class_ = getattr(importlib.import_module("src.Agents"), config.agent_params.agent_name)
    agent = agent_class_(observation_space=env.observation_space, action_space = env.action_space, hyperparams=config.agent_params, device=device)

    # Run a few random batches to generate normalization parameters
    running_mean_std_recorder = RunningMeanStd(shape=env.observation_space.shape, device=device)
    if config.env_params.env_normalization:
        state, _ = env.reset()
        with torch.no_grad():
            for _ in range(512):
                action = env.unwrapped.action_space.sample()
                next_state, _, _, _, _ = env.step(action)
                running_mean_std_recorder.update(next_state)
                state = next_state  
        running_mean_std_recorder.save(loc=config.trainer_params.save_location + "NormStates")
    
    # Train agent
    trainer = Trainer(
        agent,
        env,
        config.trainer_params,
        config.env_params,
        save_location=config.trainer_params.save_location,
        normalizer=running_mean_std_recorder,
        device=device
    )
    pbar = tqdm(total=config.trainer_params.num_epochs)
    for epoch in trainer:
        pbar.update(1)

    pbar.close()
    env.close()

    # Test the policy
    # env = gym.Env = gym.make(config.env_params.env_name)
    # test_policy(env, agent, normalizor=running_mean_std_recorder if config.env_params.env_normalization else None)