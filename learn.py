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
from src.Utilities import set_random_seeds, RunningMeanStd, clear_directories
from src.wrappers import RecordVideoWrapper

import torch
import gymnasium as gym
from gymnasium.wrappers.normalize import NormalizeReward, NormalizeObservation

# Setup arguements
parser = argparse.ArgumentParser(description='Description of your program.')
parser.add_argument('-c', '--config', type=str, help='Train configuration class', required=True)

# Register environments here
gym.register(
    id="brax-swimmer",
    entry_point="src.brax_to_gymnasium:convert_brax_to_gym",
    kwargs={
        "name": "swimmer",
        "episode_length": 1024,
        "action_repeat": 1,
        "forward_reward_weight": 1.0,
        "ctrl_cost_weight": 1e-4,
        "reset_noise_scale": 0.1,
        "exclude_current_positions_from_observation": True,
        "legacy_reward": False,
        "legacy_spring": False,
    }
)

gym.register(
    id="brax-humanoid-standup",
    entry_point="src.brax_to_gymnasium:convert_brax_to_gym",
    kwargs={
        "name": "humanoidstandup",
        "episode_length": 1024,
        "action_repeat": 1,
        "legacy_spring": False
    }
)

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
        "terminate_when_unhealthy": True,
        "exclude_current_positions_from_observation": True,
        "action_repeat": 1
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
        "terminate_when_unhealthy": True,
        "reset_noise_scale": 5e-3,
        "exclude_current_positions_from_observation": True,
        "action_repeat": 1,
        "legacy_spring": False,
        "healthy_z_range": (.7, float('inf'))
    }
)

gym.register(
    id="half-cheetah-hopper",
    entry_point="src.brax_to_gymnasium:convert_brax_to_gym",
    kwargs={
        "name": "hopper",
        "episode_length": 1024,
        "forward_reward_weight": 1.0,
        "ctrl_cost_weight": 1e-3,
        "healthy_reward": 1.0,
        "terminate_when_unhealthy": True,
        "reset_noise_scale": 5e-3,
        "exclude_current_positions_from_observation": True,
        "action_repeat": 1,
        "healthy_z_range": (.7, float('inf'))
    }
)

gym.register(
    id="brax-humanoid",
    entry_point="src.brax_to_gymnasium:convert_brax_to_gym",
    kwargs={
        "name": "humanoid",
        "episode_length": 1024,
        "ctrl_cost_weight": 0.1,
        "forward_reward_weight": 1.25,
        "healthy_reward": 5.0,
        "terminate_when_unhealthy": True,
        "reset_noise_scale": 1e-2,
        "exclude_current_positions_from_observation": True,
        "action_repeat": 1,
        
    }
)

gym.register(
    id="brax-half-cheetah",
    entry_point="src.brax_to_gymnasium:convert_brax_to_gym",
    kwargs={
        "name": "halfcheetah",
        "episode_length": 1024,
        "forward_reward_weight": 1.0,
        "ctrl_cost_weight": 1e-3,
        "legacy_spring" : False,
        "reset_noise_scale": 5e-3,
        "exclude_current_positions_from_observation": True,
        # "action_repeat": 1
    }
)

if __name__ == "__main__":
    
    # Misc setup
    set_random_seeds()
    clear_directories()  # TODO Make this a command line argument.
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
                lambda env, env_id=i: RecordVideoWrapper(env, recording_length=2048, enabled=(True)) if env_id==0 else env for i in range(config.env_params.num_envs)
            ],
            **config.env_params.misc_arguments
        )

    # Load agent
    agent_class_ = getattr(importlib.import_module("src.Agents"), config.agent_params.agent_name)
    agent = agent_class_(observation_space=env.observation_space, action_space = env.action_space, hyperparams=config.agent_params, device=device)

    # Run a few random batches to generate normalization parameters
    running_mean_std_recorder = RunningMeanStd(shape=env.observation_space.shape, device=device)
    
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
        running_mean_std_recorder.save(loc=config.trainer_params.save_location + "NormStates")
        pbar.update(1)

    pbar.close()
    env.close()

    # Test the policy
    # env = gym.Env = gym.make(config.env_params.env_name)
    # test_policy(env, agent, normalizor=running_mean_std_recorder if config.env_params.env_normalization else None)