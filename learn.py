import sys
import os
import argparse
import importlib
from tqdm import tqdm
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

# Make accessible 'src' and its modules
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.append(src_path)

from src.TrainPresets import *
from src.TrainPresets import Config
from src.Trainer import Trainer
from src.Utilities import set_random_seeds, RunningMeanStd, clear_directories

from src.wrappers import RecordVideoWrapper, ActionRepeatWrapper # FrameStack

import torch
import gymnasium as gym
from gymnasium.wrappers import TimeLimit, AutoResetWrapper, FrameStack, FlattenObservation # type: ignore

# Setup arguements
parser = argparse.ArgumentParser(description='Description of your program.')
parser.add_argument('-c', '--config', type=str, help='Train configuration class', required=True)


if __name__ == "__main__":
    
    # Misc setup
    set_random_seeds()
    clear_directories(['videos', 'saved_models']) # , 'runs'])  # TODO Make this a command line argument.
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
        env: gym.Env = gym.make(config.env_params.env_name, **config.env_params.misc_arguments) # type: ignore
    else:
        def apply_time_limit(env, timelimit = 1024):
            return TimeLimit(env, timelimit)
        
        def apply_action_repeat(env, num_repeats = 1):
            return ActionRepeatWrapper(env, num_repeats)
        
        def apply_frame_stack(env, num_frames = 16):
            return FrameStack(env, num_frames)
        
        env: gym.Env = gym.vector.make(
            id = config.env_params.env_name, 
            num_envs=config.env_params.num_envs,
            wrappers=[
                apply_time_limit, # type: ignore
                apply_action_repeat,
                AutoResetWrapper, 
                RecordVideoWrapper,
                apply_frame_stack, 
                FlattenObservation
            ],
            **config.env_params.misc_arguments # type: ignore
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
    # try:
    #     for epoch in trainer:
    #         running_mean_std_recorder.save(loc=config.trainer_params.save_location + "NormStates")
    #         pbar.update(1)
    # except:
    #     print("An error was encourtered during the training process")
    # finally:
    #     pbar.close()
    #     env.close()