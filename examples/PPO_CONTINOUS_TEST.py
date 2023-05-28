# Note: may need to run: export PYTHONPATH="RL-Grimoire/src"

import torch
import gymnasium as gym
import multiprocessing
import numpy as np
from tqdm import tqdm
from gymnasium.wrappers import RecordVideo # type: ignore

from TrainPresets import PPOBraxAntConfig as Config
from Agents import PPO as Agent
from Trainer import Trainer
from Utilities import test_policy, set_random_seeds, RunningMeanStd
from datetime import datetime
from brax_to_gymnasium import JaxToTorchWrapper, VectorGymWrapper

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
        use_contact_forces=False
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
    
    # Test the environment
    # env = gym.make(config.trainer_params.env_name, render_mode="rgb_array")
    # agent = Agent(observation_space = env.observation_space, action_space = env.action_space, hyperparams=config.agent_params, device=device)
    # agent.load(config.trainer_params.save_location)
    # test_policy(env, agent, normalization_weights=config.trainer_params.env_normalization_weights)

    # env = gym.vector.make(config.trainer_params.env_name, num_envs=config.trainer_params.num_envs)
    # Test callback
    # def test_callback(agent: Agent):
    #     current_datetime = datetime.now()
    #     current_date_string = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    #     test_env = gym.make(config.trainer_params.env_name, render_mode="rgb_array")
    #     test_env = RecordVideo(
    #         test_env,
    #         video_folder='videos/',
    #         name_prefix=f"{config.trainer_params.env_name}_{current_date_string}"
    #     )

    #     state, _ = test_env.reset()
    #     with torch.no_grad():
    #         for _ in range(512):
    #             if np.any(config.trainer_params.env_normalization_weights):
    #                 state *= config.trainer_params.env_normalization_weights
    #             action, _ = agent.get_actions(state, eval=False)
    #             next_state, _, _, _, _ = test_env.step(action.cpu().numpy())
    #             state = next_state
            
    #     test_env.close()