import torch
import numpy as np
import gymnasium as gym
import multiprocessing
from tqdm import tqdm
from gymnasium.wrappers import RecordVideo # type: ignore

from Configurations import REINFORCEHalfCheetahConfig
from Agents import REINFORCE
from Trainer import Trainer
import random

def test_policy(env, agent, num_episodes, normalization_weights, video=False):
    if video:
        env = RecordVideo(env, 'videos', episode_trigger=lambda e: True)
    with torch.no_grad():
        for _ in range(num_episodes):
            state, _ = env.reset()
            done = False
            steps = 0
            while not done:
                steps += 1
                env.render()
                if np.any(normalization_weights):
                    state *= normalization_weights
                action = agent.get_actions(state, eval=True).cpu().numpy()
                next_state, _, done, _, _ = env.step(action)
                state = next_state

    
if __name__ == "__main__":

    # Misc
    test=True
    multiprocessing.freeze_support()

    # Set the Random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Hyperparams
    config = REINFORCEHalfCheetahConfig()
    
    device = torch.device(
        "mps" if torch.has_mps else "cpu" or # MACOS
        "cuda" if torch.has_cuda else 
        "cpu"
    )

    # env = None
    if not test:
        if config.trainer_params.num_envs == 1:
            env = gym.make(config.trainer_params.env_name, render_mode="human")
        else:
            env = gym.vector.make(config.trainer_params.env_name, num_envs=config.trainer_params.num_envs)

        action_space = env.action_space
        observation_space = env.observation_space
        agent = REINFORCE(observation_space=observation_space, action_space = action_space, hyperparams=config.agent_params, device=device)
        trainer = Trainer(agent, env, config.trainer_params, save_location=config.trainer_params.save_location)

        # Train the environment
        pbar = tqdm(total=config.trainer_params.num_epochs)
        for epoch in trainer:
            pbar.update(1)
        
        pbar.close()
        env.close()
    
    # Test the environment
    env = gym.make(config.trainer_params.env_name, render_mode="human")
    action_space = env.action_space
    observation_space = env.observation_space
    agent = REINFORCE(observation_space = observation_space, action_space = action_space, hyperparams=config.agent_params, device=device)
    agent.load(config.trainer_params.save_location)
    test_policy(env, agent, 100, config.trainer_params.env_normalization_weights)
    env.close()