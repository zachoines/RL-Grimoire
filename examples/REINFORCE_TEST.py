import torch
import gymnasium as gym
import multiprocessing
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from gymnasium.wrappers.normalize import NormalizeObservation, NormalizeReward
from gymnasium.spaces import Discrete
from tqdm import tqdm

from Hyperparams import Hyperparams
from Agents import REINFORCE
from Trainer import Trainer
from Datasets import ExperienceBuffer

def create_environments(num_envs=1, env_name="CartPole-v1"):
    env = gym.vector.make(env_name, num_envs=num_envs)
    # env = RecordEpisodeStatistics(env)
    # env = NormalizeObservation(env)
    # env = NormalizeReward(env)
    return env
    
if __name__ == "__main__":
    exp_buffer_max_size = 10000
    num_actions = 2

    multiprocessing.freeze_support()
    device = torch.device(
        "mps" if torch.has_mps else "cpu" or # MACOS
        "cuda" if torch.has_cuda else 
        "cpu"
    )

    hyperparams = Hyperparams() # Default params is ok
    exp_buffer = ExperienceBuffer(exp_buffer_max_size)
    env = create_environments(num_envs=hyperparams.num_envs, env_name="CartPole-v1")
    agent = REINFORCE(state_size=4, hidden_size=256, num_actions=num_actions, action_type = Discrete(num_actions),device=device)
    trainer = Trainer(agent, exp_buffer, env, hyperparams)

    pbar = tqdm(total=hyperparams.num_epochs)
    for epoch in trainer:
        pbar.update(1)
    
    pbar.close()
    env.close()