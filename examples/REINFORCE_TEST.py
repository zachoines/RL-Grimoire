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

def test_policy(env, agent, num_episodes):

    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        
        while not done:
            env.render()
            action = agent.get_actions(state)    
            next_state, reward, done, _, _ = env.step(action.item())
            state = next_state

    
if __name__ == "__main__":
    env_name="CartPole-v1"
    exp_buffer_max_size = 10000
    num_actions = 2
    state_size = 4
    save_location = "RL-Grimoire/saved_models/REINFORCE_CARTPOLE"

    multiprocessing.freeze_support()
    device = torch.device(
        "mps" if torch.has_mps else "cpu" or # MACOS
        "cuda" if torch.has_cuda else 
        "cpu"
    )

    hyperparams = Hyperparams() # Default params is ok
    exp_buffer = ExperienceBuffer(exp_buffer_max_size)
    env = create_environments(num_envs=hyperparams.num_envs, env_name=env_name)
    agent = REINFORCE(state_size=state_size, hidden_size=256, num_actions=num_actions, action_type = Discrete(num_actions),device=device)
    trainer = Trainer(agent, exp_buffer, env, hyperparams, save_location=save_location)

    # Train the environment
    pbar = tqdm(total=hyperparams.num_epochs)
    for epoch in trainer:
        pbar.update(1)
    
    pbar.close()
    env.close()
    
    # Test the environment
    agent.load(save_location)
    env = gym.make("CartPole-v1", render_mode="human")
    test_policy(env, agent, 100)