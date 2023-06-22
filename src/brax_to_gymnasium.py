from typing import Optional, ClassVar, Dict, Any
import jax
import numpy as np
import cv2
import os
from collections import deque

import gymnasium as gym
from gymnasium import spaces

import torch
from brax.v1.io import torch as b_torch
from brax.v1.envs import env as brax_env
from brax.v1 import jumpy as jp
import brax.v1.envs as envs
from datetime import datetime

def convert_brax_to_gym(name: str, frame_stack: bool = True, stack_size: int = 8, **kwargs):
    device = torch.device(
        "mps" if torch.has_mps else "cpu" or # MACOS
        "cuda" if torch.has_cuda else 
        "cpu"
    )
    # Create the environment
    current_datetime = datetime.now()
    current_date_string = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    env = envs.create(
        name,
        **kwargs
    )
    env = VectorGymWrapper(
        env,
        record = True, 
        record_location = 'videos/',
        record_name_prefix = f"{name}_{current_date_string}",
        recording_save_frequeny = 512
    )
    env = JaxToTorchWrapper(env, device)

    if frame_stack:
       env = BraxFrameStack(env, stack_size, device=device)  # Add this line

    return env


class BraxFrameStack(gym.Wrapper):
    def __init__(self, env: gym.Env, stack_size: int, device: torch.device = torch.device("cpu")):
        super().__init__(env)
        self.stack_size = stack_size
        self.num_envs = env.num_envs
        self.device = device

        self.stacks = [torch.zeros((self.stack_size, self.env.observation_space.shape[-1]), dtype=torch.float32).to(self.device) for _ in range(self.num_envs)]
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.num_envs, self.stack_size * self.env.observation_space.shape[-1]), 
            dtype=np.float32
        )

    def reset(self):
        obs, info = self.env.reset()
        self.stacks = [torch.zeros((self.stack_size, self.env.observation_space.shape[-1]), dtype=torch.float32).to(self.device) for _ in range(self.num_envs)]
        for i in range(self.num_envs):
            self.stacks[i][-1] = obs[i, :].to(self.device)
        return self.get_obs(), info

    def step(self, action):
        obs, reward, done, truncs, info = self.env.step(action)

        # Create a list to hold the information about whether each episode is done or truncated
        reset_indices = []

        for i in range(self.num_envs):
            self.stacks[i] = torch.cat((self.stacks[i][1:], obs[i, :].unsqueeze(0).to(self.device)))

            # If the episode is done or truncated, add the index to the list
            if done[i] or truncs[i]:
                reset_indices.append(i)

        # Store the current state of the stacks
        curr_obs = self.get_obs()

        # Now, reset the stacks for the environments where the episode is done or truncated
        for i in reset_indices:
            self.stacks[i] = torch.zeros((self.stack_size, self.env.observation_space.shape[-1]), dtype=torch.float32).to(self.device)

        return curr_obs, reward, done, truncs, info

    def get_obs(self):
        # Convert the list of stacks into a 3D tensor of shape (num_envs, stack_size, state_size)
        obs = torch.stack(self.stacks)

        # Reshape the tensor to be of shape (num_envs, stack_size * state_size)
        obs = obs.view(self.num_envs, -1)

        return obs



class BraxFrameStackV1(gym.Wrapper):
    def __init__(self, env: gym.Env, stack_size: int, device: torch.device = torch.device("cpu")):
        super().__init__(env)
        self.stack_size = stack_size  # Number of frames to stack
        self.num_envs = env.num_envs  # Number of environments
        self.device = device  # Device to use for computations

        # Initialize a deque for each environment to store the frames
        self.frames = [deque([torch.zeros(self.env.observation_space.shape[-1], dtype=torch.float32).to(self.device) for _ in range(self.stack_size)], maxlen=stack_size) for _ in range(self.num_envs)]
        
        # Modify the observation space to account for the frame stacking
        self.observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.num_envs, self.stack_size * self.env.observation_space.shape[-1]), 
            dtype=np.float32
        )

    def reset(self):
        obs, info = self.env.reset()  # Reset the environment
        # Reset the frame deque for each environment
        for i in range(self.num_envs):
            self.frames[i] = deque([torch.zeros(self.env.observation_space.shape[-1], dtype=torch.float32).to(self.device) for _ in range(self.stack_size)], maxlen=self.stack_size)
            self.frames[i].append(obs[i, :].to(self.device))  # Append the initial observation to the deque
        return self.get_obs(), info  # Return the stacked frames and the info

    def step(self, action):
        obs, reward, done, truncs, info = self.env.step(action)  # Step the environment with the action
        # If the episode is done or truncated, reset the frame deque
        # Otherwise, append the new observation to the deque
        for i in range(self.num_envs):
            if done[i] or truncs[i]:
                self.frames[i] = deque([torch.zeros(self.env.observation_space.shape[-1], dtype=torch.float32).to(self.device) for _ in range(self.stack_size)], maxlen=self.stack_size)
            self.frames[i].append(obs[i, :].to(self.device))
        return self.get_obs(), reward, done, truncs, info  # Return the stacked frames, reward, done flag, truncations, and info

    def get_obs(self):
        # Convert the list of stacks into a 3D tensor of shape (num_envs, stack_size, state_size)
        obs = torch.stack(self.stacks)

        # Reshape the tensor to be of shape (num_envs, stack_size * state_size)
        obs = obs.view(self.num_envs, -1)

        return obs


class GymWrapper(gym.Env):

  # Flag that prevents `gym.register` from misinterpreting the `_step` and
  # `_reset` as signs of a deprecated gym Env API.
  _gym_disable_underscore_compat: ClassVar[bool] = True

  def __init__(self,
               env: brax_env.Env,
               seed: int = 0,
               backend: Optional[str] = None):
    self._env = env
    self.metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }
    self.seed(seed)
    self.backend = backend
    self._state = None

    obs_high = jp.inf * jp.ones(self._env.observation_size, dtype=float)
    self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

    action_high = jp.ones(self._env.action_size, dtype=float)
    self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)

    def reset(key):
      key1, key2 = jp.random_split(key)
      state = self._env.reset(key2)
      return state, state.obs, key1

    self._reset = jax.jit(reset, backend=self.backend)

    def step(state, action):
      state = self._env.step(state, action)
      info = {**state.metrics, **state.info}
      return state, state.obs, state.reward, state.done, {}, info

    self._step = jax.jit(step, backend=self.backend)

  def reset(self):
    self._state, obs, self._key = self._reset(self._key)
    # We return device arrays for pytorch users.
    return obs, {}

  def step(self, action):
    self._state, obs, reward, done, info = self._step(self._state, action)
    # We return device arrays for pytorch users.
    return obs, reward, done, {}, info

  def seed(self, seed: int = 0):
    self._key = jax.random.PRNGKey(seed)

  def render(self, mode='human'):
    # pylint:disable=g-import-not-at-top
    from brax.v1.io import image
    if mode == 'rgb_array':
      sys, qp = self._env.sys, self._state.qp # type: ignore
      return image.render_array(sys, qp, 256, 256)
    else:
      raise NotImplementedError
      # return super().render(mode=mode)  # just raise an exception

class VectorGymWrapper(gym.vector.VectorEnv):
  """A wrapper that converts batched Brax Env to one that follows Gym VectorEnv API."""

  # Flag that prevents `gym.register` from misinterpreting the `_step` and
  # `_reset` as signs of a deprecated gym Env API.
  _gym_disable_underscore_compat: ClassVar[bool] = True

  def __init__(self,
        env: brax_env.Env,
        seed: int = 0,
        backend: Optional[str] = None, 
        record: bool = True, 
        record_location: str = 'videos/',
        record_name_prefix: str = '',
        recording_save_frequeny: int = 256
    ):

    # Record related variables 
    self._record = record
    self._record_location = record_location
    self._record_name_prefix = record_name_prefix
    self._recording_save_frequeny = recording_save_frequeny
    self._steps = 0
    self._episode = 0
    self._recording = True
    self._image_buffer = []
    self._session_video = cv2.VideoWriter(
        os.path.join(self._record_location, self._record_name_prefix) + '_' + str(self._episode) + '.mp4' ,
        fourcc = cv2.VideoWriter_fourcc(*'xvid'),
        fps = 20, 
        frameSize = (256, 256)
    )

    self._env = env
    self.metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 1 / self._env.sys.config.dt # type: ignore
    }
    if not hasattr(self._env, 'batch_size'):
      raise ValueError('underlying env must be batched')

    self.num_envs = self._env.batch_size # type: ignore
    self.seed(seed)
    self.backend = backend
    self._state = None

    obs_high = jp.inf * jp.ones(self._env.observation_size)
    self.single_observation_space = gym.spaces.Box(
        -obs_high, obs_high, dtype=np.float32)
    self.observation_space = gym.vector.utils.batch_space(self.single_observation_space,
                                               self.num_envs)

    action_high = jp.ones(self._env.action_size, dtype=float)
    self.single_action_space = gym.spaces.Box(
        -action_high, action_high, dtype=np.float32)
    self.action_space = gym.vector.utils.batch_space(self.single_action_space,
                                          self.num_envs)

    def reset(key):
      key1, key2 = jp.random_split(key)
      state = self._env.reset(key2)
      return state, state.obs, key1

    self._reset = jax.jit(reset, backend=self.backend)

    def step(state, action):
      state = self._env.step(state, action)
      info = {**state.metrics, **state.info}
      return state, state.obs, state.reward, state.done, state.info["truncation"], info

    self._step = jax.jit(step, backend=self.backend)

  def reset(self):
    self._state, obs, self._key = self._reset(self._key)
    return obs, {}
  
  def release_video(self):
    if self._record:
        if self._recording:
            if len(self._image_buffer) > 1:
                self._episode +=1
                for image in self._image_buffer:
                    self._session_video.write(image)
                self._session_video.release()
                self._session_video = cv2.VideoWriter(
                    os.path.join(self._record_location, self._record_name_prefix) + '_' + str(self._episode) + '.mp4' ,
                    fourcc = cv2.VideoWriter_fourcc(*'xvid'), 
                    fps = 20, 
                    frameSize = (256, 256)
                )
                self._image_buffer = []
        else:
            self._recording = True
  
  def close(self):
    self.release_video()

  def step(self, action):
    self._steps += 1
    self._state, obs, reward, done, truncs, info = self._step(self._state, action)

    if self._record and self._recording:
        self._image_buffer.append(self.render('rgb_array'))

        if (self._steps % self._recording_save_frequeny) == 0:
            self.release_video()
        
    return obs, reward, done, truncs, info

  def seed(self, seed: int = 0):
    self._key = jax.random.PRNGKey(seed)

  def toggle_recording(self, state: bool):
    if self._record:
        self._recording = state

  def render(self, mode='human'):
    from brax.v1.io import image
    sys = self._env.sys
    qp = jp.take(self._state.qp, 0)  # type: ignore
    if mode == 'rgb_array':
      return image.render_array(sys, qp, 256, 256)
    else:
      cv2.imshow('Frame', image.render_array(sys, qp, 256, 256))
      cv2.waitKey(1)

class JaxToTorchWrapper(gym.Wrapper):

  def __init__(self,
               env: GymWrapper | VectorGymWrapper,
               device: Optional[b_torch.Device] = None):
    super().__init__(env)
    self.device: Optional[b_torch.Device] = device
    self.env = env

  def observation(self, observation) -> torch.Tensor:
    return b_torch.jax_to_torch(observation, device=self.device)

  def action(self, action: torch.Tensor):
    return b_torch.torch_to_jax(action)

  def reward(self, reward)-> torch.Tensor:
    return b_torch.jax_to_torch(reward, device=self.device)

  def done(self, done)-> torch.Tensor:
    return b_torch.jax_to_torch(done, device=self.device)

  def info(self, info)-> torch.Tensor:
    return b_torch.jax_to_torch(info, device=self.device)
  
  def truncation(self, truncs)-> torch.Tensor:
    return b_torch.jax_to_torch(truncs, device=self.device)

  def reset(self, **kwargs) -> tuple[torch.Tensor, dict]:
    obs, info = self.env.reset()
    return self.observation(obs), info

  def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    action = self.action(action)
    obs, reward, done, truncs, info= super().step(action)
    obs = self.observation(obs)
    reward = self.reward(reward)
    done = self.done(done)
    info = self.info(info)
    truncs = self.truncation(truncs)
    return obs, reward, done, truncs, info