from typing import Optional, Union, ClassVar, Dict, Any
import jax
import numpy as np
import cv2
import os

import gymnasium as gym
from gymnasium import spaces

import torch
from brax.v1.io import torch as b_torch
from brax.v1.envs import env as brax_env
from brax.v1 import jumpy as jp

from Utilities import to_tensor

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
    self._recording = False
    self._image_buffer = []
    self._session_video = cv2.VideoWriter(
        os.path.join(self._record_location, self._record_name_prefix) + '_' + str(self._episode) + '.mp4' ,
        fourcc = cv2.VideoWriter_fourcc(*'avc1'),
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
                    fourcc = cv2.VideoWriter_fourcc(*'avc1'), 
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

  def reset(self) -> tuple[torch.Tensor, dict]:
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