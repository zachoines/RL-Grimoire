import os
import cv2
from gymnasium.utils.save_video import save_video
import gymnasium as gym
from gymnasium.spaces import Box
from datetime import datetime
import torch
from typing import Any, List, Tuple, Union, Dict
import numpy as np


class FrameStack(gym.Wrapper):
    def __init__(self, env: gym.Env, stack_size: int = 8, device: torch.device = torch.device("cpu")):
        """
        Initialize a FrameStack wrapper for Gym environments.
        
        Args:
            env (gym.Env): the Gym environment.
            stack_size (int, optional): the number of frames to stack. Default is 2.
            device (torch.device, optional): the device to use for PyTorch tensors. Default is 'cpu'.
        """
        super().__init__(env)
        self.stack_size = stack_size
        self.device = device
        
        # Update the observation space
        low_bound = self.env.observation_space.low if self.env.observation_space.low is not None else float('-inf')
        high_bound = self.env.observation_space.high if self.env.observation_space.high is not None else float('inf')

        low = np.tile(low_bound, self.stack_size)
        high = np.tile(high_bound, self.stack_size)

        self.stack_shape = tuple([self.stack_size] + list(env.observation_space.shape))

        # Update the observation space
        self.observation_space = Box(low=low,
                                     high=high,
                                     shape=low.shape,
                                     dtype=self.env.observation_space.dtype)
        
        self.frames = torch.zeros(
            self.stack_shape,
            dtype=torch.float32,
            device=self.device
        )

    def reset(self) -> Tuple[torch.Tensor, Dict]:
        """
        Reset the environment and return the initial observation.
        
        Returns:
            torch.Tensor: the initial observation from the environment.
        """
        observation, info = self.env.reset()
        self.frames = torch.zeros(
            self.stack_shape,
            dtype=torch.float32,
            device=self.device
        )

        for _ in range(self.stack_size):
            self._push_frame(observation)

        return self._get_stacked_frames().reshape(-1), info

    def step(self, action: Union[int, float]) -> Tuple[torch.Tensor, float, bool, bool, Dict]:
        """
        Take a step in the environment using the given action.
        
        Args:
            action (int or float): the action to take in the environment.
        
        Returns:
            tuple: a tuple containing the next observation, the reward, the done flag, the info dictionary, and the truncation flag.
        """
        observation, reward, done, truncation, info = self.env.step(action)
        self._push_frame(observation)

        return self._get_stacked_frames().reshape(-1), reward, done, truncation, info

    def _push_frame(self, frame: Union[np.ndarray, torch.Tensor]):
        """
        Push a new frame into the frame buffer.
        
        Args:
            frame (numpy.ndarray or torch.Tensor): the frame to push into the buffer.
        """
        if type(frame) != torch.Tensor:
            frame = torch.tensor(frame, dtype=torch.float32, device=self.device)

        self.frames[:-1] = self.frames[1:].clone()
        self.frames[-1] = frame

    def _get_stacked_frames(self) -> torch.Tensor:
        """
        Get the current stack of frames.
        
        Returns:
            torch.Tensor: the current stack of frames.
        """
        return self.frames.permute([1, 2, 0]) if len(self.frames.shape) > 2 else self.frames

class ActionRepeatWrapper(gym.Wrapper):
    def __init__(self, env, repeat):
        super(ActionRepeatWrapper, self).__init__(env)
        self.repeat = repeat

    def step(self, action):
        total_reward = None
        done = False
        for _ in range(self.repeat):
            obs, reward, done, trunc, info = self.env.step(action)
            if total_reward == None:
                total_reward = reward
            else:
                total_reward += reward
            if done or trunc:
                break
        return obs, total_reward, done, trunc, info

class RecordVideoWrapper(gym.Wrapper):
    recorder_id = 0
    def __init__(self, env : gym.Env, save_folder: str = "videos", recording_length: int = 512):
        super().__init__(env)
        RecordVideoWrapper.recorder_id += 1
        self.id = RecordVideoWrapper.recorder_id
        self.save_folder = save_folder
        self.step_count = 0
        self.episode_counter = 0
        self.enabled = True if self.id == 1 else False
        self.recording_length = recording_length
        self.frames = []
        self.env = env
        if self.enabled:
            os.makedirs(self.save_folder, exist_ok=True)

    def reset(self, **kwargs):
        self.episode_counter += 1
        return super().reset(**kwargs)
    
    def close(self):
        if self.enabled:
            if len(self.frames) > 1:
                self._save()

    def step(self, action):
        observation, reward, done, truncs, info = super().step(action)
        if done:
            self.episode_counter += 1

        if self.enabled:
            self.frames.append(self.env.render())
            if len(self.frames) >= self.recording_length:
                self._save()

        return observation, reward, done, truncs, info

    def _save(self):
        """Save a video of the current episode."""
        # image_size = self.frames[0].shape
        # self._session_video = cv2.VideoWriter(
        #     os.path.join(self.save_folder, self.env.spec.id) + '_' + str(self.step_count) + '.mp4' ,
        #     fourcc = cv2.VideoWriter_fourcc(*'xvid'), 
        #     fps = 30,
        #     frameSize = image_size[:2],
        # )
        # for image in self.frames:
        #     self._session_video.write(image)
        # self._session_video.release()
        now = datetime.now()
        datetime_str = now.strftime("%Y%m%d_%H%M%S")
        save_video(
            frames = self.frames,
            video_folder=self.save_folder,
            fps=self.env.metadata.get("render_fps", 30),
            name_prefix=datetime_str
        )
        self.frames = []