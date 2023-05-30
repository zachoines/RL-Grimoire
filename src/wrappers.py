import os
from gymnasium.utils.save_video import save_video
import gymnasium as gym
from datetime import datetime

class RecordVideoWrapper(gym.Wrapper):
    def __init__(self, env, save_folder="videos", recording_length=100, enabled=True):
        super().__init__(env)
        self.save_folder = save_folder
        self.step_count = 0
        self.episode_counter = 0
        self.enabled = enabled
        self.recording_length = recording_length
        self.frames = []
        os.makedirs(self.save_folder, exist_ok=True)

    def reset(self, **kwargs):
        self.step_counter = 0
        self.episode_counter += 1
        return super().reset(**kwargs)

    def step(self, action):
        observation, reward, done, truncs, info = super().step(action)
        if done:
            self.episode_counter += 1
            self.step_counter = 0

        if self.enabled:
            self.frames.append(self.env.render())
            if len(self.frames) >= self.recording_length:
                self._save()
                self.frames = []

        return observation, reward, done, truncs, info

    def _save(self):
        """Save a video of the current episode."""
        now = datetime.now()
        datetime_str = now.strftime("%Y%m%d_%H%M%S")
        save_video(
            frames = self.frames,
            video_folder=self.save_folder,
            fps=self.env.metadata.get("render_fps", 30),
            # step_starting_index=self.step_counter - self.recording_length,
            # episode_index=self.episode_counter,
            name_prefix=datetime_str
        )
