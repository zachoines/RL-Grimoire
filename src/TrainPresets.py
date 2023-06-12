from Configurations import *
import torch
from typing import Annotated

class REINFORCECartpoleConfig(Config):
    def __init__(self):
        super().__init__(
            REINFORCEParams(
                gamma = 0.99,
                policy_learning_rate = 1e-4,
                value_learning_rate = 1e-3,
                entropy_coefficient = 0.01,
                hidden_size = 64
            ),
            TrainerParams(
                batch_transitions_by_env_trajectory = True,
                num_epochs = 1000,
                batches_per_epoch = 8,
                batch_size = 32,
                shuffle_batches = True,
                render = False,
                save_location = "./saved_models/CartPoleREINFORCE",
                squeeze_actions = True,
            ),
            EnvParams(
                env_name = "CartPole-v1",
                num_envs = 64,
                max_episode_steps = 1000
            )
        )

class REINFORCEHalfCheetahConfig(Config):
    def __init__(self):
        super().__init__(
            REINFORCEParams(
                gamma = 0.99,
                policy_learning_rate = 5e-5,
                value_learning_rate = 5e-5,
                entropy_coefficient = 0.005,
                hidden_size = 96
            ),
            TrainerParams(
                batch_transitions_by_env_trajectory = True,
                num_epochs = 100,
                batches_per_epoch = 512,
                batch_size = 64,
                shuffle_batches = True,
                render = False,
                save_location = "./saved_models/HalfCheetahREINFORCE"
            ),
            EnvParams(
                env_name = "HalfCheetah-v4",
                num_envs = 64,
                max_episode_steps = 128
            )
        )

# Revisit this config
class A2CInvertedDoublePendulumConfig(Config):
    def __init__(self):
        super().__init__(
            A2CParams(
                tau = 0.1,
                gamma = 0.99,
                policy_learning_rate = 0.01,
                value_learning_rate = 0.01,
                entropy_coefficient = 0.01,
                hidden_size = 128
            ),
            TrainerParams(
                # learningRateScheduler = True,
                # learningRateSchedulerClass="StepLR",
                # learningRateScheduleArgs={ "step_size" : 128, "gamma": 0.99},
                on_policy_training = True,
                num_epochs = 100,
                batches_per_epoch = 16,
                batch_size = 128,
                shuffle_batches=True,
                save_location = "./saved_models/InvertedDoublePendulumAC2"
            ),
            EnvParams(
                env_name = "InvertedDoublePendulum-v4",
                num_envs = 4,
                misc_arguments = {
                    "max_episode_steps": 512,
                    "render_mode": "rgb_array"
                }
            )
        )

class PPOBraxAntConfig(Config):
    def __init__(self):
        self.max_episode_steps = 1024
        self.num_envs = 256
        super().__init__(
            PPOParams(
                tau = 0.05,
                clip = 0.3,
                gamma = 0.9,
                policy_learning_rate = 1e-5,
                value_learning_rate = 1e-4,
                entropy_coefficient = 0.05,
                hidden_size = 512
            ),
            TrainerParams(
                num_epochs = 1000,
                batches_per_epoch = 16,
                batch_size = 1024,
                updates_per_batch = 8,
                shuffle_batches = True,
                record_video_frequency=10,
                save_location = "./saved_models/AntPPO",
            ),
            EnvParams(
                env_name = "brax-ant",
                env_normalization=True,
                num_envs = self.num_envs,
                max_episode_steps = self.max_episode_steps,
                vector_env=False, # Brax will init 'n' environments on its side
                misc_arguments = {
                    "batch_size": self.num_envs, # Brax's convention uses batch_size for num_environments
                    "episode_length": self.max_episode_steps,
                    "healthy_reward": 1.5,
                    "ctrl_cost_weight": .5,
                    "contact_cost_weight": 1e-4,
                    "use_contact_forces": False,
                    "terminate_when_unhealthy": True
                }
            )
        )

class PPO2BraxAntConfig(Config):
    def __init__(self):
        self.max_episode_steps = 1024
        self.num_envs = 1024

        super().__init__(
            PPO2Params(
                clip = 0.3,
                gamma = 0.99,
                policy_learning_rate = 2e-4,
                value_learning_rate = 2e-3,
                entropy_coefficient = 0.01,
                hidden_size = 256,
                gae_lambda = 0.95
            ),
            TrainerParams(
                batch_transitions_by_env_trajectory = True, # Must be enabled for PPO
                num_epochs = 1000,
                batches_per_epoch = 1,
                batch_size = 64,
                updates_per_batch = 1,
                shuffle_batches = False, # False to not interfere with GAE creation
                save_location = "./saved_models/AntPPO"
            ),
            EnvParams(
                env_name = "brax-ant",
                env_normalization=False,
                num_envs = self.num_envs,
                max_episode_steps = self.max_episode_steps,
                vector_env=False, # Brax will init 'n' environments on its side
                misc_arguments = {
                    "batch_size": self.num_envs, # Brax's convention uses batch_size for num_environments
                    "episode_length": self.max_episode_steps,
                    "healthy_reward": 0.50, # 1.00,
                    "ctrl_cost_weight": 0.25, # .5,
                    "contact_cost_weight": 2.5e-4, # 5e-4,
                    "use_contact_forces": True,
                    "terminate_when_unhealthy": True
                }
            )
        )


class PPO2BraxHopperConfig(Config):
    def __init__(self):
        self.max_episode_steps = 512
        self.num_envs = 64

        super().__init__(
            PPO2Params(
                tau = 0.1,
                clip = 0.2,
                gamma = 0.99,
                policy_learning_rate = 2e-4,
                value_learning_rate = 2e-3,
                entropy_coefficient = 0.02,
                hidden_size = 512,
                gae_lambda = 0.95
            ),
            TrainerParams(
                batch_transitions_by_env_trajectory = True, # Must be enabled for PPO
                num_epochs = 2000,
                batches_per_epoch = 1,
                batch_size = 128,
                updates_per_batch = 1,
                shuffle_batches = False, # False to not interfere with GAE creation
                save_location = "./saved_models/AntPPO"
            ),
            EnvParams(
                env_name = "brax-hopper",
                env_normalization=False,
                num_envs = self.num_envs,
                max_episode_steps = self.max_episode_steps,
                vector_env=False, # Brax will init 'n' environments on its side
                misc_arguments = {
                    "batch_size": self.num_envs, # Brax's convention uses batch_size for num_environments
                    "episode_length": self.max_episode_steps,
                    "action_repeat": 3,
                    "exclude_current_positions_from_observation": False
                }
            )
        )

class PPO2InvertedDoublePendulumConfig(Config):
    def __init__(self):
        self.max_episode_steps = 512
        self.num_envs = 3
        super().__init__(
            PPO2Params(
                tau = 0.005,
                clip = 0.2,
                gamma = 0.99,
                policy_learning_rate = 4e-4,
                entropy_coefficient = 0.01,
                hidden_size = 256,
                gae_lambda = 0.95,
                log_std_max=2,
                log_std_min=-20,
                reward_ema_coefficient = 0.99,
                clipped_value_loss_eps = 0.2,
                policy_loss_weight = 1.0,
                value_loss_weight = 0.5,
                max_grad_norm = .5
            ),
            TrainerParams(
                batch_transitions_by_env_trajectory = True, # Must be enabled for PPO
                num_epochs = 2000,
                batches_per_epoch = 1,
                batch_size = 1024,
                updates_per_batch = 1,
                shuffle_batches = False, # False to not interfere with GAE creation
                save_model_frequency=20,
                save_location = "./saved_models/InvertedDoublePendulumPPO2"
            ),
            EnvParams(
                env_name = "InvertedDoublePendulum-v4",
                env_normalization=False,
                num_envs = self.num_envs,
                max_episode_steps = self.max_episode_steps,
                vector_env=True,
                misc_arguments = {
                    "max_episode_steps": self.max_episode_steps,
                    "render_mode": "rgb_array"
                }
            )
        )

class PPO2SwimmerConfig(Config):
    def __init__(self):
        self.max_episode_steps = 256
        self.num_envs = 1
        super().__init__(
            PPO2Params(
                clip = 0.2,
                gamma = 0.99,
                policy_learning_rate = 2e-4,
                entropy_coefficient = 0.01,
                hidden_size = 256,
                gae_lambda = 0.95,
                log_std_max=2,
                log_std_min=-20,
                reward_ema_coefficient = 0.99,
                clipped_value_loss_eps = 0.2,
                max_grad_norm = .5,
                use_moving_average_reward = False,
                combined_optimizer = True,
                policy_loss_weight = 1.0,
                value_loss_weight = .5
            ),
            TrainerParams(
                batch_transitions_by_env_trajectory = True, # Must be enabled for PPO
                num_epochs = 2000,
                batches_per_epoch = 1,
                batch_size = 1024,
                updates_per_batch = 1,
                shuffle_batches = False, # False to not interfere with GAE creation
                save_model_frequency=20,
                save_location = "./saved_models/SwimmerPPO2",
                preprocess_action = lambda x: x.view((self.num_envs,2)).to(dtype=torch.float32).numpy()
            ),
            EnvParams(
                env_name = "Swimmer-v4",
                env_normalization=False,
                num_envs = self.num_envs,
                max_episode_steps = self.max_episode_steps,
                vector_env=True,
                misc_arguments = {
                    "max_episode_steps": self.max_episode_steps,
                    "render_mode": "rgb_array"
                }
            )
        )


class PPO2HalfCheetahConfig(Config):
    def __init__(self):
        self.max_episode_steps = 1024
        self.num_envs = 2
        super().__init__(
            PPO2Params(
                clip = 0.2,
                gamma = 0.99,
                policy_learning_rate = 2e-4,
                value_learning_rate = 1e-3,
                entropy_coefficient = 0.1,
                hidden_size = 256,
                gae_lambda = 0.95,
                log_std_max=2,
                log_std_min=-20,
                reward_ema_coefficient = 0.99,
                clipped_value_loss_eps = 0.2,
                max_grad_norm = 1.0,
                use_moving_average_reward = True,
                combined_optimizer = True
            ),
            TrainerParams(
                batch_transitions_by_env_trajectory = True, # Must be enabled for PPO
                num_epochs = 2000,
                batches_per_epoch = 1,
                batch_size = 128,
                updates_per_batch = 1,
                shuffle_batches = False, # False to not interfere with GAE creation
                save_model_frequency=20,
                save_location = "./saved_models/HalfCheetahPPO2",
                preprocess_action = lambda x: x.view((self.num_envs,6)).to(dtype=torch.float32).numpy()
            ),
            EnvParams(
                env_name = "HalfCheetah-v4",
                env_normalization=False,
                num_envs = self.num_envs,
                max_episode_steps = self.max_episode_steps,
                vector_env=True,
                misc_arguments = {
                    "max_episode_steps": self.max_episode_steps,
                    "render_mode": "rgb_array"
                }
            )
        )

class PPO2ReacherConfig(Config):
    def __init__(self):
        self.max_episode_steps = 1024
        self.num_envs = 2
        super().__init__(
            PPO2Params(
                clip = 0.2,
                gamma = 0.99,
                policy_learning_rate = 2e-4,
                value_learning_rate = 1e-3,
                entropy_coefficient = 0.01,
                hidden_size = 256,
                gae_lambda = 0.95,
                reward_ema_coefficient = 0.99,
                clipped_value_loss_eps = 0.2,
                max_grad_norm = .5,
                use_moving_average_reward = True,
                combined_optimizer = True
            ),
            TrainerParams(
                batch_transitions_by_env_trajectory = True, # Must be enabled for PPO
                num_epochs = 2000,
                batches_per_epoch = 1,
                batch_size = 256,
                updates_per_batch = 1,
                shuffle_batches = False, # False to not interfere with GAE creation
                save_model_frequency=20,
                save_location = "./saved_models/ReacherPPO2",
                preprocess_action = lambda x: x.numpy()
            ),
            EnvParams(
                env_name = "Reacher-v4",
                env_normalization=False,
                num_envs = self.num_envs,
                max_episode_steps = self.max_episode_steps,
                vector_env=True,
                misc_arguments = {
                    "max_episode_steps": self.max_episode_steps,
                    "render_mode": "rgb_array"
                }
            )
        )

class PPO2BraxHalfCheetahConfig(Config):
    def __init__(self):
        self.max_episode_steps = 256
        self.num_envs = 128

        super().__init__(
            PPO2Params(
                clip = 0.2,
                gamma = 0.97,
                policy_learning_rate = 2e-4,
                value_learning_rate = 1e-3,
                entropy_coefficient = 0.1,
                hidden_size = 256,
                gae_lambda = 0.95,
                clipped_value_loss_eps = 0.2,
                value_loss_weight = 0.5,
                max_grad_norm = 2.0,
                use_moving_average_reward = True,
                combined_optimizer = False
            ),
            TrainerParams(
                batch_transitions_by_env_trajectory = True, # Must be enabled for PPO
                num_epochs = 2000,
                batches_per_epoch = 1,
                batch_size = 128,
                updates_per_batch = 1,
                shuffle_batches = False, # False to not interfere with GAE creation
                save_location = "./saved_models/HalfCheetahPPO"
            ),
            EnvParams(
                env_name = "brax-half-cheetah",
                env_normalization=False,
                num_envs = self.num_envs,
                max_episode_steps = self.max_episode_steps,
                vector_env=False, # Brax will init 'n' environments on its side
                misc_arguments = {
                    "batch_size": self.num_envs, # Brax's convention uses batch_size for num_environments
                    "episode_length": self.max_episode_steps,
                    "action_repeat": 1,
                    "forward_reward_weight": 1.,
                    "ctrl_cost_weight": 0.1,
                    "legacy_spring" : True,
                    "exclude_current_positions_from_observation": False,
                    "reset_noise_scale": 0.1,
                }
            )
        )
