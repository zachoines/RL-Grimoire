from Configurations import *

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
        self.max_episode_steps = 512
        self.num_envs = 256
        super().__init__(
            PPOParams(
                tau = 0.05,
                clip = 0.2,
                gamma = 0.98,
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
                    "healthy_reward": 0.5,
                    "ctrl_cost_weight": 1.0,
                    "contact_cost_weight": 1e-3,
                    "use_contact_forces": True,
                    "terminate_when_unhealthy": True
                }
            )
        )