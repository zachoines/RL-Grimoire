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
                num_envs = 64,
                episode_length = 1000,
                num_epochs = 1000,
                batches_per_epoch = 8,
                batch_size = 32,
                shuffle_batches = True,
                render = False,
                env_name = "CartPole-v1",
                save_location = "RL-Grimoire/saved_models/CartPoleREINFORCE",
                squeeze_actions = True,
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
                num_envs = 64,
                episode_length = 128,
                num_epochs = 100,
                batches_per_epoch = 512,
                batch_size = 64,
                shuffle_batches = True,
                render = False,
                env_name = "HalfCheetah-v4",
                save_location = "RL-Grimoire/saved_models/HalfCheetahREINFORCE"
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
                learningRateScheduler = True,
                learningRateSchedulerClass="StepLR",
                learningRateScheduleArgs={ "step_size" : 128, "gamma": 0.99},
                num_envs = 32,
                episode_length = 512,
                num_epochs = 100,
                batches_per_epoch = 128,
                batch_size = 128,
                env_name = "InvertedDoublePendulum-v4",
                save_location = "RL-Grimoire/saved_models/InvertedDoublePendulumAC2"
            )
        )

class PPOAntConfig(Config):
    def __init__(self):
        super().__init__(
            PPOParams(
                tau = 0.1,
                clip=0.3,
                gamma = 0.97,
                policy_learning_rate = 1e-4,
                value_learning_rate = 1e-3,
                entropy_coefficient = 0.1,
                hidden_size = 256,
            ),
            TrainerParams(
                num_envs = 32,
                episode_length = 1000,
                num_epochs = 100,
                batches_per_epoch = 8,
                batch_size = 1024,
                updates_per_batch = 8,
                shuffle_batches = True,
                env_name = "Ant-v4",
                save_location = "RL-Grimoire/saved_models/AntPPO",
            )
        )