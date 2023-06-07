import numpy as np
from typing import Dict, Any
import torch.optim.lr_scheduler as lr_scheduler

class EnvParams(object):
    def __init__(self,
            env_name: str = "", # Name of the environment used in training (fused as save file name)
            num_envs: int = 1,
            max_episode_steps: int = -1, # Number of steps before reset is called
            env_normalization: bool = False, # weights multiplied to state (used to normalize)
            vector_env: bool = True, # Some environments dont conform to gym.vector spec (like brax)
            misc_arguments: Dict[str, object] = {} # Env specific named parameters passed calling "gym.make"
        ):
        self.env_normalization = env_normalization
        self.env_name = env_name
        self.num_envs = num_envs
        self.max_episode_steps = max_episode_steps
        self.vector_env = vector_env
        self.misc_arguments = misc_arguments
    

class AgentParams(object):
    def __init__(self,
            policy_learning_rate: float = 1e-4,
            value_learning_rate: float = 1e-3,
            gamma: float = 0.99, 
            entropy_coefficient: float = 1e-2,
            hidden_size: int = 64
        ):
        self.policy_learning_rate = policy_learning_rate
        self.value_learning_rate = value_learning_rate
        self.gamma = gamma
        self.entropy_coefficient = entropy_coefficient
        self.hidden_size = hidden_size

class TrainerParams(object):
    def __init__(self,
            learningRateScheduler: bool = False,
            learningRateSchedulerClass: str = "ExponentialLR", # Valid names: ['LambdaLR', 'MultiplicativeLR', 'StepLR', 'MultiStepLR', 'ConstantLR', 'LinearLR', 'ExponentialLR', 'SequentialLR', 'CosineAnnealingLR', 'ChainedScheduler', 'ReduceLROnPlateau', 'CyclicLR', 'CosineAnnealingWarmRestarts', 'OneCycleLR', 'PolynomialLR', 'LRScheduler']
            learningRateScheduleArgs: Dict[str, object] = { "gamma": 0.99 },
            replay_buffer_max_size: int = 1000000,
            replay_buffer_min_size: int = -1,
            replay_buffer_remove_on_sample: bool = True, # Remove experiances after sampling
            replay_buffer_shuffle_experiances: bool = False, # Shuffle experiances BEFORE sampling
            batch_transitions_by_env_trajectory: bool = False, # Some on-policy algorithms need to preserve this, rather than just random sampling. Needed when calculating returns for per env.
            on_policy_training: bool = True, # In short: Trains only on recently collected experiances, replay buffer doesn't store experiences.
            num_epochs: int = 1, 
            batches_per_epoch: int = 1, # How many batches to collect each epoch
            batch_size: int = 512,  # Size of training batch
            updates_per_batch: int = 1, # Number of times to train on a batch
            shuffle_batches: bool = False, # Shuffle batches AFTER sampling
            render: bool = False, # Render the environment (may not be possible in vector environments)
            save_location: str = "", # Location of save file
            squeeze_actions: bool = False, # Remove empty dimension for actions. For example, needed when using only one env copy.
            record_video_frequency: int = 1 # Record video after so meny epochs
        ):

        # Schedulers
        self.learningRateScheduler = learningRateScheduler
        self.learningRateSchedulerClass = learningRateSchedulerClass
        self.learningRateScheduleArgs = learningRateScheduleArgs

        # Replay buff settings
        self.replay_buffer_max_size = replay_buffer_max_size
        self.replay_buffer_min_size = replay_buffer_min_size
        self.replay_buffer_remove_on_sample = replay_buffer_remove_on_sample
        self.replay_buffer_shuffle_experiances = replay_buffer_shuffle_experiances
        self.batch_transitions_by_env_trajectory = batch_transitions_by_env_trajectory
        self.on_policy_training = on_policy_training 

        # Training Params
        self.num_epochs = num_epochs
        self.batches_per_epoch = batches_per_epoch
        self.batch_size = batch_size
        self.updates_per_batch = updates_per_batch
        self.shuffle_batches = shuffle_batches

        # Misc
        self.render = render
        self.save_location = save_location
        self.squeeze_actions = squeeze_actions
        self.record_video_frequency = record_video_frequency

class REINFORCEParams(AgentParams):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_name = "REINFORCE"

class A2CParams(AgentParams):
    def __init__(self, tau: float = 0.01, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tau = tau
        self.agent_name = "A2C"

class PPOParams(A2CParams):
    def __init__(self, clip: float = 0.3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clip = clip
        self.agent_name = "PPO"


class PPO2Params(PPOParams):
    def __init__(self, gae_lambda: float = 0.9, min_std=0.1, max_std=2.0, policy_loss_weight = 1.0, value_loss_weight = 0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gae_lambda = gae_lambda
        self.min_std=min_std
        self.max_std = max_std
        self.policy_loss_weight = policy_loss_weight
        self.value_loss_weight = value_loss_weight
        self.agent_name = "PPO2"

class Config(object):
    def __init__(self, agent_params, trainer_params: TrainerParams=TrainerParams(), env_params: EnvParams=EnvParams()):
        self.agent_params = agent_params
        self.trainer_params = trainer_params
        self.env_params = env_params




