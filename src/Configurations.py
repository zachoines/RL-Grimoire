import numpy as np
from typing import Dict

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
            num_envs: int = 1,
            episode_length: int = -1, # Number of steps before reset is called
            num_epochs: int = 1, 
            batches_per_epoch: int = 1, # How many batches to collect each epoch
            batch_size: int = 512,  # Size of training batch
            updates_per_batch: int = 1, # Number of times to train on a batch
            shuffle_batches: bool = False, # Shuffle batches AFTER sampling
            render: bool = False, # Render the environment (may not be possible in vector environments)
            env_name: str = "", # Name of the environment used in training (fused as save file name)
            save_location: str = "", # Location of save file
            env_normalization_weights: np.ndarray = np.array([]), # weights multiplied to state (used to normalize)
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

        # Training Params
        self.num_envs = num_envs
        self.episode_length = episode_length
        self.num_epochs = num_epochs
        self.batches_per_epoch = batches_per_epoch
        self.batch_size = batch_size
        self.updates_per_batch = updates_per_batch
        self.shuffle_batches = shuffle_batches

        # Env params
        self.render = render
        self.env_name = env_name
        self.save_location = save_location
        self.env_normalization_weights = env_normalization_weights
        self.squeeze_actions = squeeze_actions

        # Misc
        self.record_video_frequency = record_video_frequency

class REINFORCEParams(AgentParams):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class A2CParams(AgentParams):
    def __init__(self, tau: float = 0.01, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tau = tau

class PPOParams(A2CParams):
    def __init__(self, clip: float = 0.3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clip = clip

class Config(object):
    def __init__(self, agent_params, trainer_params: TrainerParams):
        self.agent_params = agent_params
        self.trainer_params = trainer_params     




