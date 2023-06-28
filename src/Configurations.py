import numpy as np
from typing import Dict, Any


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
            hidden_size: int = 64,
            max_grad_norm: float = 0.5
        ):
        self.policy_learning_rate = policy_learning_rate
        self.value_learning_rate = value_learning_rate
        self.gamma = gamma
        self.entropy_coefficient = entropy_coefficient
        self.hidden_size = hidden_size
        self.max_grad_norm = max_grad_norm

class TrainerParams(object):
    def __init__(self,
            replay_buffer_max_size: int = 1000000,
            replay_buffer_min_size: int = -1,
            replay_buffer_remove_on_sample: bool = True, # Remove experiances after sampling
            replay_buffer_shuffle_experiances: bool = False, # Shuffle experiances BEFORE sampling
            batch_transitions_by_env_trajectory: bool = False, # Some on-policy algorithms need to preserve this, rather than just random sampling. Needed when calculating returns for per env.
            num_epochs: int = 1, 
            batches_per_epoch: int = 1, # How many batches to collect each epoch
            batch_size: int = 512,  # Size of training batch
            updates_per_batch: int = 1, # Number of times to train on a batch
            shuffle_batches: bool = False, # Shuffle batches AFTER sampling
            render: bool = False, # Render the environment (may not be possible in vector environments)
            save_location: str = "", # Location of save file
            preprocess_action = lambda x: x,
            save_model_frequency: int = 1 # Save model after so meny epochs
        ):

        # Replay buff settings
        self.replay_buffer_max_size = replay_buffer_max_size
        self.replay_buffer_min_size = replay_buffer_min_size
        self.replay_buffer_remove_on_sample = replay_buffer_remove_on_sample
        self.replay_buffer_shuffle_experiances = replay_buffer_shuffle_experiances
        self.batch_transitions_by_env_trajectory = batch_transitions_by_env_trajectory

        # Training Params
        self.num_epochs = num_epochs
        self.batches_per_epoch = batches_per_epoch
        self.batch_size = batch_size
        self.updates_per_batch = updates_per_batch
        self.shuffle_batches = shuffle_batches

        # Misc
        self.render = render
        self.save_location = save_location
        self.preprocess_action = preprocess_action
        self.save_model_frequency = save_model_frequency

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
    def __init__(self, 
                 gae_lambda: float = 0.95, 
                 log_std_min: float = -20.0, 
                 log_std_max: float = 2.0, 
                 policy_loss_weight: float = 1.0, 
                 value_loss_weight: float = .5, 
                 use_moving_average_reward: bool = True,
                 combined_optimizer: bool = False,
                 value_loss_clipping: bool = True,
                 clipped_value_loss_eps: float = 0.2,
                 mini_batch_size: int = 64,
                 num_rounds: int = 4,
                 use_lr_scheduler: bool = False,
                 lr_scheduler_constant_steps: int = 5000,
                 lr_scheduler_max_steps: int = 150000,
                 lr_scheduler_max_factor: float = 1.0,
                 lr_scheduler_min_factor: float = 1.0 / 10.0,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        # misc
        self.gae_lambda = gae_lambda
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.use_moving_average_reward = use_moving_average_reward
        
        # Optimizer parameters
        self.policy_loss_weight = policy_loss_weight
        self.value_loss_weight = value_loss_weight
        self.combined_optimizer = combined_optimizer

        # Loss parmeters
        self.clipped_value_loss_eps = clipped_value_loss_eps
        self.value_loss_clipping = value_loss_clipping

        # PPO specific training params
        self.mini_batch_size = mini_batch_size
        self.num_rounds = num_rounds

        # Learning rate scheduler
        self.use_lr_scheduler = use_lr_scheduler
        self.lr_scheduler_constant_steps = lr_scheduler_constant_steps
        self.lr_scheduler_max_steps = lr_scheduler_max_steps
        self.lr_scheduler_max_factor = lr_scheduler_max_factor
        self.lr_scheduler_min_factor = lr_scheduler_min_factor
        
        # Class name
        self.agent_name = "PPO2"

class PPO2RecurrentParams(PPO2Params):
    def __init__(self, 
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_name = "PPO2Recurrent"

class Config(object):
    def __init__(self, agent_params, trainer_params: TrainerParams=TrainerParams(), env_params: EnvParams=EnvParams()):
        self.agent_params = agent_params
        self.trainer_params = trainer_params
        self.env_params = env_params




