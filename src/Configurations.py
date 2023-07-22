import numpy as np
from typing import Dict, Any

class ICMParams(object):
    """
    A class to hold the parameters for the Intrinsic Curiosity Module (ICM).

    Attributes:
        enabled (bool): Whether or not to enable the ICM module.
        alpha (float): The scaling factor weighting policy loss against ICM losses.
        beta (float): The weighting factor for the forward loss in the ICM module.
        n (float): The scaling factor for intrinsic reward in the ICM module. i.e. scale = (n / 2). 
        hidden_size (int): The size of the hidden layers in the ICM module.
        state_feature_size (int): The size of the state feature representation in the ICM module.
        learning_rate (float): Learning rate passed to the optimizer
    """

    def __init__(self,
            enabled: bool = False,
            alpha: float = 0.1,
            beta: float = 0.2, 
            n: float = 1,
            hidden_size: int = 256,
            state_feature_size: int = 256,
            learning_rate: float = 1e-4
        ):
        self.enabled = enabled
        self.alpha = alpha
        self.beta = beta
        self.n = n
        self.hidden_size = hidden_size
        self.state_feature_size = state_feature_size
        self.learning_rate = learning_rate

class EnvParams(object):
    """
    A class to hold the parameters for the environment.

    Attributes:
        env_name (str): Name of the environment used in training. This is also used as the save file name.
        num_envs (int): The number of environments to be used.
        max_episode_steps (int): The maximum number of steps before reset is called on the environment.
        env_normalization (bool): Whether to apply normalization to the environment. If True, weights are multiplied to the state.
        vector_env (bool): Whether the environment conforms to the gym.vector specification. Some environments, like Brax, do not.
        misc_arguments (Dict[str, object]): Environment-specific named parameters passed when calling "gym.make".
    """

    def __init__(self,
            env_name: str = "",
            num_envs: int = 1,
            max_episode_steps: int = -1,
            env_normalization: bool = False,
            vector_env: bool = True,
            misc_arguments: Dict[str, object] = {}
        ):
        self.env_normalization = env_normalization
        self.env_name = env_name
        self.num_envs = num_envs
        self.max_episode_steps = max_episode_steps
        self.vector_env = vector_env
        self.misc_arguments = misc_arguments
    
class AgentParams(object):
    """
    A class to hold the parameters for the agent.

    Attributes:
        policy_learning_rate (float): The learning rate for the policy network.
        value_learning_rate (float): The learning rate for the value network.
        gamma (float): The discount factor for future rewards.
        entropy_coefficient (float): The coefficient for the entropy bonus, encouraging exploration.
        hidden_size (int): The size of the hidden layers in the agent's networks.
        max_grad_norm (float): The maximum allowed norm for the gradient (for gradient clipping).
        icm_module (ICMParams): The parameters for the Intrinsic Curiosity Module (ICM).
    """

    def __init__(self,
            policy_learning_rate: float = 1e-4,
            value_learning_rate: float = 1e-3,
            gamma: float = 0.99, 
            entropy_coefficient: float = 1e-2,
            hidden_size: int = 64,
            max_grad_norm: float = 0.5,
            icm_module: ICMParams = ICMParams()
        ):
        self.policy_learning_rate = policy_learning_rate
        self.value_learning_rate = value_learning_rate
        self.gamma = gamma
        self.entropy_coefficient = entropy_coefficient
        self.hidden_size = hidden_size
        self.max_grad_norm = max_grad_norm
        self.icm_module = icm_module

class TrainerParams(object):
    """
    A class to hold the parameters for the trainer.

    Attributes:
        replay_buffer_max_size (int): The maximum size of the replay buffer.
        replay_buffer_min_size (int): The minimum size of the replay buffer before training starts.
        replay_buffer_remove_on_sample (bool): Whether to remove experiences from the replay buffer after they are sampled.
        replay_buffer_shuffle_experiences (bool): Whether to shuffle the experiences in the replay buffer before sampling.
        batch_transitions_by_env_trajectory (bool): Whether to preserve the order of transitions by environment trajectory when sampling from the replay buffer. Some on-policy algorithms need this.
        num_epochs (int): The number of training epochs.
        batches_per_epoch (int): The number of batches to collect each epoch.
        batch_size (int): The size of each training batch.
        updates_per_batch (int): The number of times to train on each batch.
        shuffle_batches (bool): Whether to shuffle the batches after sampling.
        render (bool): Whether to render the environment during training. Note that this may not be possible in vector environments.
        save_location (str): The location to save the trained model.
        preprocess_action (function): A function to preprocess the actions before they are sent to the environment.
        save_model_frequency (int): The frequency at which the model should be saved, in terms of epochs.
    """

    def __init__(self,
            replay_buffer_max_size: int = 1000000,
            replay_buffer_min_size: int = -1,
            replay_buffer_remove_on_sample: bool = True,
            replay_buffer_shuffle_experiences: bool = False,
            batch_transitions_by_env_trajectory: bool = False,
            num_epochs: int = 1, 
            batches_per_epoch: int = 1,
            batch_size: int = 512,
            updates_per_batch: int = 1,
            shuffle_batches: bool = False,
            render: bool = False,
            save_location: str = "",
            preprocess_action = lambda x: x,
            save_model_frequency: int = 1
        ):

        self.replay_buffer_max_size = replay_buffer_max_size
        self.replay_buffer_min_size = replay_buffer_min_size
        self.replay_buffer_remove_on_sample = replay_buffer_remove_on_sample
        self.replay_buffer_shuffle_experiences = replay_buffer_shuffle_experiences
        self.batch_transitions_by_env_trajectory = batch_transitions_by_env_trajectory

        self.num_epochs = num_epochs
        self.batches_per_epoch = batches_per_epoch
        self.batch_size = batch_size
        self.updates_per_batch = updates_per_batch
        self.shuffle_batches = shuffle_batches

        self.render = render
        self.save_location = save_location
        self.preprocess_action = preprocess_action
        self.save_model_frequency = save_model_frequency

class REINFORCEParams(AgentParams):
    """
    A class to hold the configuration parameters for the REINFORCE agent.

    Attributes:

    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the REINFORCEParams object with parameters inherited from AgentParams.

        :param args: Variable length argument list inherited from AgentParams.
        :param kwargs: Arbitrary keyword arguments inherited from AgentParams.
        """
        super().__init__(*args, **kwargs)
        self.agent_name = "REINFORCE"

class A2CParams(AgentParams):
    """
    A class to hold the configuration parameters for the A2C agent.

    Attributes:
        tau (float): The factor for soft update of the target networks. 
        agent_name (str): The name of the agent.
    """

    def __init__(self, tau: float = 0.01, *args, **kwargs):
        """
        Initialize the A2CParams object with tau and other parameters inherited from AgentParams.

        :param tau: The factor for soft update of the target networks. Defaults to 0.01.
        :param args: Variable length argument list inherited from AgentParams.
        :param kwargs: Arbitrary keyword arguments inherited from AgentParams.
        """
        super().__init__(*args, **kwargs)
        self.tau = tau
        self.agent_name = "A2C"

class PPOParams(A2CParams):
    """
    A class to hold the configuration parameters for the PPO agent.

    Attributes:
        clip (float): The clipping parameter for the policy objective function.
    """

    def __init__(self, clip: float = 0.3, *args, **kwargs):
        """
        Initialize the PPOParams object with clip and other parameters inherited from A2CParams.

        :param clip: The clipping parameter for the policy objective function. Defaults to 0.3.
        :param args: Variable length argument list inherited from A2CParams.
        :param kwargs: Arbitrary keyword arguments inherited from A2CParams.
        """
        super().__init__(*args, **kwargs)
        self.clip = clip
        self.agent_name = "PPO"

class PPO2Params(PPOParams):
    """
    A class to hold the parameters for the PPO2 agent.

    Attributes:
        gae_lambda (float): The lambda parameter for Generalized Advantage Estimation (GAE).
        log_std_min (float): The minimum value for the log standard deviation of the policy's Gaussian distribution.
        log_std_max (float): The maximum value for the log standard deviation of the policy's Gaussian distribution.
        policy_loss_weight (float): The weight for the policy loss in the total loss function.
        value_loss_weight (float): The weight for the value loss in the total loss function.
        use_moving_average_reward (bool): Whether to use a moving average of the reward for normalization.
        combined_optimizer (bool): Whether to use a combined optimizer for both the policy and value function.
        value_loss_clipping (bool): Whether to clip the value loss to prevent large updates.
        clipped_value_loss_eps (float): The epsilon parameter for value loss clipping.
        mini_batch_size (int): The size of the mini-batches for training.
        num_rounds (int): The number of rounds of mini-batch updates.
        use_lr_scheduler (bool): Whether to use a learning rate scheduler.
        lr_scheduler_constant_steps (int): The number of steps before the learning rate begins to decay.
        lr_scheduler_max_steps (int): The maximum number of steps for the learning rate scheduler.
        lr_scheduler_max_factor (float): The maximum factor for scaling the learning rate.
        lr_scheduler_min_factor (float): The minimum factor for scaling the learning rate.
    """

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

        self.gae_lambda = gae_lambda
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.use_moving_average_reward = use_moving_average_reward
        
        self.policy_loss_weight = policy_loss_weight
        self.value_loss_weight = value_loss_weight
        self.combined_optimizer = combined_optimizer

        self.clipped_value_loss_eps = clipped_value_loss_eps
        self.value_loss_clipping = value_loss_clipping

        self.mini_batch_size = mini_batch_size
        self.num_rounds = num_rounds

        self.use_lr_scheduler = use_lr_scheduler
        self.lr_scheduler_constant_steps = lr_scheduler_constant_steps
        self.lr_scheduler_max_steps = lr_scheduler_max_steps
        self.lr_scheduler_max_factor = lr_scheduler_max_factor
        self.lr_scheduler_min_factor = lr_scheduler_min_factor
        
        self.agent_name = "PPO2"

class PPO2RecurrentParams(PPO2Params):
    """
    A class to hold the configuration parameters for the PPO2Recurrent agent.

    Attributes:

    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the PPO2RecurrentParams object with parameters inherited from PPO2Params.

        :param args: Variable length argument list inherited from PPO2Params.
        :param kwargs: Arbitrary keyword arguments inherited from PPO2Params.
        """
        super().__init__(*args, **kwargs)
        self.agent_name = "PPO2Recurrent"

class Config(object):
    """
    A class to hold the configuration parameters for the agent, trainer, and environment.

    Attributes:
        agent_params (AgentParams): The parameters for the agent.
        trainer_params (TrainerParams): The parameters for the trainer.
        env_params (EnvParams): The parameters for the environment.
    """

    def __init__(self, agent_params, trainer_params: TrainerParams=TrainerParams(), env_params: EnvParams=EnvParams()):
        """
        Initialize the Config object with agent, trainer, and environment parameters.

        :param agent_params: The parameters for the agent.
        :param trainer_params: The parameters for the trainer. Defaults to TrainerParams().
        :param env_params: The parameters for the environment. Defaults to EnvParams().
        """
        self.agent_params = agent_params
        self.trainer_params = trainer_params
        self.env_params = env_params