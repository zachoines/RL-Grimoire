import numpy as np

class Config:
    def __init__(self):
        self.agent_params: AgentParams = AgentParams()
        self.trainer_params: TrainerParams = TrainerParams()


class REINFORCECartpoleConfig(Config):
    def __init__(self):
        super().__init__()

        # Agent related parameters
        self.agent_params.gamma = 0.99
        self.agent_params.policy_learning_rate = 1e-4
        self.agent_params.value_learning_rate = 1e-3
        self.agent_params.entropy_coefficient = 0.01
        self.agent_params.hidden_size = 64

        # Trainer related parameters
        self.trainer_params.num_envs = 8
        self.trainer_params.num_epochs = 1000
        self.trainer_params.steps_per_epoch = 512
        self.trainer_params.batch_size = 512
        self.trainer_params.render = False
        self.trainer_params.env_name = "CartPole-v1"
        self.trainer_params.save_location = "RL-Grimoire/saved_models/CartPoleREINFORCE"


class REINFORCEHalfCheetahConfig(Config):
    def __init__(self):
        super().__init__()

        # Agent related parameters
        self.agent_params.gamma = 0.99
        self.agent_params.policy_learning_rate = 1e-4
        self.agent_params.value_learning_rate = 1e-3
        self.agent_params.entropy_coefficient = 0.01
        self.agent_params.hidden_size = 64

        # Trainer related parameters
        self.trainer_params.num_envs = 8
        self.trainer_params.num_epochs = 1000
        self.trainer_params.steps_per_epoch = 512
        self.trainer_params.batch_size = 512
        self.trainer_params.env_name = "HalfCheetah-v4"
        self.trainer_params.save_location = "RL-Grimoire/saved_models/HalfCheetahREINFORCE"


class A2CInvertedDoublePendulumConfig(Config):
    def __init__(self):
        super().__init__()

        # Agent related parameters
        self.agent_params.tau = 0.005
        self.agent_params.gamma = 0.999
        self.agent_params.policy_learning_rate = 1e-5
        self.agent_params.value_learning_rate = 1e-4
        self.agent_params.entropy_coefficient = 0.005
        self.agent_params.hidden_size = 128

        # Trainer related parameters
        self.trainer_params.num_envs = 128       
        self.trainer_params.num_epochs = 1300
        self.trainer_params.replay_buffer_max_size = 1000000
        self.trainer_params.replay_buffer_min_size = 0
        self.trainer_params.steps_per_epoch = 512
        self.trainer_params.batch_size = 512
        self.trainer_params.update_rate = 4
        self.trainer_params.env_name = "InvertedDoublePendulum-v4"
        self.trainer_params.save_location = "RL-Grimoire/saved_models/InvertedDoublePendulumAC2"


class A2CPendulumConfig(Config):
    def __init__(self):
        super().__init__()

        # Agent related parameters
        self.agent_params.tau = 0.01
        self.agent_params.gamma = 0.99
        self.agent_params.policy_learning_rate = 1e-4
        self.agent_params.value_learning_rate = 1e-3
        self.agent_params.entropy_coefficient = 0.01
        self.agent_params.hidden_size = 64

        # Trainer related parameters
        self.trainer_params.num_envs = 64       
        self.trainer_params.num_epochs = 4000
        self.trainer_params.replay_buffer_max_size = 1000000
        self.trainer_params.replay_buffer_min_size = 0
        self.trainer_params.steps_per_epoch = 8
        self.trainer_params.batch_size = 64
        self.trainer_params.update_rate = 1
        self.trainer_params.env_name = "Pendulum-v1"
        self.trainer_params.save_location = "RL-Grimoire/saved_models/PendulumAC2"
        self.trainer_params.env_normalization_weights = np.array([1, 1,  0.125])


class AgentParams:
    def __init__(self,
            policy_learning_rate: float = 1e-4,
            value_learning_rate: float = 1e-3,
            tau: float = 0.01,
            gamma: float = 0.99, 
            entropy_coefficient: float = 1e-2,
            hidden_size: int = 64
        ):
        
        self.policy_learning_rate = policy_learning_rate
        self.value_learning_rate = value_learning_rate
        self.tau = tau
        self.gamma = gamma
        self.entropy_coefficient = entropy_coefficient
        self.hidden_size = hidden_size


class TrainerParams:
    def __init__(self,
            num_envs: int = 8,
            num_epochs: int = 1000,
            steps_per_epoch: int = 512, 
            replay_buffer_max_size: int = 1000000,
            replay_buffer_min_size: int = 10000,
            update_rate: int = 512,
            batch_size: int = 512,  
            render: bool = False,
            env_name: str = "",
            save_location: str = "",
            env_normalization_weights: np.ndarray = np.array([])
        ):

        # Replay buff settings
        self.replay_buffer_max_size = replay_buffer_max_size
        self.replay_buffer_min_size = replay_buffer_min_size

        # Training Params
        self.num_envs = num_envs
        self.num_epochs = num_epochs
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.update_rate = update_rate

        # Env params
        self.render = render
        self.env_name = env_name
        self.save_location = save_location
        self.env_normalization_weights = env_normalization_weights








