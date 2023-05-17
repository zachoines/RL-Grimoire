class Hyperparams:
    def __init__(self,
        num_envs: int = 8,
        num_epochs: int = 4000,
        samples_per_epoch: int = 512, 
        batch_size: int = 512,  
        policy_learning_rate: float = 1e-3, 
        gamma: float = 0.99, 
        entropy_coefficient:float = 1e-4,
        hidden_size: int = 64,
        render: bool = False,
        env_name: str = "CartPole-v1",
        save_location: str = "RL-Grimoire/saved_models/model"):
        
        # Env params
        self.render = render
        self.env_name = env_name
        self.save_location = save_location

        # Training Params
        self.num_envs = num_envs
        self.num_epochs = num_epochs
        self.samples_per_epoch = samples_per_epoch
        self.batch_size = batch_size

        # Network params
        self.policy_learning_rate = policy_learning_rate
        self.gamma = gamma
        self.entropy_coefficient = entropy_coefficient
        self.hidden_size = hidden_size


class InvertedPendulumHyperparams(Hyperparams):
    def __init__(self):
        super().__init__()
        self.num_envs = 8
        self.hidden_size = 32
        self.policy_learning_rate = 1e-4
        self.entropy_coefficient = 1e-5
        self.gamma = 0.999

        self.num_epochs = 1000
        self.samples_per_epoch = 2048
        self.batch_size = 512

        self.env_name = "InvertedPendulum-v4"
        self.save_location = "RL-Grimoire/saved_models/REINFORCE_InvertedPendulum"


class AntHyperparams(Hyperparams):
    def __init__(self):
        super().__init__()
        self.num_envs = 20
        self.hidden_size = 64
        self.policy_learning_rate = 1e-4
        self.entropy_coefficient = 1e-5
        self.gamma = 0.99

        self.num_epochs = 1000
        self.samples_per_epoch = 2048
        self.batch_size = 512

        self.env_name = "Ant-v4"
        self.save_location = "RL-Grimoire/saved_models/REINFORCE_Ant"


class HalfCheetahHyperparams(Hyperparams):
    def __init__(self):
        super().__init__()
        self.num_envs = 8
        self.hidden_size = 64
        self.policy_learning_rate = 1e-4
        self.entropy_coefficient = 1e-5
        self.gamma = 0.99

        self.num_epochs = 1000
        self.samples_per_epoch = 1024
        self.batch_size = 512

        self.env_name = "HalfCheetah-v4"
        self.save_location = "RL-Grimoire/saved_models/REINFORCE_HalfCheetah"

       
class PusherHyperparams(Hyperparams):
    def __init__(self):
        super().__init__()
        self.num_envs = 16
        self.hidden_size = 256
        self.policy_learning_rate = 1e-4
        self.entropy_coefficient = 1e-5
        self.gamma = 0.99

        self.num_epochs = 10000
        self.samples_per_epoch = 512
        self.batch_size = 512

        self.env_name = "Pusher-v4"
        self.save_location = "RL-Grimoire/saved_models/REINFORCE_Pusher"
        self.render = False


class MountainCarHyperparams(Hyperparams):
    def __init__(self):
        super().__init__()
        self.num_envs = 12
        self.hidden_size = 64
        self.policy_learning_rate = 1e-3
        self.entropy_coefficient = 1e-4
        self.gamma = 0.9

        self.num_epochs = 2000
        self.samples_per_epoch = 64
        self.batch_size = 64

        self.env_name = "MountainCar-v0"
        self.save_location = "RL-Grimoire/saved_models/REINFORCE_MountainCar"
        self.render = False

       