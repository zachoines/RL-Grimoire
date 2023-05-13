class Hyperparams:
    def __init__(self,
        num_envs: int = 8,
        num_epochs: int = 600,
        samples_per_epoch: int = 1024, 
        batch_size: int = 1024,  
        policy_learning_rate: float = 3e-4, 
        gamma: float = 0.99, 
        entropy_coefficient:float = 1e-4):
        
        self.num_envs = num_envs
        self.num_epochs = num_epochs
        self.samples_per_epoch = samples_per_epoch
        self.batch_size = batch_size
        self.policy_learning_rate = policy_learning_rate
        self.gamma = gamma
        self.entropy_coefficient = entropy_coefficient