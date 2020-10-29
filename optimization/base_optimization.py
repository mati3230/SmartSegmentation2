from abc import ABC, abstractmethod


class BaseOptimization(ABC):
    def __init__(
            self,
            policy_type,
            n_actions,
            optimizer,
            train_summary_writer,
            batch_size=64,
            global_norm=0.5,
            check_numerics=False,
            write_tensorboard=True,
            **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.train_summary_writer = train_summary_writer
        self.global_norm = global_norm
        self.check_numerics = check_numerics
        self.n_actions = n_actions
        self.write_tensorboard = write_tensorboard

    @abstractmethod
    def get_online_policy(self):
        pass

    @abstractmethod
    def create_policies(
            self,
            policy_type,
            n_actions,
            check_numerics,
            **kwargs):
        pass

    @abstractmethod
    def update(self, buf, step):
        pass

    @abstractmethod
    def step_update(self, buf, step):
        pass
