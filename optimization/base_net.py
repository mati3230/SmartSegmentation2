from abc import ABC, abstractmethod


class BaseNet(ABC):
    def __init__(
            self,
            name,
            outpt,
            trainable=True,
            seed=None,
            check_numerics=False):
        super().__init__()
        self.name = name
        self.seed = seed
        self.trainable = trainable
        self.outpt = outpt
        self.check_numerics = check_numerics

    @abstractmethod
    def get_vars(self):
        pass

    @abstractmethod
    def compute(self, obs):
        pass
