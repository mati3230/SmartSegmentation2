from abc import ABC, abstractmethod


class BaseEnvironment(ABC):
    """ TODO Use OpenAI Gym Environment """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def render(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def close(self):
        pass
