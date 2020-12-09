from abc import ABC, abstractmethod


class BaseEnvironment(ABC):
    """Abstract class with methods that should be brought with an environment.
    We did not used the gym framework to deny an unnecessary dependency.
    Currently, we use this framework only for point cloud processing and only
    need a few abstract methods."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def step(self, action):
        """Apply an action in the environment.

        Parameters
        ----------
        action : int
            A valid action of the environment.

        Returns
        -------
        tuple(np.ndarray, float, boolean, dict)
            Should return the next observation, a reward value, flag if episode
            is finished and addtitional information.

        """
        pass

    @abstractmethod
    def render(self):
        """Rendering of the environment."""
        pass

    @abstractmethod
    def reset(self, train=True):
        """Resets the environment.

        Parameters
        ----------
        train : boolean
            Will the agent train with the environment?

        Returns
        -------
        np.ndarray
            The initial observation.

        """
        pass

    @abstractmethod
    def close(self):
        pass
