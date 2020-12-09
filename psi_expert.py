import numpy as np


class PsiExpert():
    """This class implements the expert function that is mentioned in the
    publication."""
    def __init__(self):
        pass

    def action_m(self, env, state):
        """Calculates the expert action.

        Parameters
        ----------
        env : BaseEnv
            Reference to the superpoint environment.
        state : np.ndarray
            Current observation.

        Returns
        -------
        int
            The recommended action.

        """
        action = 0
        psi = env._psi(action)
        if psi == -1:
            action = 1
        return action

    def __call__(self, env, state):
        return self.action_m(env, state)
