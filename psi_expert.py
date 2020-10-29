import numpy as np


class PsiExpert():
    def __init__(self):
        pass

    def compute_key(self, state):
        P = state[0]
        P_idxs = state[1]
        n_idxs = state[2]
        P_mean = np.mean(P[P_idxs], axis=0)
        n_mean = np.mean(P[n_idxs], axis=0)
        return\
            str(P_mean[0]) + "," + str(P_mean[1]) + "," + str(P_mean[2]) +\
            "_" + str(n_mean[0]) + "," + str(n_mean[1]) + "," + str(n_mean[2])

    def action_m(self, env, state):
        action = 0
        psi = env._psi(action)
        if psi == -1:
            action = 1
        return action

    def __call__(self, env, state):
        return self.action_m(env, state)
