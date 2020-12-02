from environment.base_environment import BaseEnvironment
import gym

class GymEnv(BaseEnvironment):
    def __init__(self, env_str="CartPole-v0", **kwargs):
        self.env = gym.make(env_str)
        self.reset()

    def step(self, action):
        # action = action.numpy()
        action = int(action[0])
        return self.env.step(action)

    def render(self):
        self.env.render()

    def reset(self, train=True):
        return self.env.reset()

    def close(self):
        self.env.close()
