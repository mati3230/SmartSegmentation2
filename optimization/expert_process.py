import numpy as np
from .agent_process import AgentProcess

from deprecated import deprecated

class ExpertProcess(AgentProcess):
    @deprecated
    def __init__(
            self,
            conn,
            id,
            n_cpus,
            n_steps,
            agent_type,
            agent_args,
            env_type,
            env_args,
            model_dir,
            model_filename,
            seed=42,
            add_args=None,
            async_mode=False):
        super().__init__(
            conn=conn,
            id=id,
            n_cpus=n_cpus,
            n_steps=n_steps,
            agent_type=agent_type,
            agent_args=agent_args,
            env_type=env_type,
            env_args=env_args,
            model_dir=model_dir,
            model_filename=model_filename,
            add_args=add_args,
            async_mode=async_mode)
        expert_type = add_args["expert_type"]
        expert_args = add_args["expert_args"]
        if expert_args:
            self.expert = expert_type(expert_args)
        else:
            self.expert = expert_type()

    def prepare_agent_args(self):
        super().prepare_agent_args()
        self.agent_args["mode"] = "pre"

    def load_method(self, msg, agent):
        return

    def select_action(self, agent, obs):
        p_obs = agent.preprocess(obs)
        # pi_action = agent.action(p_obs)
        expert_action = self.expert(self.env, obs)
        pi_action = {}
        pi_action["expert"] = expert_action
        action = int(np.random.randint(low=0, high=2))
        return p_obs, pi_action, action
