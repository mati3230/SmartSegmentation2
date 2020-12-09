import numpy as np
from .agent_process import AgentProcess

class ExpertProcess(AgentProcess):
    """This class stores the expert action in the process of the action
    selection of the agent. It is stored in the dictionary 'pi_action'
    under the key 'expert'.

    Parameters
    ----------
    conn : Multiprocessing.Pipe
        Connection to communicate with the master process.
    id : int
        ID of the agent process.
    n_cpus : int
        Number of agent processes.
    n_steps : int
        Number of steps/samples that should be calculated.
    agent_type : type
        Class type of the agent to generate a copy in the agent processes.
    agent_args : dict
        Input arguments of the agent class to generate a copy in the agent
        processes.
    env_type : type
        Class type of the environment to generate a copy in the agent
        processes.
    env_args : dict
        Input arguments of the environment class to generate a copy in the
        agent processes.
    model_dir : str
        Directory where the models will be stored.
    model_filename : str
        Filename of a model that will be stored.
    seed : int
        Random seed that should be used by the agent processes.
    add_args : dict
        Additional arguments for this class.
    async_mode : boolean
        If True, samples will collected while the agent is trained to speed
        up the procedure, i.e. a non-blocking training process.

    Attributes
    ----------
    expert : type
        Instance of an expert.

    """
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
        """Constructor.

        Parameters
        ----------
        conn : Multiprocessing.Pipe
            Connection to communicate with the master process.
        id : int
            ID of the agent process.
        n_cpus : int
            Number of agent processes.
        n_steps : int
            Number of steps/samples that should be calculated.
        agent_type : type
            Class type of the agent to generate a copy in the agent processes.
        agent_args : dict
            Input arguments of the agent class to generate a copy in the agent
            processes.
        env_type : type
            Class type of the environment to generate a copy in the agent
            processes.
        env_args : dict
            Input arguments of the environment class to generate a copy in the
            agent processes.
        model_dir : str
            Directory where the models will be stored.
        model_filename : str
            Filename of a model that will be stored.
        seed : int
            Random seed that should be used by the agent processes.
        add_args : dict
            Additional arguments for this class.
        async_mode : boolean
            If True, samples will collected while the agent is trained to speed
            up the procedure, i.e. a non-blocking training process.
        """
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
        """Preparation of the agent before the sample collection starts."""
        super().prepare_agent_args()
        self.agent_args["mode"] = "pre"

    def load_method(self, msg, agent):
        """Method that is called after the master process has finished a
        training cycle.

        Parameters
        ----------
        msg : str
            Message from the master process.
        agent : BasePolicy
            Agent of that agent process.
        """
        return

    def select_action(self, agent, obs):
        """Action selection of the agent. Expert action will be also stored.

        Parameters
        ----------
        agent : BasePolicy
            The agent of that agent process.
        obs : np.ndarray
            Observation from the environment.

        Returns
        -------
        tuple(np.ndarray, dict, int)
            The preprocessed observation so that the agent can make a decision.
            A dictionary with information of the decision making process. The
            action of the agent.

        """
        p_obs = agent.preprocess(obs)
        # pi_action = agent.action(p_obs)
        expert_action = self.expert(self.env, obs)
        pi_action = {}
        pi_action["expert"] = expert_action
        action = int(np.random.randint(low=0, high=2))
        return p_obs, pi_action, action
