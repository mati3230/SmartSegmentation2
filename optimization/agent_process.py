from multiprocessing import Process
import threading
import tensorflow as tf
import numpy as np
from .tf_utils import sync2
import os
import time


class AgentProcess(Process):
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
        super().__init__()
        self.conn = conn
        self.n_steps = n_steps
        self.id = id
        self.msg_queue = []
        self.agent_type = agent_type
        self.agent_args = agent_args
        self.env_type = env_type
        self.env_args = env_args
        np.random.seed(seed + id)
        tf.random.set_seed(seed + id)
        self.model_dir = model_dir
        self.model_filename = model_filename
        self.w_gpu_mem = add_args["w_gpu_mem"]
        self.transitions = []
        self.sample = True
        self.async_mode = async_mode
        self.n_cpus = n_cpus

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
        agent_vars = msg[1]
        sync2(agent_vars, agent)

    def select_action(self, agent, obs):
        """Action selection of the agent.

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
        pi_action = agent.action(p_obs, training=False)
        # print("step")
        action = pi_action["action"]
        # action = action.numpy()[0]
        action = agent.preprocess_action(action)
        return p_obs, pi_action, action

    def prepare_agent_args(self):
        """Preparation of the agent before the sample collection starts."""
        self.agent_args["trainable"] = True
        self.agent_args["stateful"] = True

    def run(self):
        """Start of the sample collection."""
        print("Agent Process ID:", os.getpid(), "internal:", self.id)
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            try:
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=self.w_gpu_mem)])
                logical_gpus = tf.config.experimental.list_logical_devices("GPU")
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                print(e)
        self.prepare_agent_args()
        agent = self.agent_type(**self.agent_args)
        # agent.load(self.model_dir, self.model_filename)
        train = True

        def treatQueue():
            nonlocal train, agent
            if self.async_mode:
                # print("sampler", self.id, "ready")
                while True:
                    msg = self.conn.recv()
                    if msg[0] == "load":
                        self.load_method(msg, agent)
                        while True:
                            # if master is faster than agent process
                            if len(self.transitions) <= self.n_steps:
                                time.sleep(0.03)
                                continue
                            self.conn.send((self.transitions, ))
                            self.sample = True
                            # print("sampler", self.id, "transitions send")
                            break
                    elif msg[0] == "stop":
                        break
            else:
                msg = self.conn.recv()
                if msg[0] == "load":
                    self.load_method(msg, agent)
                elif msg[0] == "stop":
                    train = False

        if self.async_mode:
            t = threading.Thread(target=treatQueue)
            t.start()

        self.env_args["batch_id"] = self.id
        self.env_args["n_cpus"] = self.n_cpus
        self.env = self.env_type(**self.env_args)
        while train:
            # sample in async_mode if data is send
            if self.async_mode and not self.sample:
                time.sleep(0.03)
                continue
            # print("sampler", self.id, "start sampling")
            # print("Process", self.id, "starts playing", self.n_steps, "steps")
            self.transitions = []
            done = False
            obs = self.env.reset()
            while not done:
                p_obs, pi_action, action = self.select_action(agent, obs)
                next_obs, reward, done, info = self.env.step(action)
                # p_next_obs = agent.preprocess(next_obs)
                self.transitions.append((
                    pi_action,
                    action,
                    np.array(p_obs, copy=True),
                    None,
                    reward,
                    done))
                obs = next_obs
                if done:
                    # agent.reset()
                    if len(self.transitions) > self.n_steps:
                        self.sample = False
                        # print("sampler", self.id, "sampling finished")
                        break
                    done = False
                    obs = self.env.reset()
            if train and not self.async_mode:
                self.conn.send((self.transitions, ))
                treatQueue()
        print("done: agent", self.id)
