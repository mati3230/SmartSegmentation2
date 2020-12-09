from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np
from .utils import save_config
from .tf_utils import sync


class BasePretrainer(ABC):
    """Abstract class to realise a trainer for supervised learning in an
    environment.

    Parameters
    ----------
    n_cpus : int
        Number of agent processes.
    env_type : type
        Class type of the environment to generate a copy in the agent
        processes.
    env_args : dict
        Input arguments of the environment class to generate a copy in the
        agent processes.
    policy_type : type
        Type of the policy that should be trained to initialize the target
        policy.
    policy_args : dictionary
        Arguments that are used to initialize instances of the policy.
    model_name : str
        Name of the neural net.
    model_dir : str
        Directory where the models will be stored.
    log_dir : str
        Directory where the logs will be saved.
    train_summary_writer : tf.summary.SummaryWriter
        Summary writer to write tensorboard logs.
    n_actions : int
        Number of available actions.
    optimizer : tf.keras.optimizers.Optimizer
        Optimizer such as SGD or ADAM.
    check_numerics : boolean
        If True, an exception is thrown in case of NaN values.
    test_freq : int
        Number that specifies after how many training updates a test is
        calculated. Note that we use a train test split.
    shared_value : multiprocessing.Value
        Shared value of the multiprocessing library to stop the training
        process.

    Attributes
    ----------
    learner : type
        Description of attribute `learner`.
    env : BaseEnvironment
        Instance of the environment to test the policy.
    test_policy : BasePolicy
        The policy is only tested in the environment.
    test_step : int
        Number of tests so far.
    test_reward : float
        Reward that was collected during the test.
    best_test_reward : float
        Best test result so far.
    n_cpus : int
        Number of agent processes.
    model_name : str
        Name of the neural net.
    model_dir : str
        Directory where the models will be stored.
    log_dir : str
        Directory where the logs will be saved.
    train_summary_writer : tf.summary.SummaryWriter
        Summary writer to write tensorboard logs.
    n_actions : int
        Number of available actions.
    optimizer : tf.keras.optimizers.Optimizer
        Optimizer such as SGD or ADAM.
    check_numerics : boolean
        If True, an exception is thrown in case of NaN values.
    test_freq : int
        Number that specifies after how many training updates a test is
        calculated. Note that we use a train test split.
    shared_value : multiprocessing.Value
        Shared value of the multiprocessing library to stop the training
        process over multi processes.

    """
    def __init__(
            self,
            n_cpus,
            env_type,
            env_args,
            policy_type,
            policy_args,
            model_name,
            model_dir,
            log_dir,
            train_summary_writer,
            n_actions,
            optimizer,
            check_numerics=False,
            test_freq=100,
            shared_value=None):
        """Constructor.

        Parameters
        ----------
        n_cpus : int
            Number of agent processes.
        env_type : type
            Class type of the environment to generate a copy in the agent
            processes.
        env_args : dict
            Input arguments of the environment class to generate a copy in the
            agent processes.
        policy_type : type
            Type of the policy that should be trained to initialize the target
            policy.
        policy_args : dictionary
            Arguments that are used to initialize instances of the policy.
        model_name : str
            Name of the neural net.
        model_dir : str
            Directory where the models will be stored.
        log_dir : str
            Directory where the logs will be saved.
        train_summary_writer : tf.summary.SummaryWriter
            Summary writer to write tensorboard logs.
        n_actions : int
            Number of available actions.
        optimizer : tf.keras.optimizers.Optimizer
            Optimizer such as SGD or ADAM.
        check_numerics : boolean
            If True, an exception is thrown in case of NaN values.
        test_freq : int
            Number that specifies after how many training updates a test is
            calculated. Note that we use a train test split.
        shared_value : multiprocessing.Value
            Shared value of the multiprocessing library to stop the training
            process over multi processes.
        """
        super().__init__()
        self.n_cpus = n_cpus
        policy_args["trainable"] = True
        policy_args["stateful"] = False
        self.learner = policy_type(**policy_args)
        self.model_name = model_name
        self.model_dir = model_dir
        if train_summary_writer:
            self.train_summary_writer = train_summary_writer
        else:
            self.train_summary_writer = tf.summary.create_file_writer(log_dir)
        save_config(
            log_dir, str(locals()) + ", " + str(optimizer.learning_rate))
        self.n_actions = n_actions
        self.optimizer = optimizer
        self.check_numerics = check_numerics

        self.test_freq = test_freq
        self.env = env_type(**env_args)

        if self.learner.use_lstm:
            state = self.env.reset()
            test_args = policy_args.copy()
            test_args["trainable"] = False
            test_args["stateful"] = True
            self.test_policy = policy_type(**test_args)

            state = self.test_policy.preprocess(state)
            self.test_policy.action(state, training=False)
            self.test_policy.reset()
            sync(self.learner, self.test_policy)
        self.test_step = 0
        self.shared_value = shared_value
        self.test_reward = 0
        self.best_test_reward = 0

    @abstractmethod
    def train():
        pass

    def test(self):
        if self.env._data_prov.train_mode:
            n_test_scenes = len(self.env._data_prov.test_scenes)
        else:
            n_test_scenes = len(self.env._data_prov.scenes)
        policy = self.learner
        ratios = []
        rewards = []
        steps = []
        for i in range(n_test_scenes):
            done = False
            episode_reward = 0
            episode_length = 0
            state = self.env.reset(train=False)
            if policy.use_lstm:
                policy.reset()
            while not done:
                # print("test")
                pp_state = policy.preprocess(state)
                pi_action = policy.action(pp_state, training=False)
                action = pi_action["action"]
                action = policy.preprocess_action(action)
                state_, reward, done, info = self.env.step(action)
                episode_reward += reward
                episode_length += 1
                state = state_
            ratio = episode_reward/episode_length
            ratios.append(ratio)
            rewards.append(episode_reward)
            steps.append(episode_length)
        mean_ratio = np.mean(ratios)
        with self.train_summary_writer.as_default():
            tf.summary.scalar(
                "test/rew_steps", mean_ratio, step=self.test_step)
            tf.summary.scalar(
                "test/rew_steps_std", np.std(ratios), step=self.test_step)
        self.test_step += 1
        self.test_reward = mean_ratio
