import tensorflow as tf
import numpy as np
from .master_process import MasterProcess
from .agent_process import AgentProcess
from .tf_utils import sync
from .utils import save_config
import os


class MultiProcessTrainer():
    """Short summary.

    Parameters
    ----------

    optimizer : tf.keras.optimizers
        An optimizer such as the Adam optimizer.
    env_type : type
        Type of the environment.
    env_args : dictionary
        Parameters of the environment.
    policy_type : type
        Type of the policy such as a specific actor critic neural net.
    policy_args : dictionary
        Arguments of a specific policy.
    n_actions : int
        Number of actions that are available in the environment.
    optimization_algo_type : type
        Type of the optimization algorithm such as PPO2.
    log_dir : str
        Directory where the logs will be saved.
    model_dir : str
        Directory where the weights of the nets will be saved.
    model_name : str
        Name of the neural net.
    n_cpus : int
        Number of CPUs that will be used.
    w_gpu_mem : int
        Size of gpu memory that is used by the worker processes.
    n_batches : int
        Number of batches that is used in the training.
    batch_size : int
        Size of a batch.
    global_norm : float
        Threshold to clip the gradients according to a maximum global norm.
    check_numerics : boolean
        If True, an exception is thrown in case of NaN values.
    shared_value : multiprocessing.Value
        Shared value of the multiprocessing library to stop the training
        process over multi processes.
    **kwargs : dict
        Additional arguments.

    Attributes
    ----------
    n_steps : type
        Description of attribute `n_steps`.
    train_summary_writer : tf.summary.SummaryWriter
        Summary writer to write tensorboard logs.
    optimization_algo : type
        Class type of the optimization algorithm such as PPO.
    policy : BasePolicy
        Policy that will be optimized by the optimization algoroithm such as
        PPO.
    test_policy : BasePolicy
        Policy that is used for testing.
    master_process : MasterProcess
        Process that controls the agent processes which sample data. The master
        process controls also the policy update cycle.
    env : BaseEnvironment
        Instance of the environment.
    best_reward : float
        Best test reward so far.
    test_step : int
        Number of test steps so far.
    write_tensorboard : boolean
        If True, tensorboard will be written.
    model_dir : str
        Directory where the weights of the nets will be saved.
    model_name : str
        Name of the neural net.
    n_cpus : int
        Number of CPUs that will be used.
    shared_value : multiprocessing.Value
        Shared value of the multiprocessing library to stop the training
        process over multiple processes.

    """
    def __init__(
            self,
            optimizer,
            env_type,
            env_args,
            policy_type,
            policy_args,
            n_actions,
            optimization_algo_type,
            log_dir,
            model_dir,
            model_name,
            n_cpus,
            w_gpu_mem,
            n_batches,
            batch_size,
            global_norm=0.5,
            check_numerics=False,
            shared_value=None,
            **kwargs):
        """Constructor.

        Parameters
        ----------
        optimizer : tf.keras.optimizers
            An optimizer such as the Adam optimizer.
        env_type : type
            Type of the environment.
        env_args : dictionary
            Parameters of the environment.
        policy_type : type
            Type of the policy such as a specific actor critic neural net.
        policy_args : dictionary
            Arguments of a specific policy.
        n_actions : int
            Number of actions that are available in the environment.
        optimization_algo_type : type
            Type of the optimization algorithm such as PPO2.
        log_dir : str
            Directory where the logs will be saved.
        model_dir : str
            Directory where the weights of the nets will be saved.
        model_name : str
            Name of the neural net.
        n_cpus : int
            Number of CPUs that will be used.
        w_gpu_mem : int
            Size of gpu memory that is used by the worker processes.
        n_batches : int
            Number of batches that is used in the training.
        batch_size : int
            Size of a batch.
        global_norm : float
            Threshold to clip the gradients according to a maximum global norm.
        check_numerics : boolean
            If True, an exception is thrown in case of NaN values.
        shared_value : multiprocessing.Value
            Shared value of the multiprocessing library to stop the training
            process over multi processes.
        **kwargs : dict
            Additional arguments.
        """
        print("Master Process ID:", os.getpid())
        self.n_cpus = n_cpus
        self.n_steps = n_batches * batch_size
        self.train_summary_writer = tf.summary.create_file_writer(log_dir)
        # log some parameters
        save_config(
            log_dir, str(locals()) + ", " + str(optimizer.learning_rate))
        # initialize the optimization algorithm such as the ppo algorithm
        self.optimization_algo = optimization_algo_type(
            policy_type=policy_type,
            policy_args=policy_args,
            n_actions=n_actions,
            optimizer=optimizer,
            train_summary_writer=self.train_summary_writer,
            batch_size=batch_size,
            global_norm=global_norm,
            check_numerics=check_numerics,
            **kwargs)
        # assign a test policy
        self.policy = self.optimization_algo.target_policy
        if self.policy.use_lstm:
            test_args = policy_args.copy()
            test_args["trainable"] = False
            test_args["stateful"] = True
            self.test_policy = policy_type(**test_args)
        else:
            self.test_policy = self.policy
        # assign the gpu memory of the workers
        agent_process_args = {
            "w_gpu_mem": w_gpu_mem
        }

        # the master process manages the worker processes
        self.master_process = MasterProcess(
            n_process=self.n_cpus,
            n_samples=self.n_steps,
            agent_type=policy_type,
            agent_args=policy_args,
            env_type=env_type,
            env_args=env_args,
            model_dir=model_dir,
            model_filename=model_name,
            agent=self.policy,
            shared_value=shared_value,
            agent_process_type=AgentProcess,
            agent_process_args=agent_process_args,
            **kwargs)
        self.model_name = model_name
        self.model_dir = model_dir
        # initialize an environment
        self.env = env_type(**env_args)
        state = self.env.reset()
        state = self.test_policy.preprocess(state)
        self.test_policy.action(state, training=False)
        self.test_policy.reset()
        self.best_reward = 0
        self.test_step = 0
        self.write_tensorboard = True
        if kwargs["write_tensorboard"]:
            self.write_tensorboard = kwargs["write_tensorboard"]
        self.shared_value = shared_value

    def train(self):
        """Start of the training."""
        self.master_process.sample_transitions()

    def stop(self):
        """Stop the training by flipping the multiprocess value."""
        self.master_process.train = False

    def test(self):
        """Test the model.

        Returns
        -------
        boolean
            Returns True if a new best model is found.

        """
        # get the number of test scenes
        n_test_scenes=1
        if hasattr(self.env, "_data_prov"):
            if self.env._data_prov.train_mode:
                n_test_scenes = len(self.env._data_prov.test_scenes)
            else:
                n_test_scenes = len(self.env._data_prov.scenes)
        # update the test policy
        sync(self.policy, self.test_policy)
        policy = self.test_policy
        # ratios between episode rewards and episode lengths
        ratios = []
        rewards = []
        # container to store the episode lengths
        steps = []
        for i in range(n_test_scenes):
            done = False
            episode_reward = 0
            episode_length = 0
            state = self.env.reset(train=False)
            if policy.use_lstm:
                policy.reset()
            while not done:
                # calculate an action
                pp_state = policy.preprocess(state)
                pi_action = policy.action(pp_state, training=False)
                action = pi_action["action"]
                action = policy.preprocess_action(action)

                # apply the action
                state_, reward, done, info = self.env.step(action)
                episode_reward += reward
                episode_length += 1
                # set the state for the next iteration
                state = state_
            ratio = episode_reward/episode_length
            ratios.append(ratio)
            rewards.append(episode_reward)
            steps.append(episode_length)
        mean_ratio = np.mean(ratios)
        mean_reward = np.mean(rewards)
        # log the containers
        with self.train_summary_writer.as_default():
            tf.summary.scalar(
                "test/mean_reward", mean_reward, step=self.test_step)
            tf.summary.scalar(
                "test/rew_steps", mean_ratio, step=self.test_step)
            tf.summary.scalar(
                "test/rew_steps_std", np.std(ratios), step=self.test_step)
        self.test_step += 1
        self.test_reward = mean_ratio
        # new best policy
        if mean_ratio > self.best_reward:
            self.best_reward = mean_ratio
            return True
        return False
