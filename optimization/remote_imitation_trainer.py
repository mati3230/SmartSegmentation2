from .base_remote_trainer import BaseRemoteTrainer
from .expert_imitation import ExpertImitation
from .utils import get_type
import tensorflow as tf
import numpy as np


class RemoteImitationTrainer(BaseRemoteTrainer):
    def __init__(self, args_file, types_file):
        """Constructor. This class specifies the train method for the imitation
        learning algorithm (see imitation_types.json).

        Parameters
        ----------
        args_file : str
            Path (relative or absolute) to a json file where the parameters are
            specified.
        types_file : str
            Path (relative or absolute) to a json file where the types of the
            parameters are specified.
        """
        super().__init__(args_file=args_file, types_file=types_file)

    def train_method(
            self,
            shared_value,
            params,
            env_type,
            env_args,
            policy_type,
            n_ft_outpt,
            n_actions,
            train_log_dir,
            model_dir,
            model_name,
            test_interval):
        """This methods starts the training procedure. It creates a
        MultiProcessTrainer instance to apply a training on multiple cores.

        Parameters
        ----------
        shared_value : multiprocessing.Value
            Shared value of the multiprocessing library to stop the training
            process over multi processes.
        params : dictionary
            Dictionary where parameters such as the path of the environment
            class or data provider class are specified. It should also store
            parameters for the training such as the batch size. The parameter
            types can be found in, e.g., imitation_types.json.
        env_type : type
            Type of the environment.
        env_args : dictionary
            Parameters of the environment.
        policy_type : type
            Type of the policy such as a specific actor critic neural net.
        n_ft_outpt : int
            Number of features the will a specific net should output before a
            action and a state value mlp are calculated. In other words, the
            number of neurons that is used by the action and value function.
            Note that we split a neural net in a general feature approximator
            and, e.g., an action approximator
        n_actions : int
            Number of actions that are available in the environment.
        train_log_dir : str
            Directory where the logs will be saved.
        model_dir : str
            Directory where the weights of the nets will be saved.
        model_name : str
            Name of the neural net.
        test_interval : int
            Number that specifies after how many training updates a test is
            calculated. Note that we use a train test split.
        """

        expert_type = get_type(params["expert_path"], params["expert_type"])

        np.random.seed(params["seed"])
        tf.random.set_seed(params["seed"])

        expert_imi = ExpertImitation(
            optimizer=tf.optimizers.Adam(
                learning_rate=params["learning_rate"]),
            n_cpus=params["n_cpus"],
            w_gpu_mem=params["w_gpu_mem"],
            expert_type=expert_type,
            expert_args=None,
            env_type=env_type,
            env_args=env_args,
            policy_type=policy_type,
            policy_args={
                "name": "target_policy",
                "n_ft_outpt": n_ft_outpt,
                "n_actions": n_actions,
                "state_size": params["state_size"],
                "seed": params["seed"],
                "stddev": params["stddev"],
                "initializer": params["initializer"],
                "mode": "half"},
            n_actions=n_actions,
            log_dir=train_log_dir,
            model_dir=model_dir,
            model_name=model_name,
            state_size=params["state_size"],
            global_norm=params["global_norm"],
            batch_size=params["batch_size"],
            n_batches=params["n_batches"],
            beta=params["beta"],
            test_freq=test_interval,
            ce_factor=params["ce_factor"],
            shared_value=shared_value)

        expert_imi.train()
