import tensorflow as tf
from .multi_process_trainer import MultiProcessTrainer
from .ppo2 import PPO2
from .base_remote_trainer import BaseRemoteTrainer


class RemotePPO2Trainer(BaseRemoteTrainer):
    """This class specifies the train method to use the PPO2
    algorithm. PPO2-specific parameters are set in the _args.json file such as
    ppo2_types.json. See BaseRemoteTrainer for more attributes.

    Parameters
    ----------
    args_file : str
        Path (relative or absolute) to a json file where the parameters are
        specified.
    types_file : str
        Path (relative or absolute) to a json file where the types of the
        parameters are specified.

    Attributes
    ----------
    trainer : MultiProcessTrainer
        Instance of a MultiProcessTrainer to start the training.

    """
    def __init__(self, args_file, types_file):
        """Constructor.

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
            types can be found in, e.g., ppo2_types.json.
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
        trainer = MultiProcessTrainer(
            optimizer=tf.optimizers.Adam(
                learning_rate=params["learning_rate"]),
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
                "mode": "full"},
            n_actions=n_actions,
            optimization_algo_type=PPO2,
            log_dir=train_log_dir,
            model_dir=model_dir,
            model_name=model_name,
            n_cpus=params["n_cpus"],
            w_gpu_mem=params["w_gpu_mem"],
            n_batches=params["n_batches"],
            batch_size=params["batch_size"],
            global_norm=params["global_norm"],
            seed=params["seed"],
            gamma=params["gamma"],
            K_epochs=params["K_epochs"],
            eps_clip=params["eps_clip"],
            lmbda=params["lmbda"],
            entropy_factor=params["entropy_factor"],
            value_factor=params["value_factor"],
            normalize_returns=params["normalize_returns"],
            normalize_advantages=params["normalize_advantages"],
            write_tensorboard=True,
            shared_value=shared_value
        )
        # integer to count the number of training steps
        train_step = 0
        # store a reference to the optimization algorithm
        ppo2 = trainer.optimization_algo

        def update_f(transitions):
            """Function that applies a ppo update.

            Parameters
            ----------
            transitions : dictionary
                The keys of this dictionary are process ids. If only a single
                process is used, only one element is in the dictionary.
                The values of the dictionary are lists. A single list contains
                tuples that are actually the transitions.
                A tuple consist of:
                    * dictionary that contains informartion from the agent,
                    * the agent action,
                    * the observation according to which the agent selects its
                      action,
                    * the next observation of the environment,
                    * the reward value and
                    * a flag if the episode is over.

            """
            # necessary to use the variable which is defined above
            nonlocal train_step
            # apply a ppo2 update
            ppo2.update(transitions, train_step)
            train_step += 1
            # after n training steps
            if train_step % test_interval == 0:
                if trainer.test(): # if test produces a new best model
                    ppo2.target_policy.save(
                        model_dir, model_name + "_target_" + str(trainer.best_reward))
        # link the update function with the master process
        # it will call the update if enough data is available
        trainer.master_process.update_f = update_f
        # start the training
        self.trainer = trainer
        self.trainer.train()
