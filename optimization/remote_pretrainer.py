from .base_remote_trainer import BaseRemoteTrainer
from .dagger import DAgger
from .utils import get_type
import tensorflow as tf
import numpy as np

from deprecated import deprecated

class RemotePretrainer(BaseRemoteTrainer):
    @deprecated
    def __init__(self, args_file, types_file):
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

        expert_type = get_type(params["expert_path"], params["expert_type"])
        expert = expert_type()

        np.random.seed(params["seed"])
        tf.random.set_seed(params["seed"])

        def save_state_f(state, key, expert_action):
            P_idxs = state[1]
            neighbour_idxs = state[2]
            segments = state[3]
            np.savez(
                "./data/pcseg/" + str(expert_action) + "_" + str(key),
                P_idxs=P_idxs,
                neighbour_idxs=neighbour_idxs,
                segments=segments)
        pretrainer = DAgger(
            optimizer=tf.optimizers.Adam(
                learning_rate=params["learning_rate"]),
            expert=expert,
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
                "initializer": params["initializer"]},
            n_actions=n_actions,
            log_dir=train_log_dir,
            model_dir=model_dir,
            model_name=model_name,
            state_size=params["state_size"],
            global_norm=params["global_norm"],
            iterations=params["iterations"],
            batch_size=params["batch_size"],
            maxlen=params["maxlen"],
            save_state_f=save_state_f,
            beta=params["beta"],
            test_freq=test_interval,
            value_factor=params["value_factor"],
            gamma=params["gamma"],
            ce_factor=params["ce_factor"],
            normalize_returns=params["normalize_returns"],
            shared_value=shared_value)

        pretrainer.train()
