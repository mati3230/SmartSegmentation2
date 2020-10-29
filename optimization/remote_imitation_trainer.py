from .base_remote_trainer import BaseRemoteTrainer
from .expert_imitation import ExpertImitation
from .utils import get_type
import tensorflow as tf
import numpy as np


class RemoteImitationTrainer(BaseRemoteTrainer):
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
