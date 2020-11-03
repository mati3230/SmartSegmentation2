import tensorflow as tf
from .multi_process_trainer import MultiProcessTrainer
from .ppo2 import PPO2
from .base_remote_trainer import BaseRemoteTrainer


class RemotePPO2Trainer(BaseRemoteTrainer):
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

        train_step = 0
        ppo2 = trainer.optimization_algo

        def update_f(transitions):
            nonlocal train_step
            ppo2.update(transitions, train_step)
            # ppo2.target_policy.save(model_dir, model_name)
            train_step += 1
            if train_step % test_interval == 0:
                if trainer.test():
                    ppo2.target_policy.save(
                        model_dir, model_name + "_target_" + str(trainer.best_reward))
        trainer.master_process.update_f = update_f
        self.trainer = trainer
        self.trainer.train()
