import tensorflow as tf
import numpy as np
from .master_process import MasterProcess
from .agent_process import AgentProcess
from .tf_utils import sync
from .utils import save_config
import os


class MultiProcessTrainer():
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
        print("Master Process ID:", os.getpid())
        self.n_cpus = n_cpus
        self.n_steps = n_batches * batch_size
        self.train_summary_writer = tf.summary.create_file_writer(log_dir)
        save_config(
            log_dir, str(locals()) + ", " + str(optimizer.learning_rate))

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
        self.policy = self.optimization_algo.target_policy
        if self.policy.use_lstm:
            test_args = policy_args.copy()
            test_args["trainable"] = False
            test_args["stateful"] = True
            self.test_policy = policy_type(**test_args)
        else:
            self.test_policy = self.policy
        agent_process_args = {
            "w_gpu_mem": w_gpu_mem
        }

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
        self.master_process.sample_transitions()

    def stop(self):
        self.master_process.train = False

    def test(self):
        if self.env._data_prov.train_mode:
            n_test_scenes = len(self.env._data_prov.test_scenes)
        else:
            n_test_scenes = len(self.env._data_prov.scenes)
        sync(self.policy, self.test_policy)
        policy = self.test_policy
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
        if mean_ratio > self.best_reward:
            self.best_reward = mean_ratio
            return True
        return False
