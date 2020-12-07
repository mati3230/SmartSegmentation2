from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np
from .utils import save_config
from .tf_utils import sync

from deprecated import deprecated

class BasePretrainer(ABC):
    @deprecated
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
