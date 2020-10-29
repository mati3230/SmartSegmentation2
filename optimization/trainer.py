import numpy as np
import tensorflow as tf
# from collections import deque
# import time


class Trainer:
    def __init__(
            self,
            optimizer,
            env,
            policy_type,
            n_actions,
            optimization_algo_type,
            log_dir,
            model_name,
            buf,
            test_interval=1,
            model_dir=None,
            n_episodes=1000000,
            verbose=True,
            global_norm=0.5,
            check_numerics=False,
            batch_size=64,
            step_update_f=None,
            pretrainer=None,
            load_pretrain=None,
            preprocess_action=None,
            **kwargs):
        print("-----------------------------------")
        print("Trainer: ", locals())
        print("-----------------------------------")
        self.env = env
        if kwargs["seed"]:
            seed = kwargs["seed"]
            np.random.seed(seed)
            tf.random.set_seed(seed)
        self.train_summary_writer = tf.summary.create_file_writer(log_dir)
        self.n_episodes = n_episodes
        self.save_config(
            log_dir, str(locals()) + ", " + str(optimizer.learning_rate))

        self.optimization_algo = optimization_algo_type(
            policy_type=policy_type,
            n_actions=n_actions,
            optimizer=optimizer,
            train_summary_writer=self.train_summary_writer,
            batch_size=batch_size,
            global_norm=global_norm,
            check_numerics=check_numerics,
            **kwargs)
        self.policy = self.optimization_algo.get_online_policy()

        self.buf = buf

        self.model_name = model_name
        self.model_dir = model_dir
        self.verbose = verbose
        self.step = 0
        self.step_update_f = step_update_f
        self.preprocess_action = preprocess_action
        self.desired_state = False
        self.pretrainer = pretrainer
        self.load_pretrain = load_pretrain
        self.best_reward = 0
        self.test_step = 0
        self.test_interval = test_interval
        self.write_tensorboard = True
        if kwargs["write_tensorboard"]:
            self.write_tensorboard = kwargs["write_tensorboard"]

    def save_config(self, log_dir, config):
        text_file = open(log_dir + "/config.txt", "w")
        text_file.write(config)
        text_file.close()

    def test(self):
        done = False
        episode_reward = 0
        state = self.env.reset()
        episode_length = 0
        while not done:
            pp_state = self.policy.preprocess(state)
            pi_action = self.policy.action(pp_state, training=False)
            action = pi_action["action"]
            if self.preprocess_action:
                action = self.preprocess_action(action)
            state_, reward, done, info = self.env.step(action)
            episode_reward += reward
            episode_length += 1
            state = state_
        if self.write_tensorboard:
            with self.train_summary_writer.as_default():
                tf.summary.scalar(
                    "episodes/test_reward",
                    episode_reward,
                    step=self.test_step)
                tf.summary.scalar(
                    "episodes/test_steps",
                    episode_length,
                    step=self.test_step)
        self.test_step += 1
        if episode_reward > self.best_reward:
            # print("new best model:", episode_reward)
            self.best_reward = episode_reward
            return True
        return False

    def train(self):
        if self.load_pretrain:
            self.optimization_algo.load(
                self.model_dir,
                self.load_pretrain)
        if self.pretrainer:
            self.pretrainer.train()
        episode_reward = 0
        self.step = 0
        state = self.env.reset()
        pp_state = self.policy.preprocess(state)

        for episode in range(self.n_episodes):
            done = False
            len_episode = 0
            while not done:
                # print("step:", self.step)
                pi_action = self.policy.action(pp_state)
                action = pi_action["action"]
                len_episode += 1

                if self.preprocess_action:
                    action = self.preprocess_action(action)
                state_, reward, done, info = self.env.step(action)
                episode_reward = episode_reward + reward
                if self.buf.preprocess_state:
                    self.buf.append(
                        pi_action,
                        action,
                        state,
                        state_,
                        reward,
                        done)
                    pp_state_ = self.policy.preprocess(state_)
                else:
                    pp_state_ = self.policy.preprocess(state_)
                    self.buf.append(
                        pi_action,
                        action,
                        pp_state,
                        pp_state_,
                        reward,
                        done)

                state = state_
                pp_state = pp_state_
                self.step += 1

                if self.step_update_f:
                    self.step_update_f()

                if done:
                    if (episode % self.test_interval) == 0:
                        if self.test():
                            self.policy.save(
                                directory=self.model_dir,
                                filename=self.model_name + "_online_best_" + str(self.best_reward))
                    state = self.env.reset()
                    pp_state = self.policy.preprocess(state)
                    if self.write_tensorboard:
                        with self.train_summary_writer.as_default():
                            tf.summary.scalar(
                                "episodes/reward",
                                episode_reward,
                                step=episode)
                            tf.summary.scalar(
                                "episodes/length",
                                len_episode,
                                step=episode)
                            self.train_summary_writer.flush()
                    episode_reward = 0
                    len_episode = 0
                    self.policy.reset()
        self.env.close()
