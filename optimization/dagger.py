import numpy as np
import tensorflow as tf
import math
from .utils import normalize
from .base_pretrainer import BasePretrainer
from .tf_utils import sync


class DAgger(BasePretrainer):
    def __init__(
            self,
            optimizer,
            expert,
            env_type,
            env_args,
            policy_type,
            policy_args,
            n_actions,
            log_dir,
            model_dir,
            model_name,
            state_size,
            train_summary_writer=None,
            global_norm=None,
            iterations=50,
            batch_size=64,
            maxlen=2000,
            save_state_f=None,
            beta=1,
            test_freq=1,
            value_factor=0.5,
            gamma=0.99,
            ce_factor=1,
            normalize_returns=False,
            check_numerics=False,
            shared_value=None):
        """
        iterations:
            Nr of iterations to sample data and training.
        maxlen:
            Size of the whole batch/dataset per iteration.
        save_state_f:
            Function to save a state action pair. The input args are the state,
            filename, action.
        """
        super().__init__(
            env_type=env_type,
            env_args=env_args,
            policy_type=policy_type,
            policy_args=policy_args,
            model_name=model_name,
            model_dir=model_dir,
            log_dir=log_dir,
            train_summary_writer=train_summary_writer,
            n_actions=n_actions,
            optimizer=optimizer,
            check_numerics=check_numerics,
            test_freq=test_freq,
            shared_value=shared_value
            )
        self.expert = expert
        self.state_size = state_size
        self.iterations = iterations
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.global_norm = global_norm
        self.save_state_f = save_state_f
        self.beta = beta
        self.value_factor = value_factor
        self.gamma = gamma
        self.ce_factor = ce_factor
        self.normalize_returns = normalize_returns

    @tf.function
    def update(self, X_batch, y_batch, returns_batch):
        with tf.GradientTape() as tape:
            # action_probs: (N, N_ACTIONS)
            action_probs, _ = self.learner._action_probs(X_batch)

            act_onehot = tf.one_hot(y_batch, depth=self.n_actions)
            ce = tf.nn.softmax_cross_entropy_with_logits(
                labels=act_onehot, logits=action_probs)
            _, state_values, entropy, _ =\
                self.learner.evaluate(X_batch, y_batch)
            critic_loss = tf.math.square(returns_batch - state_values)
            vars_ = tape.watched_variables()
            ce_loss = tf.reduce_mean(ce)
            critic_loss = 0.5 * tf.reduce_mean(critic_loss)
            entropy = tf.reduce_mean(entropy)
            loss = self.ce_factor * ce_loss + self.value_factor * critic_loss
            sum_ = 0
            for var in vars_:
                sum_ += tf.reduce_sum(tf.square(var))
            loss += self.beta * 0.5 * sum_
            if self.check_numerics:
                loss = tf.debugging.check_numerics(loss, "loss")

            grads = tape.gradient(loss, vars_)
            global_norm = tf.linalg.global_norm(grads)
            if self.global_norm:
                grads, _ = tf.clip_by_global_norm(
                    grads, self.global_norm, use_norm=global_norm)
            self.optimizer.apply_gradients(zip(grads, vars_))

            return loss, global_norm, sum_, tf.reduce_mean(action_probs), ce_loss, critic_loss, entropy

    def train(self):
        X = np.zeros((tuple([self.maxlen]) + self.state_size))
        y = np.zeros((self.maxlen, 1), np.int32)
        masks = np.zeros((self.maxlen, 1), np.bool)
        rewards = np.zeros((self.maxlen, 1), np.float32)
        returns = np.zeros((self.maxlen, 1), np.float32)
        buf_idx = 0

        # only used for logging
        episode = 0
        episode_reward = 0
        n_learner_actions = 0
        step = 0
        tstep = 0

        def clear():
            nonlocal X, y, returns, rewards, masks
            del X
            del y
            del returns
            del rewards
            del masks

        for i in range(self.iterations):
            state = self.env.reset()
            while True:
                if not self.shared_value:
                    clear()
                    return
                action = 0
                expert_action = self.expert(self.env, state)
                if self.save_state_f:
                    key = self.expert.compute_key(state)
                    self.save_state_f(state, key, expert_action)
                state = self.test_policy.preprocess(state)
                if np.random.rand() < (
                        1 - (i/self.iterations)):
                    action = expert_action
                else:
                    pi_action = self.test_policy.action(state, training=False)
                    action = pi_action["action"]
                    n_learner_actions += 1
                    action = self.test_policy.preprocess_action(action)
                state_, reward, done, info = self.env.step(action)
                episode_reward += reward
                X[buf_idx] = state
                y[buf_idx] = expert_action
                rewards[buf_idx] = reward
                masks[buf_idx] = not done
                buf_idx += 1
                state = state_
                step += 1
                if done or buf_idx == self.maxlen:
                    if done:
                        with self.train_summary_writer.as_default():
                            tf.summary.scalar(
                                "pretrain/reward", episode_reward, step=episode)
                            tf.summary.scalar(
                                "pretrain/n_learner_actions",
                                n_learner_actions,
                                step=episode)
                            tf.summary.scalar(
                                "pretrain/steps",
                                step,
                                step=episode)
                        episode += 1
                    episode_reward = 0
                    n_learner_actions = 0
                    step = 0
                    # reset the buffer idx to sample new trajectories
                    self.test_policy.reset()
                    state = self.env.reset()
                    if buf_idx == self.maxlen:
                        buf_idx = 0
                        # maxlen reached -> continue with optimization
                        break
            # training
            len_X = self.maxlen
            prev_return = 0
            last_reward_idx = -1
            for i in reversed(range(len_X)):
                if last_reward_idx == -1:
                    if rewards[i] == 0:
                        continue
                    last_reward_idx = i
                prev_return = rewards[i] +\
                    (self.gamma * prev_return * masks[i])
                returns[i] = prev_return
            len_X = last_reward_idx + 1
            if len_X < self.batch_size:
                continue

            if self.learner.use_lstm:
                batch_indxs = np.where(masks == False)[0]
                n_batches = batch_indxs.shape[0]
                batch_indxs = np.vstack(
                    (np.array([0], np.int32)[:, None], batch_indxs[:, None]))
                batch_indxs = batch_indxs.reshape((batch_indxs.shape[0], ))
            else:
                n_batches = math.floor(len_X / self.batch_size)
                batch_indxs = np.arange(len_X)
                np.random.shuffle(batch_indxs)
            if self.normalize_returns:
                returns = normalize(returns)

            for b in range(n_batches):
                if not self.shared_value:
                    clear()
                    return
                if self.learner.use_lstm:
                    indxs = np.arange(batch_indxs[b], batch_indxs[b+1])
                else:
                    b_start = b * self.batch_size
                    indxs = batch_indxs[b_start:b_start+self.batch_size]
                # mini-batch training
                X_batch = X[indxs]
                y_batch = y[indxs]
                returns_batch = returns[indxs]
                loss, global_norm, var_sum, action_probs, ce_loss, critic_loss, mean_entropy =\
                    self.update(X_batch, y_batch, returns_batch)
                with self.train_summary_writer.as_default():
                    tf.summary.scalar(
                        "pretrain/loss", loss, step=tstep)
                    tf.summary.scalar(
                        "pretrain/global_norm", global_norm, step=tstep)
                    tf.summary.scalar(
                        "pretrain/var_sum", var_sum, step=tstep)
                    # tf.summary.scalar(
                    #     "pretrain/action_probs", action_probs, step=tstep)
                    tf.summary.scalar(
                        "pretrain/ce_loss", ce_loss, step=tstep)
                    tf.summary.scalar(
                        "pretrain/critic_loss", critic_loss, step=tstep)
                    tf.summary.scalar(
                        "pretrain/mean returns",
                        np.mean(returns_batch), step=tstep)
                    tf.summary.scalar(
                        "pretrain/mean entropy", mean_entropy, step=tstep)
                tstep += 1
            sync(self.learner, self.test_policy)
            if i % self.test_freq == 0:
                self.test()
                if self.test_reward > self.best_test_reward:
                    self.best_test_reward = self.test_reward
                    self.learner.save(
                        directory=self.model_dir,
                        filename=self.model_name + "_" + str(self.best_test_reward) +  "_DAgger")

        clear()
