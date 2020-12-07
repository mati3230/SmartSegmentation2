import numpy as np
import tensorflow as tf
from .base_pretrainer import BasePretrainer
import os
import random

from deprecated import deprecated

class ImitationPretrainer(BasePretrainer):
    @deprecated
    def __init__(
            self,
            learner,
            train_summary_writer,
            model_name,
            model_dir,
            n_actions,
            check_numerics,
            state_size,
            load_f,
            optimizer=None,
            pretrain_f=None,
            global_norm=0.5,
            epochs=100,
            batch_size=64,
            preprocess_action=None,
            save_pretrain_path=None,
            load_all=False,
            step_update_f=None,
            beta=1,
            test_freq=1,
            env=None,
            early_reward=180):
        super().__init__(
            learner=learner,
            model_name=model_name,
            model_dir=model_dir,
            train_summary_writer=train_summary_writer,
            n_actions=n_actions,
            check_numerics=check_numerics,
            test_freq=test_freq,
            env=env,
            early_reward=early_reward)
        self.optimizer = optimizer
        self.load_f = load_f
        self.batch_size = batch_size
        self.state_size = state_size
        self.pretrain_f = pretrain_f
        self.global_norm = global_norm
        self.save_pretrain_path = save_pretrain_path
        self.preprocess_action = preprocess_action
        self.epochs = epochs
        self.load_all = load_all
        self.step_update_f = step_update_f
        self.beta = beta

    @tf.function
    def update(self, X_batch, y_batch):
        with tf.GradientTape() as tape:
            # action_probs: (N, N_ACTIONS)
            action_probs, _ = self.learner._action_probs(X_batch)

            act_onehot = tf.one_hot(y_batch, depth=self.n_actions)
            ce = tf.nn.softmax_cross_entropy_with_logits(
                labels=act_onehot, logits=action_probs)
            vars_ = tape.watched_variables()
            ce_loss = tf.reduce_mean(ce)
            loss = ce_loss
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

            return loss, global_norm, sum_, tf.reduce_mean(action_probs), ce_loss

    def compute_loss(self, X_batch, y_batch, step):
        loss, global_norm, reg_sum, action_probs, ce_loss =\
            self.update(X_batch, y_batch)
        with self.train_summary_writer.as_default():
            tf.summary.scalar(
                "pretrain/loss", loss, step=step)
            tf.summary.scalar(
                "pretrain/global_norm", global_norm, step=step)
            tf.summary.scalar(
                "pretrain/reg_sum", reg_sum, step=step)
            tf.summary.scalar(
                "pretrain/action_probs", action_probs, step=step)
            tf.summary.scalar(
                "pretrain/ce_loss", ce_loss, step=step)
        if self.step_update_f:
            self.step_update_f(step)

    def train(self):
        files = os.listdir(self.save_pretrain_path)
        random.shuffle(files)
        n_batches = int(np.floor(len(files)/self.batch_size))
        # create batches
        X_batch = np.zeros((tuple([self.batch_size]) + self.state_size))
        y_batch = np.zeros((self.batch_size, 1), np.int32)
        early_stop = False
        if self.load_all:
            X = []
            Y = []
            for i in range(len(files)):
                filename = self.save_pretrain_path + "/" + files[i]
                state = self.load_f(filename)
                X.append(state)
                Y.append(int(files[i][0]))
            for epoch in range(self.epochs):
                for b in range(n_batches):
                    p_start = b*self.batch_size
                    p_end = p_start+self.batch_size
                    for i in range(p_start, p_end, 1):
                        state = X[i]
                        b_idx = i - p_start
                        X_batch[b_idx] = self.learner.preprocess(state)
                        y_batch[b_idx] = Y[i]
                    step = epoch * n_batches + b
                    self.compute_loss(X_batch, y_batch, step)
                if epoch % self.test_freq == 0:
                    early_stop = self.test()
                if early_stop:
                    break
        else:
            for epoch in range(self.epochs):
                for b in range(n_batches):
                    p_start = b*self.batch_size
                    p_end = p_start+self.batch_size
                    batch_files = files[p_start:p_end]
                    for i in range(batch_files):
                        filename = self.save_pretrain_path + "/" + batch_files[i]
                        state = self.load_f(filename)
                        X_batch[i] = self.learner.preprocess(state)
                        y_batch[i] = int(files[i][0])
                    step = epoch * n_batches + b
                    self.compute_loss(X_batch, y_batch, step)
                if epoch % self.test_freq == 0:
                    early_stop = self.test()
                if early_stop:
                    break
        self.learner.save(
            directory=self.model_dir,
            filename=self.model_name + "_IT")
        print("pretrain finished")
        if self.pretrain_f:
            self.pretrain_f()
