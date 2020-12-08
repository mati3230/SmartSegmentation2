import tensorflow as tf
import numpy as np
import math
from .tf_utils import sync
from .base_pretrainer import BasePretrainer
from .master_process import MasterProcess
from .expert_process import ExpertProcess
import cv2
import time

class ExpertImitation(BasePretrainer):
    def __init__(
            self,
            n_cpus,
            w_gpu_mem,
            expert_type,
            expert_args,
            env_type,
            env_args,
            policy_type,
            policy_args,
            model_name,
            model_dir,
            log_dir,
            n_actions,
            optimizer,
            state_size,
            test_freq=1,
            shared_value=None,
            train_summary_writer=None,
            ce_factor=1,
            beta=0,
            check_numerics=False,
            global_norm=None,
            batch_size=64,
            n_batches=5
            ):
        """Short summary.

        Parameters
        ----------
        n_cpus : int
            Number of agent processes.
        w_gpu_mem : int
            Size of gpu memory that is used by the worker processes.
        expert_type : type
            Class type of the expert agent.
        expert_args : dict
            Arguments to initialize the expert agent.
        env_type : type
            Class type of the environment to generate a copy in the agent
            processes.
        env_args : dict
            Input arguments of the environment class to generate a copy in the
            agent processes.
        policy_type : type
            Type of the policy that should be trained to initialize the target
            policy.
        policy_args : dictionary
            Arguments that are used to initialize instances of the policy.
        model_name : str
            Name of the neural net.
        model_dir : str
            Directory where the models will be stored.
        log_dir : str
            Directory where the logs will be saved.
        n_actions : int
            Number of available actions.
        optimizer : tf.keras.optimizers.Optimizer
            Optimizer such as SGD or ADAM.
        state_size : tuple(int)
            Size of an observation from the environment.
        test_freq : int
            Number that specifies after how many training updates a test is
            calculated. Note that we use a train test split.
        shared_value : multiprocessing.Value
            Shared value of the multiprocessing library to stop the training
            process over multi processes.
        train_summary_writer : tf.summary.SummaryWriter
            Summary writer to write tensorboard logs.
        ce_factor : float
            Factor of the cross entropy loss.
        beta : float
            Factor for the L2 regularization.
        check_numerics : boolean
            If True, an exception is thrown in case of NaN values.
        global_norm : float
            Threshold to clip the gradients according to a maximum global norm.
        batch_size : int
            Size of a batch.
        n_batches : int
            Number of batches that is used in the training.
        """
        super().__init__(
            n_cpus=n_cpus,
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
        self.beta = beta
        self.ce_factor = ce_factor
        self.global_norm = global_norm
        self.batch_size = batch_size
        self.state_size = state_size
        agent_process_args = {
            "expert_type": expert_type,
            "expert_args": expert_args,
            "w_gpu_mem": w_gpu_mem
        }

        self.maxlen = n_batches * batch_size
        if self.n_cpus == 1:
            self.expert = expert_type()
        else:
            print("Start initial test")
            t1 = time.time()
            self.test()
            t2 = time.time()
            print("Initial test finished in", str(t2 - t1), " seconds start data sampling")
            self.master_process = MasterProcess(
                n_process=self.n_cpus,
                n_samples=self.maxlen,
                agent_type=policy_type,
                agent_args=policy_args,
                env_type=env_type,
                env_args=env_args,
                model_dir=model_dir,
                model_filename=model_name,
                agent=self.learner,
                shared_value=shared_value,
                agent_process_type=ExpertProcess,
                agent_process_args=agent_process_args,
                async_mode=True)
            self.master_process.update_f = self.train_multi_cpu
        self.train_step = 0
        self.epoch = 0
        self.batch_step = 0
        self.X = np.zeros((tuple([self.maxlen]) + self.state_size))
        self.y = np.zeros((self.maxlen, 1), np.int32)
        self.masks = np.zeros((self.maxlen, 1), np.bool)

    @tf.function
    def update(self, X_batch, y_batch):
        """Updates the neural net. Applies supervised learning.

        Parameters
        ----------
        X_batch : np.ndarray
            An array with the input type of the net.
        y_batch : np.ndarray
            Array with the expert actions.

        Returns
        -------
        tuple(float, float, float)
            The total loss, the global norm of the gradient and the cross
            entropy loss.

        """
        with tf.GradientTape() as tape:
            # action_probs: (N, N_ACTIONS)
            # action_probs, _ = self.learner._action_probs(X_batch)
            features = self.learner.net.compute(X_batch, training=True)
            latent_action = self.learner.latent_action(features)

            act_onehot = tf.one_hot(y_batch, depth=self.n_actions)
            ce = tf.nn.softmax_cross_entropy_with_logits(
                labels=act_onehot, logits=latent_action)
            ce_loss = tf.reduce_mean(ce)
            loss = self.ce_factor * ce_loss
            """
            sum_ = 0
            vars_ = tape.watched_variables()
            for var in vars_:
                sum_ += tf.reduce_sum(tf.square(var))
            loss += self.beta * 0.5 * sum_
            """
            if self.check_numerics:
                loss = tf.debugging.check_numerics(loss, "loss")
            # vars_ = self.learner.get_vars()
            vars_ = tape.watched_variables()
            grads = tape.gradient(loss, vars_)
            global_norm = tf.linalg.global_norm(grads)
            if self.global_norm:
                grads, _ = tf.clip_by_global_norm(
                    grads, self.global_norm, use_norm=global_norm)
            self.optimizer.apply_gradients(zip(grads, vars_))

            return loss, global_norm, ce_loss

    def get_batch_idxs(self):
        """Calculate the batches and the number of batches from the data.

        Returns
        -------
        tuple(np.ndarray, int)
            The indices of the data samples and the number of batches.

        """
        if self.learner.use_lstm:
            batch_indxs = np.where(self.masks == False)[0]
            n_batches = batch_indxs.shape[0]
            batch_indxs = np.vstack(
                (np.array([0], np.int32)[:, None], batch_indxs[:, None]))
            batch_indxs = batch_indxs.reshape((batch_indxs.shape[0], ))
        else:
            n_batches = math.floor(self.maxlen / self.batch_size)
            batch_indxs = np.arange(self.maxlen)
            np.random.shuffle(batch_indxs)
        return batch_indxs, n_batches

    def train_batches(self, ratios):
        """Train the policy.

        Parameters
        ----------
        ratios : np.ndarray
            Ratios between the number of taking expert actions and length of the
            episodes.
        """
        batch_indxs, n_batches = self.get_batch_idxs()
        for b in range(n_batches):
            if self.learner.use_lstm:
                indxs = np.arange(batch_indxs[b], batch_indxs[b+1])
            else:
                b_start = b * self.batch_size
                indxs = batch_indxs[b_start:b_start+self.batch_size]
            X_batch = self.X[indxs]
            y_batch = self.y[indxs]
            loss, global_norm, ce_loss =\
                self.update(X_batch, y_batch)
            with self.train_summary_writer.as_default():
                tf.summary.scalar(
                    "pretrain/loss", loss, step=self.batch_step)
                tf.summary.scalar(
                    "pretrain/global_norm", global_norm, step=self.batch_step)
                tf.summary.scalar(
                    "pretrain/ce_loss", ce_loss, step=self.batch_step)
            self.batch_step += 1
        with self.train_summary_writer.as_default():
            tf.summary.scalar(
                "pretrain/train_rew_steps", np.mean(ratios), step=self.train_step)
            tf.summary.scalar(
                "pretrain/train_std_rew_steps", np.std(ratios), step=self.train_step)
            self.train_step += 1
        self.epoch += 1
        if self.learner.use_lstm:
            sync(self.learner, self.test_policy)
        if self.epoch % self.test_freq == 0:
            self.test()
            if self.test_reward > self.best_test_reward:
                self.best_test_reward = self.test_reward
                self.learner.save(
                    directory=self.model_dir,
                    filename=self.model_name + "_" + str(self.best_test_reward) +  "_Imi")

    def train_single_cpu(self):
        """Applies the training with a single CPU."""
        while self.shared_value:
            state = self.env.reset()
            rewards = []
            for buf_idx in range(self.maxlen):
                expert_action = self.expert(self.env, state)
                action = int(np.random.randint(low=0, high=2))
                state = self.learner.preprocess(state)
                state_, reward, done, info = self.env.step(action)
                self.X[buf_idx] = state
                self.y[buf_idx] = expert_action
                self.masks[buf_idx] = not done
                rewards.append(reward)
                state = state_
                if done:
                    if self.learner.use_lstm:
                        self.test_policy.reset()
                    state = self.env.reset()
            self.train_batches(ratios=np.array(rewards)/self.maxlen)

    def train_multi_cpu(self, transitions):
        """Applies the training with different CPUs."""
        n_samples = 0
        rewards = []
        for trans in transitions.values():
            for i in range(len(trans)):
                pi_action, action, obs, next_obs, reward, done = trans[i]
                self.X[n_samples] = obs
                """
                if n_samples > 10 and n_samples < 20:
                    cv2.imshow("obs 0", obs[0])
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                """
                self.y[n_samples] = pi_action["expert"]
                self.masks[n_samples] = not done
                rewards.append(reward)
                n_samples += 1
                if n_samples == self.maxlen:
                    break
            if n_samples == self.maxlen:
                break
        self.train_batches(ratios=np.array(rewards)/self.maxlen)

    def train(self):
        """Starts the training process."""
        if self.n_cpus == 1:
            self.train_single_cpu()
        else:
            self.master_process.sample_transitions()
