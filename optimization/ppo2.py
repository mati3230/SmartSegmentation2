import tensorflow as tf
import numpy as np
from .tf_utils import kl_div
from .utils import normalize
import math


class PPO2():
    def __init__(
            self,
            policy_type,
            policy_args,
            n_actions,
            optimizer,
            train_summary_writer,
            batch_size=64,
            global_norm=0.5,
            check_numerics=False,
            gamma=0.99,
            K_epochs=3,
            eps_clip=0.2,
            lmbda=0.95,
            entropy_factor=0.01,
            value_factor=0.5,
            beta=0,
            write_tensorboard=False,
            normalize_returns=True,
            normalize_advantages=True,
            **kwargs):
        self.gamma = gamma
        self.n_actions = n_actions
        self.optimizer = optimizer
        self.train_summary_writer = train_summary_writer
        self.batch_size = batch_size
        self.global_norm = global_norm
        self.check_numerics = check_numerics
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        self.lmbda = lmbda
        self.entropy_factor = entropy_factor
        self.value_factor = value_factor
        self.beta = beta
        pa = policy_args.copy()
        pa["trainable"] = True
        pa["stateful"] = False
        print("target_policy")
        self.target_policy = policy_type(**pa)
        self.write_tensorboard = write_tensorboard
        self.normalize_returns = normalize_returns
        self.normalize_advantages = normalize_advantages
        print("-----------------------------------")
        print("PPO2: ", locals())
        print("-----------------------------------")

    @tf.function
    def ppo_update(
            self,
            returns,
            advantages,
            states,
            actions,
            logprobs,
            action_probs):
        """Updates the policy according to the PPO algorithm.

        Parameters
        ----------
        returns : np.ndarray
            Discounted rewards.
        advantages : np.ndarray
            Advantages.
        states : np.ndarray
            Observations.
        actions : np.ndarray
            Actions that were taken in context of certain observations.
        logprobs : np.ndarray
            Log probabilities of the actions taken.
        action_probs : np.ndarray
            Probabilities of the actions taken.

        Returns
        -------
        tf.Tensor
            Surrogate loss.
        tf.Tensor
            Critic loss of estimating the returns.
        tf.Tensor
            Entropy of the action selection.
        tf.Tensor
            Overall loss.
        tf.Tensor
            Global norm of the gradients.
        tf.Tensor
            Kullback leibler divergence of the actions taken.

        """
        with tf.GradientTape() as tape:
            # advantages: (B, )
            if self.check_numerics:
                action_probs = tf.debugging.check_numerics(
                    action_probs, "action_probs")
            # evaluating actions
            logprobs_, state_values, dist_entropy, action_probs_ =\
                self.target_policy.evaluate(states, actions)
            if self.check_numerics:
                # tf.print(action_probs_, summarize=100)
                action_probs_ = tf.debugging.check_numerics(
                    action_probs_, "action_probs_")
            # logprobs: (B, )
            logprobs = tf.reshape(logprobs, [tf.shape(logprobs)[0]])
            # tf.print(tf.shape(logprobs))
            kl_divergence = kl_div(action_probs, action_probs_)
            if self.check_numerics:
                kl_divergence = tf.debugging.check_numerics(
                    kl_divergence, "kl_divergence")
            kl_divergence = tf.reduce_mean(kl_divergence)

            # finding the ratio (pi_theta / pi_theta__old)
            # ratios: (B, )
            ratios = tf.math.exp(logprobs_ - tf.stop_gradient(logprobs))
            if self.check_numerics:
                ratios = tf.debugging.check_numerics(ratios, "ratios")
            # finding surrogate loss
            # surr1: (B, )
            surr1 = ratios * advantages
            # surr2: (B, )
            surr2 = tf.clip_by_value(
                ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            # surr_loss: (B, )
            surr_loss = tf.math.minimum(surr1, surr2)
            # critic_loss: (B, )
            critic_loss = tf.math.square(returns - state_values)
            critic_loss = 0.5*tf.reduce_mean(critic_loss)
            surr_loss = tf.reduce_mean(surr_loss)
            dist_entropy = tf.reduce_mean(dist_entropy)
            pi_loss = -(surr_loss + (self.entropy_factor*dist_entropy))
            # vars_ = self.target_policy.get_vars()
            vars_ = tape.watched_variables()
            loss = pi_loss + (self.value_factor*critic_loss)
            # tf.print(loss)
            global_norm = 0
            # """
            # vars_ = tape.watched_variables()
            grads = tape.gradient(loss, vars_)
            global_norm = tf.linalg.global_norm(grads)
            # for i in range(len(grads)):
            #    grads[i] = tf.clip_by_value(grads[i], -1e12, 1e12)

            if self.global_norm:
                grads, _ = tf.clip_by_global_norm(
                    grads,
                    self.global_norm,
                    use_norm=global_norm)
            self.optimizer.apply_gradients(zip(grads, vars_))
            return \
                surr_loss,\
                critic_loss,\
                dist_entropy,\
                loss,\
                global_norm,\
                kl_divergence

    def update(self, transitions, step):
        returns = []
        advantages = []
        actions = []
        all_obs = []
        logprobs = []
        action_probs = []
        # rewards = []
        dones = []
        for trans in transitions.values():
            gae = 0
            prev_return = 0
            prev_value = 0
            for i in reversed(range(len(trans))):
                pi_action, action, obs, next_obs, reward, done = trans[i]
                mask = 1 - done
                prev_return = reward + (self.gamma * prev_return * mask)
                returns.insert(0, prev_return)
                value = pi_action["state_value"]
                delta = reward + (self.gamma * prev_value * mask) - value
                prev_value = value
                gae = delta + (self.gamma * self.lmbda * mask * gae)
                advantages.insert(0, gae)
                logprob = pi_action["logprob"]
                logprobs.insert(0, logprob)
                action_prob = pi_action["action_probs"]
                action_probs.insert(0, action_prob)
                actions.insert(0, action)
                all_obs.insert(0, obs)
                dones.insert(0, done)

        surr_loss = 0
        critic_loss = 0
        dist_entropy = 0
        loss = 0
        global_norm = 0
        kl_divergence = 0
        if self.target_policy.use_lstm:
            dones = np.array(dones, np.bool)
            batch_indxs = np.where(dones == True)[0]
            n_batches = batch_indxs.shape[0]
            batch_indxs = np.vstack(
                (np.array([0], np.int32)[:, None], batch_indxs[:, None]))
            batch_indxs = batch_indxs.reshape((batch_indxs.shape[0], ))
        else:
            n_samples = len(returns)
            n_batches = math.floor(n_samples / self.batch_size)
            batch_indxs = np.arange(n_samples)
            np.random.shuffle(batch_indxs)

        div = self.K_epochs * n_batches
        returns = np.array(returns, np.float32)
        if self.normalize_returns:
            returns = normalize(returns)
        advantages = np.array(advantages, np.float32)
        if self.normalize_advantages:
            advantages = normalize(advantages)
        all_obs = np.array(all_obs, np.float32)
        actions = np.array(actions, np.int32)
        logprobs = np.array(logprobs, np.float32)
        action_probs = np.array(action_probs, np.float32)
        for _ in range(self.K_epochs):
            for i in range(n_batches):
                if self.target_policy.use_lstm:
                    indxs = np.arange(batch_indxs[i], batch_indxs[i+1])
                else:
                    j = i * self.batch_size
                    indxs = batch_indxs[j:j+self.batch_size]
                surr_loss_,\
                    critic_loss_,\
                    dist_entropy_,\
                    loss_, global_norm_,\
                    kl_divergence_ = self.ppo_update(
                        returns[indxs],
                        advantages[indxs],
                        all_obs[indxs],
                        actions[indxs],
                        logprobs[indxs],
                        action_probs[indxs])
                # self.target_policy.reset()
                surr_loss += (surr_loss_ / div)
                critic_loss += (critic_loss_ / div)
                dist_entropy += (dist_entropy_ / div)
                loss += (loss_ / div)
                global_norm += (global_norm_ / div)
                kl_divergence += (kl_divergence_ / div)
        # print("train done")
        if self.write_tensorboard:
            with self.train_summary_writer.as_default():
                tf.summary.scalar("train/loss", loss, step=step)
                tf.summary.scalar("train/surrogate_loss", surr_loss, step=step)
                tf.summary.scalar("train/critic_loss", critic_loss, step=step)
                tf.summary.scalar(
                    "train/dist_entropy", dist_entropy, step=step)
                if self.global_norm is not None:
                    tf.summary.scalar(
                        "train/global_norm", global_norm, step=step)
                tf.summary.scalar(
                    "train/kl_divergence", kl_divergence, step=step)
                tf.summary.scalar(
                    "train/returns", np.mean(returns), step=step)
            self.train_summary_writer.flush()
