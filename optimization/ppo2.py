import tensorflow as tf
import numpy as np
from .tf_utils import kl_div
from .utils import normalize
import math


class PPO2():
    """This class implements methods to optimize a policy with the PPO
    algorithm.

    Parameters
    ----------
    policy_type : type
        Type of the policy that should be trained to initialize the target
        policy.
    policy_args : dictionary
        Arguments that are used to initialize instances of the policy.
    n_actions : int
        Number of available actions.
    optimizer : tf.keras.optimizers.Optimizer
        Optimizer such as SGD or ADAM.
    train_summary_writer : tf.summary.SummaryWriter
        Summary writer to write tensorboard logs.
    batch_size : int
        Number of samples in one batch.
    global_norm : float
        Threshold to clip the gradients according to a maximum global norm.
    check_numerics : boolean
        If True, an exception is thrown in case of NaN values.
    gamma : float
        The discount factor gamma should be between [0, 1].
    K_epochs : int
        Number of epochs.
    eps_clip : float
        Cliprange epsilon to clip the ratio of the new and old policy in a
        ppo update.
    lmbda : float
        Factor of the generalized advantage estimation that should be
        between [0, 1].
    entropy_factor : float
        Factor to weight the importance of the entropy in the loss
        function. The higher this factor, the more exploration will be
        encouraged.
    value_factor : float
        Factor to weight the importance of the state value estimation.
    write_tensorboard : boolean
        If True, summary will be written.
    normalize_returns : boolean
        If True, the expected returns will be normalized in ppo update.
    normalize_advantages : boolean
        If True, the advantages will be normalized in ppo update.

    Attributes
    ----------
    gamma : float
        The discount factor gamma should be between [0, 1].
    n_actions : int
        Number of available actions.
    optimizer : tf.keras.optimizers.Optimizer
        Optimizer such as SGD or ADAM.
    train_summary_writer : tf.summary.SummaryWriter
        Summary writer to write tensorboard logs.
    batch_size : int
        Number of samples in one batch.
    global_norm : float
        Threshold to clip the gradients according to a maximum global norm.
    check_numerics : boolean
        If True, an exception is thrown in case of NaN values.
    K_epochs : int
        Number of epochs.
    eps_clip : float
        Cliprange epsilon to clip the ratio of the new and old policy in a
        ppo update.
    lmbda : float
        Factor of the generalized advantage estimation that should be
        between [0, 1].
    entropy_factor : float
        Factor to weight the importance of the entropy in the loss
        function. The higher this factor, the more exploration will be
        encouraged.
    value_factor : float
        Factor to weight the importance of the state value estimation.
    write_tensorboard : boolean
        If True, summary will be written.
    normalize_returns : boolean
        If True, the expected returns will be normalized in ppo update.
    normalize_advantages : boolean
        If True, the advantages will be normalized in ppo update.
    target_policy : BasePolicy
        Policy, that will be optimized.

    """
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
            write_tensorboard=False,
            normalize_returns=True,
            normalize_advantages=True,
            **kwargs):
        """Constructor. Initializes the target policy.

        Parameters
        ----------
        policy_type : type
            Type of the policy that should be trained to initialize the target
            policy.
        policy_args : dictionary
            Arguments that are used to initialize instances of the policy.
        n_actions : int
            Number of available actions.
        optimizer : tf.keras.optimizers.Optimizer
            Optimizer such as SGD or ADAM.
        train_summary_writer : tf.summary.SummaryWriter
            Summary writer to write tensorboard logs.
        batch_size : int
            Number of samples in one batch.
        global_norm : float
            Threshold to clip the gradients according to a maximum global norm.
        check_numerics : boolean
            If True, an exception is thrown in case of NaN values.
        gamma : float
            The discount factor gamma should be between [0, 1].
        K_epochs : int
            Number of epochs.
        eps_clip : float
            Cliprange epsilon to clip the ratio of the new and old policy in a
            ppo update.
        lmbda : float
            Factor of the generalized advantage estimation that should be
            between [0, 1].
        entropy_factor : float
            Factor to weight the importance of the entropy in the loss
            function. The higher this factor, the more exploration will be
            encouraged.
        value_factor : float
            Factor to weight the importance of the state value estimation.
        write_tensorboard : boolean
            If True, summary will be written.
        normalize_returns : boolean
            If True, the expected returns will be normalized in ppo update.
        normalize_advantages : boolean
            If True, the advantages will be normalized in ppo update.
        """
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
        self.write_tensorboard = write_tensorboard
        self.normalize_returns = normalize_returns
        self.normalize_advantages = normalize_advantages

        # initialize the target policy
        pa = policy_args.copy()
        pa["trainable"] = True
        pa["stateful"] = False
        print("target_policy")
        self.target_policy = policy_type(**pa)

        # print a summary
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
            Total loss.
        tf.Tensor
            Global norm of the gradients.
        tf.Tensor
            Kullback leibler divergence of the actions taken.

        """
        with tf.GradientTape() as tape:
            if self.check_numerics:
                action_probs = tf.debugging.check_numerics(
                    action_probs, "action_probs")
            # evaluate the actions
            logprobs_, state_values, dist_entropy, action_probs_ =\
                self.target_policy.evaluate(states, actions)
            if self.check_numerics:
                action_probs_ = tf.debugging.check_numerics(
                    action_probs_, "action_probs_")

            # compute the kullback leibler divergence
            kl_divergence = kl_div(action_probs, action_probs_)
            if self.check_numerics:
                kl_divergence = tf.debugging.check_numerics(
                    kl_divergence, "kl_divergence")

            # compute the ratio (pi_theta / pi_theta__old)
            logprobs = tf.reshape(logprobs, [tf.shape(logprobs)[0]])
            ratios = tf.math.exp(logprobs_ - tf.stop_gradient(logprobs))
            if self.check_numerics:
                ratios = tf.debugging.check_numerics(ratios, "ratios")

            # compute surrogate loss
            surr1 = ratios * advantages
            surr2 = tf.clip_by_value(
                ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            surr_loss = tf.math.minimum(surr1, surr2)

            critic_loss = tf.math.square(returns - state_values)
            critic_loss = 0.5*tf.reduce_mean(critic_loss)

            surr_loss = tf.reduce_mean(surr_loss)
            dist_entropy = tf.reduce_mean(dist_entropy)
            kl_divergence = tf.reduce_mean(kl_divergence)

            # compute the policy loss
            pi_loss = -(surr_loss + (self.entropy_factor*dist_entropy))

            # gradient update
            vars_ = tape.watched_variables()
            # compute the total loss
            loss = pi_loss + (self.value_factor*critic_loss)
            global_norm = 0
            # compute the gradients
            grads = tape.gradient(loss, vars_)
            # norm of the gradients
            global_norm = tf.linalg.global_norm(grads)

            if self.global_norm:
                grads, _ = tf.clip_by_global_norm(
                    grads,
                    self.global_norm,
                    use_norm=global_norm)
            # apply gradient update
            self.optimizer.apply_gradients(zip(grads, vars_))
            return \
                surr_loss,\
                critic_loss,\
                dist_entropy,\
                loss,\
                global_norm,\
                kl_divergence

    def update(self, transitions, step):
        """Performes a PPO update.

        Parameters
        ----------
        transitions : dictionary
            The keys of this dictionary are process ids. If only a single
            process is used, only one element is in the dictionary.
            The values of the dictionary are lists. A single list contains
            tuples that are actually the transitions.
            A tuple consist of:
                * dictionary that contains informartion from the agent,
                * the agent action,
                * the observation according to which the agent selects its
                  action,
                * the next observation of the environment,
                * the reward value and
                * a flag if the episode is over.
        step : int
            Global step number of the ppo update. It is used to log values in
            tensorboard.
        """
        # create some lists from the transitions for the ppo calculation
        # the expected returns
        returns = []
        advantages = []
        actions = []
        # list that contains all observations
        all_obs = []
        # list of all log-probabilities
        logprobs = []
        # list of all action-probabilities
        action_probs = []
        dones = []
        for trans in transitions.values():
            # generalized advantage estimation
            gae = 0
            # previous expected return
            prev_return = 0
            # previous state value
            prev_value = 0
            for i in reversed(range(len(trans))):
                # unpack the i-th transition
                pi_action, action, obs, next_obs, reward, done = trans[i]

                # flip the done values to use the mask as factor
                mask = 1 - done
                prev_return = reward + (self.gamma * prev_return * mask)
                returns.insert(0, prev_return)

                # calculate the generalized advantage estimation
                value = pi_action["state_value"]
                delta = reward + (self.gamma * prev_value * mask) - value
                prev_value = value
                gae = delta + (self.gamma * self.lmbda * mask * gae)
                advantages.insert(0, gae)

                # fill the remaining lists (see lists above)
                logprob = pi_action["logprob"]
                logprobs.insert(0, logprob)
                action_prob = pi_action["action_probs"]
                action_probs.insert(0, action_prob)
                actions.insert(0, action)
                all_obs.insert(0, obs)
                dones.insert(0, done)

        # calculate the batches
        # if using an lstm - store the end indices of each episode
        if self.target_policy.use_lstm:
            dones = np.array(dones, np.bool)
            # end of the episodes
            batch_indxs = np.where(dones == True)[0]
            # number of batches
            n_batches = batch_indxs.shape[0]
            # append the first index
            batch_indxs = np.vstack(
                (np.array([0], np.int32)[:, None], batch_indxs[:, None]))
            batch_indxs = batch_indxs.reshape((batch_indxs.shape[0], ))
        else: # if using non sequential net
            n_samples = len(returns)
            # number of batches
            n_batches = math.floor(n_samples / self.batch_size)
            batch_indxs = np.arange(n_samples)
            np.random.shuffle(batch_indxs)

        # define some values that will be logged in tensorboard
        # surrogate loss
        surr_loss = 0
        critic_loss = 0
        # entropy of the action distribution
        dist_entropy = 0
        # total loss value
        loss = 0
        # global norm of the gradient
        global_norm = 0
        # kullback leibler divergence
        kl_divergence = 0
        # the above values will be divided by div
        div = self.K_epochs * n_batches

        # convert to numpy arrays
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

        # update loop
        for _ in range(self.K_epochs):
            for i in range(n_batches):
                # calculate the indexes of a batch
                if self.target_policy.use_lstm:
                    # we have to take care of the sequence
                    indxs = np.arange(batch_indxs[i], batch_indxs[i+1])
                else:
                    j = i * self.batch_size
                    indxs = batch_indxs[j:j+self.batch_size]

                # conduct the ppo update
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

                # update the values to log in tensorboard
                surr_loss += (surr_loss_ / div)
                critic_loss += (critic_loss_ / div)
                dist_entropy += (dist_entropy_ / div)
                loss += (loss_ / div)
                global_norm += (global_norm_ / div)
                kl_divergence += (kl_divergence_ / div)

        # write tensorboard logs
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
