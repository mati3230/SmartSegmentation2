from abc import abstractmethod

import tensorflow as tf
import tensorflow_probability as tfp

from .base_policy import BasePolicy


class ActorCritic(BasePolicy):
    def __init__(
            self,
            name,
            n_ft_outpt,
            n_actions,
            seed=None,
            stddev=0.3,
            trainable=True,
            check_numerics=False,
            initializer="glorot_uniform",
            mode="full"):
        """Constructor.

        Parameters
        ----------
        name : str
            Name of the neural net.
        n_ft_outpt : int
            Number of features that should be calculated by the feature
            detector.
        n_actions : int
            Number of actions that are valid in the environment.
        seed : int
            Random seed that should be used.
        stddev : float
            (Deprecated) Standard deviation of a gaussian with zero mean to
            initialize the neurons.
        trainable : boolean
            If True the value of the neurons can be changed.
        check_numerics : boolean
            If True numeric values will be checked in tensorflow calculation to
            detect, e.g., NaN values.
        initializer : str
            Keras initializer that will be used (e.g. orthogonal).
        mode : str
            Full or Half. If Half, then only the action without the value will
            be calculated.
        """
        super().__init__(
            name,
            n_ft_outpt,
            n_actions,
            seed=seed,
            stddev=stddev,
            trainable=trainable,
            check_numerics=check_numerics,
            initializer=initializer,
            mode=mode)

    @tf.function(experimental_relax_shapes=True)
    def _action_probs(self, obs, training=True):
        """Computes a probability distribution over the actions by a softmax
        layer given the actions.

        Parameters
        ----------
        obs : tf.Tensor
            Observation.

        Returns
        -------
        tf.Tensor
            Probability distribution over the actions.
        tf.Tensor
            Latent representation of the observation.

        """
        features = self.net.compute(obs, training=training)
        if self.check_numerics:
            features = tf.debugging.check_numerics(features, "features")
        latent_action = self.latent_action(features)
        if self.check_numerics:
            latent_action = tf.debugging.check_numerics(
                latent_action, "latent_action")
        action_probs = tf.nn.softmax(logits=latent_action)
        if self.check_numerics:
            action_probs = tf.debugging.check_numerics(
                action_probs, "action_probs")
        return action_probs, features

    @abstractmethod
    def preprocess(self, state):
        pass

    @tf.function
    def action(self, obs, training=True):
        """Action selection of the actor critic model.

        Parameters
        ----------
        obs : np.ndarray
            Observation.
        training : bool
            Switch between deterministic (False) and stochastic (True) action
            selection.

        Returns
        -------
        dict
            The dictionary contains the chosen action, the log probabilities
            over the actions, the value of an observation and the probabilities
            over the actions. They can be accessed with 'action', 'logprob',
            'obs_value' and 'action_probs'.

        """
        obs = tf.expand_dims(obs, axis=0)
        # action_probs: (1, N_ACTIONS)
        action_probs, features = self._action_probs(obs, training=training)
        dist = tfp.distributions.Categorical(probs=action_probs)
        # action: (1, )
        if training:
            action = dist.sample(seed=self.seed)
        else:
            action = tf.math.argmax(
                action_probs,
                axis=1,
                output_type=tf.dtypes.int32)
        # log_prob: (1, )
        log_prob = dist.log_prob(action)

        # action_probs: (N_ACTIONS, )
        action_probs = tf.reshape(action_probs, [tf.shape(action_probs)[1]])
        if self.mode == "half":
            return {"action": action,
                    "logprob": log_prob,
                    "action_probs": action_probs}
        # obs_value: (1, 1)
        obs_value = self.latent_value(features)
        if self.check_numerics:
            features = tf.debugging.check_numerics(features, "features")
            log_prob = tf.debugging.check_numerics(log_prob, "log_prob")
            obs_value = tf.debugging.check_numerics(
                obs_value, "obs_value")
            action_probs = tf.debugging.check_numerics(
                action_probs, "action_probs")

        # obs_value: ()
        obs_value = tf.squeeze(obs_value)
        return {"action": action,
                "logprob": log_prob,
                "state_value": obs_value,
                "action_probs": action_probs}

    @tf.function
    def evaluate(self, obs, action_inpt):
        """Evaluates the an action taken.

        Parameters
        ----------
        obs : tf.Tensor
            Observation.
        action_inpt : np.ndarray
            Action that was taken in context of a certain obersvation.

        Returns
        -------
        tf.Tensor
            Log propabilities of the action.
        tf.Tensor
            Value of the state.
        tf.Tensor
            Entropy of the categorical distribution (discrete action space).
        tf.Tensor
            Probabilities of the actions.

        """
        action_inpt = tf.reshape(action_inpt, [tf.shape(action_inpt)[0]])
        if self.check_numerics:
            action_inpt = tf.debugging.check_numerics(
                action_inpt, "action_inpt")
            obs = tf.debugging.check_numerics(obs, "obs")
        # features: (B, N_FEATURES)
        # action_probs: (B, N_ACTIONS)
        # action_inpt: (B,)
        action_probs, features = self._action_probs(obs, training=True)
        if self.check_numerics:
            action_probs = tf.debugging.check_numerics(
                action_probs, "action_probs")
            features = tf.debugging.check_numerics(features, "features")

        dist = tfp.distributions.Categorical(probs=action_probs)
        # logprobs: (B, )
        logprobs = dist.log_prob(action_inpt)
        # dist_entropy: (B, )
        dist_entropy = dist.entropy()
        # state_value: (B, 1)
        state_value = self.latent_value(features)
        # state_value: (B, )
        state_value = tf.squeeze(state_value)

        return logprobs, state_value, dist_entropy, action_probs

    @abstractmethod
    def init_variables(
            self,
            name,
            n_ft_outpt,
            n_actions,
            stddev=0.3,
            trainable=True,
            seed=None,
            initializer="glorot_uniform",
            mode="full"):
        pass

    @abstractmethod
    def init_net(
            self,
            name,
            n_ft_outpt,
            seed=None,
            stddev=0.3,
            trainable=True,
            check_numerics=False,
            initializer="glorot_uniform",
            mode="full"):
        pass

    @abstractmethod
    def latent_action(self, features):
        pass

    @abstractmethod
    def latent_value(self, features):
        pass

    @abstractmethod
    def get_vars(self):
        pass

    @abstractmethod
    def reset(self):
        pass
