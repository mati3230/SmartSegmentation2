import tensorflow as tf
import tensorflow.keras as k
from .dense_net_k import DenseNet
from cloud_projector import CloudProjector
from optimization.actor_critic import ActorCritic
import numpy as np


class DenseNet_AC(ActorCritic):
    def __init__(
            self,
            name,
            n_ft_outpt,
            n_actions,
            state_size,
            seed=None,
            stddev=0.3,
            check_numerics=False,
            trainable=True,
            stateful=False,
            initializer="he_normal",
            mode="full",
            **kwargs):
        self.stateful = stateful
        super().__init__(
            name=name,
            n_ft_outpt=n_ft_outpt,
            n_actions=n_actions,
            seed=seed,
            stddev=stddev,
            check_numerics=check_numerics,
            trainable=trainable,
            initializer=initializer,
            mode=mode)
        self.use_lstm = False
        self.cloud_projector = CloudProjector(state_size)

    def init_net(
            self,
            name,
            n_ft_outpt,
            seed=None,
            stddev=0.3,
            trainable=True,
            check_numerics=False,
            initializer="he_normal",
            mode="full"):
        self.net = DenseNet(
            name=name,
            outpt=n_ft_outpt,
            trainable=trainable,
            seed=seed,
            stddev=stddev,
            check_numerics=check_numerics,
            stateful=self.stateful,
            initializer=initializer)

    def init_variables(
            self,
            name,
            n_ft_outpt,
            n_actions,
            trainable=True,
            stddev=0.3,
            seed=None,
            initializer="he_normal",
            mode="full"):
        self.a1 = k.layers.Dense(
            units=12,
            name=name+"/d_a_1",
            trainable=trainable,
            activation="relu",
            kernel_initializer=initializer)
        self.a2 = k.layers.Dense(
            units=n_actions,
            name=name+"/d_a_2",
            trainable=trainable,
            activation="linear",
            kernel_initializer=initializer)
        if mode == "half":
            return
        self.v1 = k.layers.Dense(
            units=12,
            name=name+"/d_v_1",
            trainable=trainable,
            activation="relu",
            kernel_initializer=initializer)
        self.v2 = k.layers.Dense(
            units=1,
            name=name+"/d_v_2",
            trainable=trainable,
            activation="linear",
            kernel_initializer=initializer)

    @tf.function
    def latent_action(self, features):
        """Computes a latent representation of the actions (e.g. linear neurons
        before a softmax computation).

        Parameters
        ----------
        features : tf.Tensor
            features from a preprocessing net.

        Returns
        -------
        tf.Tensor
            Latent representation of the actions (e.g. linear neurons before a
            softmax computation).

        """
        net = self.a1(features)
        net = self.a2(net)
        return net

    @tf.function
    def latent_value(self, features):
        """Computes the state value.

        Parameters
        ----------
        features : tf.Tensor
            features from a preprocessing net.

        Returns
        -------
        tf.Tensor
            State value.

        """
        net = self.v1(features)
        net = self.v2(net)
        return net

    def preprocess(self, state):
        return self.cloud_projector.preprocess(state)

    def reset(self):
        if self.mode == "pre":
            return
        self.net.reset()

    def get_head_vars(self):
        vars_ = []
        if self.trainable:
            vars_.extend(self.a1.trainable_weights)
            vars_.extend(self.a2.trainable_weights)
            if self.mode == "full":
                vars_.extend(self.v1.trainable_weights)
                vars_.extend(self.v2.trainable_weights)
        else:
            vars_.extend(self.a1.non_trainable_weights)
            vars_.extend(self.a2.non_trainable_weights)
            if self.mode == "full":
                vars_.extend(self.v1.non_trainable_weights)
                vars_.extend(self.v2.non_trainable_weights)
        return vars_

    def get_vars(self):
        vars_ = self.net.get_vars()
        vars_.extend(self.get_head_vars())
        return vars_


if __name__ == "__main__":
    policy = DenseNet_AC(
        name="mvcnn_lstm_ac",
        n_ft_outpt=64,
        n_actions=2,
        state_size=(4, 64, 64, 3),
        trainable=False)
    V = 4
    W = H = 64
    C = 3
    data = np.zeros((V, W, H, C))
    action = policy.action(data, training=False)
    v = policy.get_vars()
    # print(v)
    for k_var in v:
        print(k_var.name)
