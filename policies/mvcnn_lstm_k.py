import tensorflow.keras as k
import tensorflow as tf
import numpy as np
from optimization.base_net import BaseNet


class MVCNN_LSTM(BaseNet):
    def __init__(
            self,
            name,
            outpt,
            trainable=True,
            seed=None,
            check_numerics=False,
            stddev=0.3,
            stateful=False,
            initializer="glorot_uniform"):
        super().__init__(
            name=name,
            outpt=outpt,
            trainable=trainable,
            seed=seed)
        self.trainable = trainable
        self.conv1 = k.layers.Conv2D(
            filters=16,
            kernel_size=(3, 3),
            strides=(2, 2),
            activation="relu",
            name=name+"/c1",
            trainable=trainable,
            kernel_initializer=initializer)
        self.conv2 = k.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(2, 2),
            activation="relu",
            name=name+"/c2",
            trainable=trainable,
            kernel_initializer=initializer)
        self.conv3 = k.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation="relu",
            name=name+"/c3",
            trainable=trainable,
            kernel_initializer=initializer)
        self.lstm = tf.keras.layers.LSTM(
            units=outpt,
            trainable=trainable,
            return_state=False,
            stateful=stateful,
            name=name+"/lstm",
            kernel_initializer=initializer)
        self.mp1 = k.layers.MaxPool2D(
            strides=(2, 2),
            name=name+"/mp1")
        self.mp2 = k.layers.MaxPool2D(
            strides=(2, 2),
            name=name+"/mp2")
        self.flatten = k.layers.Flatten(
            name=name+"/flatten")

    @tf.function
    def compute(self, obs, training=False):
        # B X V X W X H X C
        obs = tf.dtypes.cast(obs, tf.float32)
        features = []
        for i in range(obs.shape[1]):
            net = obs[:, i]
            net = self.conv1(net)
            net = self.mp1(net)
            net = self.conv2(net)
            net = self.mp2(net)
            net = self.conv3(net)
            net = self.flatten(net)
            net = tf.expand_dims(net, axis=1)
            features.append(net)
        vp = tf.concat(features, axis=1)
        net = tf.math.reduce_max(vp, axis=1, keepdims=True)
        net = self.lstm(net)
        return net

    def reset(self):
        if self.lstm.stateful:
            self.lstm.reset_states()

    def get_vars(self):
        vars_ = []
        if self.trainable:
            vars_.extend(self.conv1.trainable_weights)
            vars_.extend(self.conv2.trainable_weights)
            vars_.extend(self.conv3.trainable_weights)
            vars_.extend(self.lstm.trainable_weights)
        else:
            vars_.extend(self.conv1.non_trainable_weights)
            vars_.extend(self.conv2.non_trainable_weights)
            vars_.extend(self.conv3.non_trainable_weights)
            vars_.extend(self.lstm.non_trainable_weights)
        return vars_


if __name__ == "__main__":
    model = MVCNN_LSTM(
        name="mvcnn_lstm",
        outpt=32
    )
    B = 2
    V = 2
    W = H = 64
    C = 3
    data = np.zeros((B, V, W, H, C))
    features = model.compute(data)
    # print(features)
    v = model.get_vars()
    print(v)
