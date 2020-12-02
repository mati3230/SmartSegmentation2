import tensorflow.keras as k
import tensorflow as tf
import numpy as np
from optimization.base_net import BaseNet


class MLP(BaseNet):
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
        self.d = k.layers.Dense(
            units=outpt,
            name=name+"/d_f_1",
            trainable=trainable,
            activation="relu",
            kernel_initializer=initializer)

    @tf.function
    def compute(self, obs, training=False):
        net = self.d(obs)
        return net

    def reset(self):
        pass

    def get_vars(self):
        vars_ = []
        if self.trainable:
            vars_.extend(self.d.trainable_weights)
        else:
            vars_.extend(self.d.non_trainable_weights)
        return vars_


if __name__ == "__main__":
    model = MLP(
        name="mlp",
        outpt=4
    )
    B = 2
    W = 4
    data = np.zeros((B, W))
    features = model.compute(data)
    # print(features)
    v = model.get_vars()
    # print(v)
