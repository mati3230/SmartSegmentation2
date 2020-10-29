import tensorflow.keras as k
import tensorflow as tf
import numpy as np
from optimization.base_net import BaseNet
from resnet_blocks import IdentityBlock, ConvBlock


class ResNet(BaseNet):
    def __init__(
            self,
            name,
            outpt,
            trainable=True,
            seed=None,
            check_numerics=False,
            stddev=0.3,
            stateful=False,
            initializer="he_normal"):
        super().__init__(
            name=name,
            outpt=outpt,
            trainable=trainable,
            seed=seed)
        self.trainable = trainable
        self.conv1 = k.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(2, 2),
            activation=None,
            name=name+"/c1",
            trainable=trainable,
            kernel_initializer=initializer,
            padding="same",
            use_bias=False)
        self.bn1 = k.layers.BatchNormalization(
            name=name+"/bn1",
            trainable=trainable)
        self.act1 = tf.keras.layers.Activation("relu")
        self.mp1 = k.layers.MaxPool2D(
            strides=(2, 2),
            name=name+"/mp1")
        self.ib1_1 = IdentityBlock(
            filters=[64, 64, 256],
            kernel_size=(3, 3),
            name=name + "/ib1_1",
            trainable=trainable,
            kernel_initializer=initializer)
        self.ib1_2 = IdentityBlock(
            filters=[64, 64, 256],
            kernel_size=(3, 3),
            name=name + "/ib1_2",
            trainable=trainable,
            kernel_initializer=initializer)
        self.ib1_3 = IdentityBlock(
            filters=[64, 64, 256],
            kernel_size=(3, 3),
            name=name + "/ib1_3",
            trainable=trainable,
            kernel_initializer=initializer)

        self.cb2 = ConvBlock(
            filters=[128, 128, 512],
            kernel_size=(3, 3),
            name=name + "/cb2",
            trainable=trainable,
            kernel_initializer=initializer)
        self.ib2_1 = IdentityBlock(
            filters=[128, 128, 512],
            kernel_size=(3, 3),
            name=name + "/ib2_1",
            trainable=trainable,
            kernel_initializer=initializer)
        self.ib2_2 = IdentityBlock(
            filters=[128, 128, 512],
            kernel_size=(3, 3),
            name=name + "/ib2_2",
            trainable=trainable,
            kernel_initializer=initializer)

        self.cb3 = ConvBlock(
            filters=[256, 256, 1024],
            kernel_size=(3, 3),
            name=name + "/cb3",
            trainable=trainable,
            kernel_initializer=initializer)
        self.ib3_1 = IdentityBlock(
            filters=[256, 256, 1024],
            kernel_size=(3, 3),
            name=name + "/ib3_1",
            trainable=trainable,
            kernel_initializer=initializer)
        self.ib3_2 = IdentityBlock(
            filters=[256, 256, 1024],
            kernel_size=(3, 3),
            name=name + "/ib3_2",
            trainable=trainable,
            kernel_initializer=initializer)
        self.ib3_3 = IdentityBlock(
            filters=[256, 256, 1024],
            kernel_size=(3, 3),
            name=name + "/ib3_1",
            trainable=trainable,
            kernel_initializer=initializer)
        self.ib3_4 = IdentityBlock(
            filters=[256, 256, 1024],
            kernel_size=(3, 3),
            name=name + "/ib3_2",
            trainable=trainable,
            kernel_initializer=initializer)
        self.ib3_5 = IdentityBlock(
            filters=[256, 256, 1024],
            kernel_size=(3, 3),
            name=name + "/ib3_2",
            trainable=trainable,
            kernel_initializer=initializer)

        self.cb4 = ConvBlock(
            filters=[512, 512, 2048],
            kernel_size=(3, 3),
            name=name + "/cb4",
            trainable=trainable,
            kernel_initializer=initializer)
        self.ib4_1 = IdentityBlock(
            filters=[512, 512, 2048],
            kernel_size=(3, 3),
            name=name + "/ib4_1",
            trainable=trainable,
            kernel_initializer=initializer)
        self.ib4_2 = IdentityBlock(
            filters=[512, 512, 2048],
            kernel_size=(3, 3),
            name=name + "/ib4_2",
            trainable=trainable,
            kernel_initializer=initializer)
        self.ap1 = k.layers.GlobalAveragePooling2D(name=name+"/gap1")
        self.flatten = k.layers.Flatten(name=name+"/flatten")

    @tf.function
    def compute(self, obs, training=False):
        """Computes a latent representation from images of multiple views of a
        point cloud.

        Parameters
        ----------
        obs : tf.Tensor
            Multiple views of a rendered point cloud.
        training : bool
            Switch for batch norm.

        Returns
        -------
        tf.Tensor
            Latent representation of the images.

        """
        # B X V X W X H X C
        obs = tf.dtypes.cast(obs, tf.float32)
        # print(len(tf.shape(obs)))
        obs_shape = tf.shape(obs)
        if len(obs_shape) == 4:
            obs = tf.expand_dims(obs, axis=0)
        n_iter = obs_shape[1]
        features = tf.TensorArray(tf.float32, size=n_iter)
        for i in range(n_iter):
            # print(i)
            net = obs[:, i]
            # print(net.shape)
            net = self.conv1(net)
            net = self.bn1(net, training=training)
            net = self.act1(net)
            net = self.mp1(net)
            # print(net.shape)

            net = self.ib1_1(net, training=training)
            net = self.ib1_2(net, training=training)
            net = self.ib1_3(net, training=training)
            # print(net.shape)

            net = self.cb2(net, training=training)
            net = self.ib2_1(net, training=training)
            net = self.ib2_2(net, training=training)
            # print(net.shape)

            net = self.cb3(net, training=training)
            net = self.ib3_1(net, training=training)
            net = self.ib3_2(net, training=training)
            net = self.ib3_3(net, training=training)
            net = self.ib3_4(net, training=training)
            net = self.ib3_5(net, training=training)
            # print(net.shape)

            net = self.cb4(net, training=training)
            net = self.ib4_1(net, training=training)
            net = self.ib4_2(net, training=training)

            net = self.ap1(net)
            # print(net.shape)
            net = self.flatten(net)
            # print("Done:", net.shape)
            features = features.write(i, net)
        # vp = tf.concat(features, axis=0)
        vp = features.stack()
        # print("vp:", vp.shape)
        net = tf.math.reduce_max(vp, axis=0, keepdims=True)
        # print(net.shape)
        net = tf.squeeze(net, axis=0)
        # print(net.shape)
        return net

    def reset(self):
        pass

    def get_vars(self):
        vars_ = []
        if self.trainable:
            vars_.extend(self.conv1.trainable_weights)
            vars_.extend(self.bn1.trainable_weights)

        else:
            vars_.extend(self.conv1.non_trainable_weights)
            vars_.extend(self.bn1.non_trainable_weights)
        vars_.extend(self.ib1_1.get_vars())
        vars_.extend(self.ib1_2.get_vars())
        vars_.extend(self.ib1_3.get_vars())

        vars_.extend(self.cb2.get_vars())
        vars_.extend(self.ib2_1.get_vars())
        vars_.extend(self.ib2_2.get_vars())

        vars_.extend(self.cb3.get_vars())
        vars_.extend(self.ib3_1.get_vars())
        vars_.extend(self.ib3_2.get_vars())
        vars_.extend(self.ib3_3.get_vars())
        vars_.extend(self.ib3_4.get_vars())
        vars_.extend(self.ib3_5.get_vars())

        vars_.extend(self.cb4.get_vars())
        vars_.extend(self.ib4_1.get_vars())
        vars_.extend(self.ib4_2.get_vars())
        return vars_


if __name__ == "__main__":
    model = ResNet(
        name="resnet",
        outpt=64
    )
    B = 8
    V = 4
    W = H = 64
    C = 3
    data = np.zeros((B, V, W, H, C))
    features = model.compute(data)
