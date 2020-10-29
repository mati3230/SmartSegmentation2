import tensorflow as tf
import numpy as np
from optimization.base_net import BaseNet
from dense_blocks import DenseBlock, TransitionLayer


class DenseNet(BaseNet):
    def __init__(
            self,
            name,
            outpt,
            trainable=True,
            seed=None,
            check_numerics=False,
            stddev=0.3,
            stateful=False,
            initializer="he_normal",
            nd_blocks=4,
            growth_k=12,
            activation="relu",
            use_bias=False,
            use_max=False):
        super().__init__(
            name=name,
            outpt=outpt,
            trainable=trainable,
            seed=seed)
        self.nd_blocks = nd_blocks
        self.trainable = trainable
        self.use_max = use_max
        self.act = tf.keras.layers.Activation(activation)
        self.bn1 = tf.keras.layers.BatchNormalization(
            name=name+"/bn1",
            trainable=trainable)
        self.conv1 = tf.keras.layers.Conv2D(
            filters=2 * growth_k,
            kernel_size=(7, 7),
            strides=(2, 2),
            activation=None,
            name=name+"/c1",
            trainable=trainable,
            kernel_initializer=initializer,
            padding="same",
            use_bias=use_bias)
        self.mp1 = tf.keras.layers.MaxPool2D(
            pool_size=(3, 3),
            strides=(2, 2),
            padding="same",
            name=name+"/mp1")
        self.gap = tf.keras.layers.GlobalAveragePooling2D(name=name+"/gap")
        self.d_blocks = []
        for i in range(self.nd_blocks):
            db = DenseBlock(
                filters=growth_k,
                nb_layers=4,
                name="db_"+str(i),
                kernel_initializer=initializer,
                trainable=trainable,
                use_bias=use_bias)
            tl = TransitionLayer(
                kernel_initializer=initializer,
                name="tl_"+str(i),
                trainable=trainable,
                use_bias=use_bias,
                filter_size=None)
            self.d_blocks.append((db, tl))

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
            net = obs[:, i]
            # print(i, net.shape)
            net = self.conv1(net)
            net = self.mp1(net)
            # print(i, net.shape)
            for j in range(self.nd_blocks):
                db, tl = self.d_blocks[j]
                net = db(net, training=training)
                # print(i, j, net.shape)
                net = tl(net, training=training)
                # print(i, j, net.shape)
            net = self.bn1(net, training=training)
            # print(i, net.shape)
            net = self.act(net)
            # print(i, net.shape)
            net = self.gap(net)
            # print(i, net.shape)
            features = features.write(i, net)
        # vp = tf.concat(features, axis=0)
        vp = features.stack()
        # print("vp:", vp.shape)
        if self.use_max:
            net = tf.math.reduce_max(vp, axis=0, keepdims=True)
        else:
            net = tf.math.reduce_mean(vp, axis=0, keepdims=True)
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
        for i in range(self.nd_blocks):
            db, tl = self.d_blocks[i]
            vars_.extend(db.get_vars())
            vars_.extend(tl.get_vars())
        return vars_


if __name__ == "__main__":
    model = DenseNet(
        name="densenet",
        outpt=64,
        growth_k=12,
        use_bias=False,
        use_max=False
    )
    B = 2
    V = 4
    W = H = 224
    C = 3
    data = np.zeros((B, V, W, H, C))
    features = model.compute(data, training=False)
