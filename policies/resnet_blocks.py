import tensorflow as tf
import numpy as np


class IdentityBlock(tf.keras.layers.Layer):
    def __init__(
            self,
            filters,
            kernel_size=(3, 3),
            kernel_initializer="he_normal",
            activation="relu",
            name="rip",
            trainable=True):
        super(IdentityBlock, self).__init__(name=name, trainable=trainable)
        self.kernel_size = kernel_size
        self.filters = filters
        self.kernel_initializer = kernel_initializer
        self.activation = activation

    def build(self, input_shape):
        filter1, filter2, filter3 = self.filters
        c_name = self.name + "/c_"
        bn_name = self.name + "/bn_"
        self.conv2a = tf.keras.layers.Conv2D(
            filter1,
            kernel_size=(1, 1),
            activation=None,
            padding="same",
            kernel_initializer=self.kernel_initializer,
            use_bias=False,
            name=c_name+"a",
            trainable=self.trainable)
        self.bn2a = tf.keras.layers.BatchNormalization(
            name=bn_name+"a",
            trainable=self.trainable)
        self.conv2b = tf.keras.layers.Conv2D(
            filter2,
            kernel_size=self.kernel_size,
            activation=None,
            padding="same",
            kernel_initializer=self.kernel_initializer,
            use_bias=False,
            name=c_name+"b",
            trainable=self.trainable)
        self.bn2b = tf.keras.layers.BatchNormalization(
            name=bn_name+"b",
            trainable=self.trainable)
        self.conv2c = tf.keras.layers.Conv2D(
            filter3,
            kernel_size=(1, 1),
            activation=None,
            padding="same",
            kernel_initializer=self.kernel_initializer,
            use_bias=False,
            name=c_name+"c",
            trainable=self.trainable)
        self.bn2c = tf.keras.layers.BatchNormalization(
            name=bn_name+"c",
            trainable=self.trainable)
        self.act = tf.keras.layers.Activation(self.activation)

        self.conv2d = None
        if input_shape[-1] != filter3:
            self.conv2d = tf.keras.layers.Conv2D(
                filter3,
                self.kernel_size,
                activation=None,
                padding="same",
                kernel_initializer=self.kernel_initializer,
                use_bias=False,
                name=c_name+"d",
                trainable=self.trainable)
            self.bn2d = tf.keras.layers.BatchNormalization(
                name=bn_name+"d",
                trainable=self.trainable)

    @tf.function
    def call(self, input, training=False):
        x = self.conv2a(input)
        x = self.bn2a(x, training=training)
        x = self.act(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = self.act(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)

        if self.conv2d:
            input_conv = self.conv2d(input)
            input_conv = self.bn2d(input_conv)
            x += input_conv
        else:
            x += input
        return self.act(x)

    def get_vars(self):
        vars_ = []
        if self.trainable:
            vars_.extend(self.conv2a.trainable_weights)
            vars_.extend(self.conv2b.trainable_weights)
            vars_.extend(self.conv2c.trainable_weights)
            vars_.extend(self.bn2a.trainable_weights)
            vars_.extend(self.bn2b.trainable_weights)
            vars_.extend(self.bn2c.trainable_weights)
            if self.conv2d:
                vars_.extend(self.conv2d.trainable_weights)
                vars_.extend(self.bn2d.trainable_weights)
        else:
            vars_.extend(self.conv2a.non_trainable_weights)
            vars_.extend(self.conv2b.non_trainable_weights)
            vars_.extend(self.conv2c.non_trainable_weights)
            vars_.extend(self.bn2a.non_trainable_weights)
            vars_.extend(self.bn2b.non_trainable_weights)
            vars_.extend(self.bn2c.non_trainable_weights)
            if self.conv2d:
                vars_.extend(self.conv2d.non_trainable_weights)
                vars_.extend(self.bn2d.non_trainable_weights)
        return vars_


class ConvBlock(tf.keras.layers.Layer):
    def __init__(
            self,
            filters,
            kernel_size=(3, 3),
            kernel_initializer="he_normal",
            activation="relu",
            name="rip",
            trainable=True):
        super(ConvBlock, self).__init__(name=name, trainable=trainable)
        self.kernel_size = kernel_size
        self.filters = filters
        self.kernel_initializer = kernel_initializer
        self.activation = activation

    def build(self, input_shape):
        filter1, filter2, filter3 = self.filters
        c_name = self.name + "/c_"
        bn_name = self.name + "/bn_"
        self.conv2a = tf.keras.layers.Conv2D(
            filter1,
            kernel_size=(1, 1),
            activation=None,
            padding="same",
            kernel_initializer=self.kernel_initializer,
            use_bias=False,
            name=c_name+"a",
            trainable=self.trainable)
        self.bn2a = tf.keras.layers.BatchNormalization(
            name=bn_name+"a",
            trainable=self.trainable)
        self.conv2b = tf.keras.layers.Conv2D(
            filter2,
            self.kernel_size,
            strides=(2, 2),
            activation=None,
            padding="same",
            kernel_initializer=self.kernel_initializer,
            use_bias=False,
            name=c_name+"b",
            trainable=self.trainable)
        self.bn2b = tf.keras.layers.BatchNormalization(
            name=bn_name+"b",
            trainable=self.trainable)
        self.conv2c = tf.keras.layers.Conv2D(
            filter3,
            kernel_size=(1, 1),
            activation=None,
            padding="same",
            kernel_initializer=self.kernel_initializer,
            use_bias=False,
            name=c_name+"c",
            trainable=self.trainable)
        self.bn2c = tf.keras.layers.BatchNormalization(
            name=bn_name+"c",
            trainable=self.trainable)
        self.act = tf.keras.layers.Activation(self.activation)

        self.conv2d = None
        if input_shape[-1] != filter3:
            self.conv2d = tf.keras.layers.Conv2D(
                filter3,
                kernel_size=(1, 1),
                strides=(2, 2),
                activation=None,
                padding="same",
                kernel_initializer=self.kernel_initializer,
                use_bias=False,
                name=c_name+"d",
                trainable=self.trainable)
            self.bn2d = tf.keras.layers.BatchNormalization(
                name=bn_name+"d",
                trainable=self.trainable)

    @tf.function
    def call(self, input, training=False):
        x = self.conv2a(input)
        x = self.bn2a(x, training=training)
        x = self.act(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = self.act(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)

        if self.conv2d:
            input_conv = self.conv2d(input)
            input_conv = self.bn2d(input_conv)
            x += input_conv
        else:
            x += input
        return self.act(x)

    def get_vars(self, to_sync=False):
        vars_ = []
        if self.trainable:
            vars_.extend(self.conv2a.trainable_weights)
            vars_.extend(self.conv2b.trainable_weights)
            vars_.extend(self.conv2c.trainable_weights)
            if to_sync:
                vars_.extend(self.bn2a.non_trainable_weights)
                vars_.extend(self.bn2b.non_trainable_weights)
                vars_.extend(self.bn2c.non_trainable_weights)
            else:
                vars_.extend(self.bn2a.trainable_weights)
                vars_.extend(self.bn2b.trainable_weights)
                vars_.extend(self.bn2c.trainable_weights)
            if self.conv2d:
                vars_.extend(self.conv2d.trainable_weights)
                if to_sync:
                    vars_.extend(self.bn2d.non_trainable_weights)
                else:
                    vars_.extend(self.bn2d.trainable_weights)
        else:
            vars_.extend(self.conv2a.non_trainable_weights)
            vars_.extend(self.conv2b.non_trainable_weights)
            vars_.extend(self.conv2c.non_trainable_weights)
            vars_.extend(self.bn2a.non_trainable_weights)
            vars_.extend(self.bn2b.non_trainable_weights)
            vars_.extend(self.bn2c.non_trainable_weights)
            if self.conv2d:
                vars_.extend(self.conv2d.non_trainable_weights)
                vars_.extend(self.bn2d.non_trainable_weights)
        return vars_


if __name__ == "__main__":
    img = np.random.randn(1, 64, 64, 3)
    img = img.astype(np.float32)
    i_b1 = IdentityBlock(filters=[64, 64, 256], kernel_size=(3, 3), name="rib1")
    i_b2 = IdentityBlock(filters=[64, 64, 256], kernel_size=(3, 3), name="rib2")
    i_b3 = IdentityBlock(filters=[64, 64, 256], kernel_size=(3, 3), name="rib3")

    c_b1 = ConvBlock(filters=[128, 128, 512], kernel_size=(3, 3), name="rcb1")
    i_b4 = IdentityBlock(filters=[128, 128, 512], kernel_size=(3, 3), name="rib4")
    i_b5 = IdentityBlock(filters=[128, 128, 512], kernel_size=(3, 3), name="rib5")
    i_b6 = IdentityBlock(filters=[128, 128, 512], kernel_size=(3, 3), name="rib6")
    x = i_b1(img)
    x = i_b2(x)
    x = i_b3(x)
    print(x.shape)
    x = c_b1(x)
    print(x.shape)
    x = i_b4(x)
    x = i_b5(x)
    x = i_b6(x)
    print(x.shape)
