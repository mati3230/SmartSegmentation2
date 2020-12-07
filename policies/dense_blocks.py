import tensorflow as tf


class BottleNeckLayer(tf.keras.layers.Layer):
    def __init__(
            self,
            filter_size,
            kernel_size=(3, 3),
            kernel_initializer="he_normal",
            activation="relu",
            name="btln",
            trainable=True,
            dropout_rate=0.2,
            use_bias=False):
        super(BottleNeckLayer, self).__init__(name=name, trainable=trainable)
        self.kernel_size = kernel_size
        self.filter_size = filter_size
        self.kernel_initializer = kernel_initializer
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias

    def build(self, input_shape):
        c_name = self.name + "/c_"
        bn_name = self.name + "/bn_"
        d_name = self.name + "/drop_"
        self.conv2a = tf.keras.layers.Conv2D(
            4*self.filter_size,
            kernel_size=(1, 1),
            activation=None,
            padding="same",
            kernel_initializer=self.kernel_initializer,
            use_bias=self.use_bias,
            name=c_name+"a",
            trainable=self.trainable)
        self.conv2b = tf.keras.layers.Conv2D(
            self.filter_size,
            kernel_size=self.kernel_size,
            activation=None,
            padding="same",
            kernel_initializer=self.kernel_initializer,
            use_bias=self.use_bias,
            name=c_name+"b",
            trainable=self.trainable)
        self.act = tf.keras.layers.Activation(self.activation)
        self.bn2a = tf.keras.layers.BatchNormalization(
            name=bn_name+"a",
            trainable=self.trainable)
        self.bn2b = tf.keras.layers.BatchNormalization(
            name=bn_name+"b",
            trainable=self.trainable)
        self.dropa = tf.keras.layers.Dropout(
            rate=self.dropout_rate, name=d_name+"a")
        self.dropb = tf.keras.layers.Dropout(
            rate=self.dropout_rate, name=d_name+"b")

    def call(self, input, training=False):
        x = self.bn2a(input, training=training)
        x = self.act(x)
        x = self.conv2a(x)
        x = self.dropa(x, training=training)

        x = self.bn2b(x, training=training)
        x = self.act(x)
        x = self.conv2b(x)
        x = self.dropb(x, training=training)
        return x

    def get_vars(self):
        vars_ = []
        if self.trainable:
            vars_.extend(self.conv2a.trainable_weights)
            vars_.extend(self.conv2b.trainable_weights)
            vars_.extend(self.bn2a.trainable_weights)
            vars_.extend(self.bn2b.trainable_weights)
        else:
            vars_.extend(self.conv2a.non_trainable_weights)
            vars_.extend(self.conv2b.non_trainable_weights)
            vars_.extend(self.bn2a.non_trainable_weights)
            vars_.extend(self.bn2b.non_trainable_weights)
        return vars_


class TransitionLayer(tf.keras.layers.Layer):
    def __init__(
            self,
            kernel_initializer="he_normal",
            activation="relu",
            name="trans",
            trainable=True,
            dropout_rate=0.2,
            use_bias=False,
            filter_size=None):
        super(TransitionLayer, self).__init__(name=name, trainable=trainable)
        self.kernel_initializer = kernel_initializer
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.filter_size = filter_size

    def build(self, input_shape):
        c_name = self.name + "/c"
        bn_name = self.name + "/bn"
        d_name = self.name + "/drop"
        if not self.filter_size:
            in_channel = input_shape[-1]
            filter_size = tf.constant(0.5)*tf.cast(in_channel, tf.float32)
            self.filter_size = tf.cast(filter_size, tf.int32)
        self.conv2 = tf.keras.layers.Conv2D(
            self.filter_size,
            kernel_size=(1, 1),
            activation=None,
            padding="same",
            kernel_initializer=self.kernel_initializer,
            use_bias=self.use_bias,
            name=c_name,
            trainable=self.trainable)
        self.act = tf.keras.layers.Activation(self.activation)
        self.bn2 = tf.keras.layers.BatchNormalization(
            name=bn_name,
            trainable=self.trainable)
        self.drop = tf.keras.layers.Dropout(
            rate=self.dropout_rate, name=d_name)
        self.ap = tf.keras.layers.AveragePooling2D(name=self.name+"/ap")

    def call(self, input, training=False):
        x = self.bn2(input, training=training)
        x = self.act(x)
        x = self.conv2(x)
        x = self.drop(x, training=training)
        x = self.ap(x)
        return x

    def get_vars(self):
        vars_ = []
        if self.trainable:
            vars_.extend(self.conv2.trainable_weights)
            vars_.extend(self.bn2.trainable_weights)
        else:
            vars_.extend(self.conv2.non_trainable_weights)
            vars_.extend(self.bn2.non_trainable_weights)
        return vars_


class DenseBlock(tf.keras.layers.Layer):
    def __init__(
            self,
            filters,
            nb_layers,
            dropout_rate=0.2,
            kernel_size=(3, 3),
            kernel_initializer="he_normal",
            activation="relu",
            name="db",
            trainable=True,
            use_bias=False):
        super(DenseBlock, self).__init__(name=name, trainable=trainable)
        self.filters = filters
        self.nb_layers = nb_layers
        self.kernel_size = kernel_size
        self.kernel_initializer = kernel_initializer
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias

    def build(self, input_shape):
        self.btln_layers = []
        for i in range(self.nb_layers):
            btln_layer = BottleNeckLayer(
                filter_size=self.filters,
                kernel_size=self.kernel_size,
                kernel_initializer=self.kernel_initializer,
                activation=self.activation,
                name="btln_"+str(i),
                trainable=self.trainable,
                dropout_rate=self.dropout_rate,
                use_bias=self.use_bias
                )
            self.btln_layers.append(btln_layer)

    def call(self, input, training=False):
        layers_conc = list()
        layers_conc.append(input)
        x = self.btln_layers[0](input, training=training)
        layers_conc.append(x)

        for i in range(1, self.nb_layers):
            x = tf.concat(layers_conc, axis=3)
            x = self.btln_layers[i](x, training=training)
            layers_conc.append(x)
        x = tf.concat(layers_conc, axis=3)
        return x

    def get_vars(self):
        vars_ = []
        for i in range(self.nb_layers):
            vars_.extend(self.btln_layers[i].get_vars())
        return vars_
