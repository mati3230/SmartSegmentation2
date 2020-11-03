import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as k
import numpy as np


def get_lstm_var(
        num_units,
        name,
        trainable=True,
        stateful=False):
    lstm = tf.keras.layers.LSTM(
        units=num_units,
        name=name,
        trainable=trainable,
        return_state=False,
        stateful=stateful)
    return lstm


def get_keras_vars(k_var):
    vars_ = []
    for tr in k_var._self_unconditional_checkpoint_dependencies:
        vars_.append(tr.ref)
    return vars_


def get_conv2d_vars(
        inpt,
        outpt,
        name,
        fh=1,
        fw=1,
        trainable=True,
        seed=None,
        stddev=0.01,
        use_he=False):
    if use_he:
        conv = tf.Variable(
            initial_value=tf.random.normal(
                shape=[fh, fw, inpt, outpt],
                seed=seed,
                stddev=1) * tf.sqrt(2 / (fh*fw*inpt)),
            name=name+"_w",
            trainable=trainable)
    else:
        conv = tf.Variable(
            initial_value=tf.random.normal(
                shape=[fh, fw, inpt, outpt],
                seed=seed,
                stddev=stddev),
            name=name+"_w",
            trainable=trainable)
    b = tf.Variable(
        initial_value=tf.zeros(shape=[outpt]),
        name=name+"_b",
        trainable=trainable)
    return conv, b


def get_dense_vars(
        inpt,
        outpt,
        name,
        trainable=True,
        seed=None,
        stddev=0.03,
        use_he=False):
    if use_he:
        dense_w = tf.Variable(
            initial_value=tf.random.normal(
                shape=[inpt, outpt],
                seed=seed,
                stddev=1) * tf.sqrt(2/inpt),
            name=name+"_w",
            trainable=trainable)
    else:
        dense_w = tf.Variable(
            initial_value=tf.random.normal(
                shape=[inpt, outpt],
                seed=seed,
                stddev=stddev),
            name=name+"_w",
            trainable=trainable)
    dense_b = tf.Variable(
        initial_value=tf.zeros(shape=[outpt]),
        name=name+"_b",
        trainable=trainable)
    return dense_w, dense_b


@tf.function
def normalize(x):
    x_mean, x_var = tf.nn.moments(x, axes=[0])
    x = tf.nn.batch_normalization(
        x,
        mean=x_mean,
        variance=x_var,
        offset=0,
        scale=1,
        variance_epsilon=1e-6)
    return x


@tf.function
def conv2d(x, conv, b, activation=tf.nn.relu, sh_c=1, sw_c=1):
    net = tf.nn.conv2d(
        input=x,
        filters=conv,
        strides=[1, sh_c, sw_c, 1],
        padding="VALID"
    )

    net = tf.nn.bias_add(
        value=net,
        bias=b
    )

    y = activation(net)
    return y


@tf.function
def max_pool2d(x, ks_h, ks_w, s_h=2, s_w=2):
    return tf.nn.max_pool(
        input=x,
        ksize=[1, ks_h, ks_w, 1],
        strides=[1, s_h, s_w, 1],
        padding="VALID"
    )


@tf.function
def dense(x, w, b, activation=tf.nn.relu):
    """Fully connected activation.

    Parameters
    ----------
    x : tf.Tensor
        Input.
    w : tf.Tensor
        Weights.
    b : tf.Tensor
        Bias.
    activation : function
        Activation function.

    Returns
    -------
    tf.Tensor
        Description of returned object.

    """
    a = tf.matmul(x, w)
    a = tf.add(a, b)
    if activation:
        a = activation(a)
    return a


@tf.function
def kl_div(x, y):
    """Computes the kullback leibler divergence.

    Parameters
    ----------
    x : tf.Tensor
        Distribution.
    y : tf.Tensor
        Distribution.

    Returns
    -------
    tf.Tensor
        Kullback leibler divergence.

    """
    X = tfp.distributions.Categorical(logits=x)
    Y = tfp.distributions.Categorical(logits=y)
    return tfp.distributions.kl_divergence(X, Y, allow_nan_stats=False)


@tf.function
def sync(a, b):
    """Sync two networks such that $b \leftarrow a$. The networks a and b
    should have the same parameters.

    Parameters
    ----------
    a : BaseNet
        Net where parameters can be accessed as list via 'net.get_vars()'
        method.
    b : BaseNet
        Net where parameters can be accessed as list via 'net.get_vars()'
        method.

    """
    a_vars = a.get_vars()
    b_vars = b.get_vars()
    for a_var, b_var in zip(a_vars, b_vars):
        b_var.assign(a_var)


# @tf.function
def sync2(a_vars, b):
    """Sync two networks such that $b \leftarrow a$. The networks a and b
    should have the same parameters.

    Parameters
    ----------
    a_vars : list
        List of net parameters.
    b : BaseNet
        Net where parameters can be accessed as list via 'net.get_vars()'
        method.

    """
    b_vars = b.get_vars()
    if len(a_vars) != len(b_vars):
        max_ = min(len(a_vars), len(b_vars))
        offset = len("target_policy/")
        for i in range(max_):
            print(i, "\t", a_vars[i].name[offset:], "\t", b_vars[i].name[offset:])
        raise Exception("different number of weights to sync")
    for a_var, b_var in zip(a_vars, b_vars):
        b_var.assign(a_var)


@tf.function
def flatten_image(x):
    # x.shape[0]
    return tf.reshape(x, [tf.shape(x)[0], 1,  x.shape[1]*x.shape[2]*x.shape[3]])


if __name__ == "__main__":
    # Test of sync functionality
    conv_train = k.layers.Conv2D(
        filters=16,
        kernel_size=(3, 3),
        strides=(2, 2),
        activation="relu",
        name="train",
        trainable=True)
    conv_test = k.layers.Conv2D(
        filters=16,
        kernel_size=(3, 3),
        strides=(2, 2),
        activation="relu",
        name="test",
        trainable=False)
    sample_img = np.random.randn(1, 30, 30, 3)
    conv_train(sample_img)
    conv_test(sample_img)
    i = 0
    print(
        conv_train.trainable_weights[i],
        conv_test.non_trainable_weights[i])
    conv_test.non_trainable_weights[i].assign(conv_train.trainable_weights[i])
    print(
        conv_train.trainable_weights[i],
        conv_test.non_trainable_weights[i])
