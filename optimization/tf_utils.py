import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as k
import numpy as np


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
