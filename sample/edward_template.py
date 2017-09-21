#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description
-----------
A Bayesian shadow neural network by Edward, c.f.
[here](https://github.com/blei-lab/edward/blob/master/examples/bayesian_nn.py).

This is a template of how Edward is played.


Remark
------
Running in `eshell` will raise `UnicodeEncodeError`; however, run it in `bash`
instead ceases this problem.


TODO
----
[ ] - Use MNIST dataset instead.
"""


import numpy as np
import tensorflow as tf
import edward as ed
from edward.models import Normal



ed.set_seed(42)  # for debugging.



def generate_dataset(
        n_data, noise_std, data_range=(-7, 7),
        dtype=np.float32, seed=None):
    """ Generate a dataset with target function math:`\sin(x)`.

    Args:
        n_data:
            `int`, as the number of data
        noise_std:
            `float`, as the standard derivative of Gaussian noise that is to
            add to output `y`.
        data_range:
             Tuple of `float` or `int`, as the range of data `x`
        dtype:
            Numpy `Dtype` object of `float`, optional. As the dtype of output
            data. Default is `np.float32`.
        seed:
            `int` or `None`, optional. If `int`, then set the random-seed of
            the noise in the output data. If `None`, do nothing. This arg is
            for debugging. Default is `None`.

    Returns:
        Tuple of three numpy arraies `(x, y, y_error)`, for the inputs of the,
        model, the observed outputs of the model , and the standard derivatives
        of the observation, respectively. They share the shape `[n_data, 1]`
        and dtype 'np.float32'.
    """

    if seed is not None:
        np.random.seed(seed)

    def target_func(x):
        """ The target function neural network is to fit. """
        return np.sin(x) * 0.5

    # Generate `x`
    x = np.linspace(*data_range, n_data)
    x = np.expand_dims(x, -1)  # shape: [n_data, 1]
    x.astype(dtype)

    # Generate `y`
    y = target_func(x)  # shape: [n_data, 1]
    y += noise_std * np.random.normal(size=[n_data, 1])
    y.astype(dtype)

    # Generate `y_error`
    y_error = np.random.uniform(0.0, 0.2, size=[n_data, 1])

    return (x, y, y_error)



# MODEL
with tf.name_scope("model"):


    # -- Set the number of data. This is absent in traditional way in bulding
    #    model, instead using a `None` for data-size or mini-batch-size.
    #    However, in Baysian way, this is essential, since more data encodes
    #    more confidience (and such encoding is absent in traditional way).
    n_data = 100


    # -- This sets the priors of model parameters. I.e. the :math:`p(\theta)`
    #    in the documentation.
    n_hiddens = 10  # number of perceptrons in the (single) hidden layer.
    w_h = Normal(loc=tf.zeros([1, n_hiddens]),
                 scale=tf.ones([1, n_hiddens]),
                 name="w_h")
    w_a = Normal(loc=tf.zeros([n_hiddens, 1]),
                 scale=tf.ones([n_hiddens, 1]),
                 name="w_a")
    b_h = Normal(loc=tf.zeros([n_hiddens]),
                 scale=tf.ones([n_hiddens]),
                 name="b_h")
    b_a = Normal(loc=tf.zeros([1]),
                 scale=tf.ones([1]),
                 name="b_a")


    # -- Placeholder for input data.
    x = tf.placeholder(tf.float32, [n_data, 1],
                       name='x')
    y_error = tf.placeholder(tf.float32, [n_data, 1],
                             name='y_error')


    # -- The model architecture. This is for getting the likelihood.
    def neural_network(x):
        """
        Args:
            x:
                Tensor object with shape `[n_data, 1]` and dtype `tf.float32`.

        Returns:
            Tensor object with shape `[n_data, 1]` and dtype `tf.float32`.
        """
        # shape: `[n_data, n_hidden]`
        hidden = tf.sigmoid(tf.matmul(x, w_h) + b_h)
        # shape: `[n_data, 1]`
        activation = tf.matmul(hidden, w_a) + b_a
        return activation

    # -- This sets the likelihood. I.e. the :math:`p( D \mid \theta )`
    #    in the documentation.
    #    Precisely, this gives :math:`p( y \mid w, b; x)`, where :math:`(w, b)`
    #    is the :math:`\theta`, and :math:`x` is deterministic and fixed
    #    throughout the process of Bayesian inference.
    #    (The `0.1` may not be equal to the `noise_std` in the data, which we
    #     do not known. This number of a prior in fact.)
    y = Normal(loc=neural_network(x),  # recall shape: `[n_data, 1]`.
               scale=y_error,
               name="y")



# INFERENCE
with tf.name_scope("posterior"):


    # -- This sets the :math:`q( \theta \mid \mu )`, where :math:`\mu` is as
    #    the :math:`(a, \mu, \zeta)` in the documentation.
    with tf.name_scope("qw_h"):
        qw_h = Normal(loc=tf.Variable(tf.random_normal([1, n_hiddens]),
                                      name="loc"),
                      scale=tf.nn.softplus(
                          tf.Variable(tf.random_normal([1, n_hiddens]),
                                      name="scale")))
    with tf.name_scope("qw_a"):
        qw_a = Normal(loc=tf.Variable(tf.random_normal([n_hiddens, 1]),
                                      name="loc"),
                      scale=tf.nn.softplus(
                          tf.Variable(tf.random_normal([n_hiddens, 1]),
                                      name="scale")))
    with tf.name_scope("qb_h"):
        qb_h = Normal(loc=tf.Variable(tf.random_normal([n_hiddens]),
                                      name="loc"),
                      scale=tf.nn.softplus(
                          tf.Variable(tf.random_normal([n_hiddens]),
                                      name="scale")))
    with tf.name_scope("qb_a"):
        qb_a = Normal(loc=tf.Variable(tf.random_normal([1]),
                                      name="loc"),
                      scale=tf.nn.softplus(
                          tf.Variable(tf.random_normal([1]),
                                      name="scale")))



# PLAY
# Remind `n_data` has been defined in building up the Bayesian.
x_train, y_train, y_error_train = generate_dataset(n_data=n_data, noise_std=0.1)

inference = ed.KLqp(latent_vars={w_h: qw_h, b_h: qb_h,
                                 w_a: qw_a, b_a: qb_a},
                    data={x: x_train, y: y_train, y_error: y_error_train})
inference.run(logdir='../dat/log', n_iter=2000)



# EVALUATE
x_test, y_test, y_error_test = generate_dataset(n_data=n_data, noise_std=0.0)
y_post = ed.copy(y, {w_h: qw_h, b_h: qb_h,
                     w_a: qw_a, b_a: qb_a})
print("Mean squared error on test data:")
print(ed.evaluate('mean_squared_error',
                  data={x: x_test, y_post: y_test, y_error: y_error_test}))
print("Mean absolute error on test data:")
print(ed.evaluate('mean_absolute_error',
                  data={x: x_test, y_post: y_test, y_error: y_error_test}))
