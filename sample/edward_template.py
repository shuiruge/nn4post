#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description
-----------
This is a template of how Edward is played.

A Bayesian shadow neural network by Edward, c.f. [here](https://github.com/blei\
-lab/edward/blob/master/examples/bayesian_nn.py).

Herein we use `sin()`,as the target function that the neural network is to fit.



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
import matplotlib.pyplot as plt



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

    def target_fn(x):
        """ The target function neural network is to fit. """
        return np.sin(x) * 0.5

    # Generate `x`
    x = np.linspace(*data_range, n_data)
    x = np.expand_dims(x, -1)  # shape: [n_data, 1]
    x.astype(dtype)

    # Generate `y`
    y = target_fn(x)  # shape: [n_data, 1]
    y += noise_std * np.random.normal(size=[n_data, 1])
    y.astype(dtype)

    # Generate `y_error`
    # -- When we measure `y` in practice, we expect that the `y_error`
    #    characterizes its noise forsooth.
    y_error = np.ones([n_data, 1]) * noise_std

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
    #
    #    Notice that the priors of the biases shall be uniform on
    #    :math:`\mathbb{R}`; herein we use `Normal()` with large `scale`
    #    (e.g. `100`) to approximate it.
    n_hiddens = 10  # number of perceptrons in the (single) hidden layer.
    w_h = Normal(loc=tf.zeros([1, n_hiddens]),
                 scale=tf.ones([1, n_hiddens]),
                 name="w_h")
    w_a = Normal(loc=tf.zeros([n_hiddens, 1]),
                 scale=tf.ones([n_hiddens, 1]),
                 name="w_a")
    b_h = Normal(loc=tf.zeros([n_hiddens]),
                 scale=tf.ones([n_hiddens]) * 100,
                 name="b_h")
    b_a = Normal(loc=tf.zeros([1]),
                 scale=tf.ones([1]) * 100,
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
    prediction = neural_network(x)

    # -- This sets the likelihood. I.e. the :math:`p( D \mid \theta )`
    #    in the documentation.
    #    Precisely, this gives :math:`p( y \mid w, b; x)`, where :math:`(w, b)`
    #    is the :math:`\theta`, and :math:`x` is deterministic and fixed
    #    throughout the process of Bayesian inference.
    #    (The `0.1` may not be equal to the `noise_std` in the data, which we
    #     do not known. This number of a prior in fact.)
    y = Normal(loc=prediction,  # recall shape: `[n_data, 1]`.
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

# -- Inference by minimizing the K-L divergence (ELBO in fact).
#    The `ed.KLqp()` inherits the abstract base class `Inference()`.
inference = ed.KLqp(latent_vars={w_h: qw_h, b_h: qb_h,
                                 w_a: qw_a, b_a: qb_a},
                    data={x: x_train, y: y_train, y_error: y_error_train})

# -- `Inference.run()` will:
#        1. setup computational graph by `Inference.initialize()`;
#        2. iteratively run the graph by `Inference.update()`.
inference.run(logdir='../dat/log', n_iter=1000, n_samples=100)



# EVALUATE
# -- That is, check your result.
x_test, y_test, y_error_test = generate_dataset(n_data=n_data, noise_std=0.0)
y_post = ed.copy(y, {w_h: qw_h, b_h: qb_h,
                     w_a: qw_a, b_a: qb_a})
mean_abs_err = ed.evaluate(
    'mean_absolute_error',
    data={x: x_test, y_post: y_test, y_error: y_error_test})
print('Mean absolute error on test data: {0}'\
      .format(mean_abs_err))
# Question:
#     It seems that `y_error_test` is essential, or ERROR raises. But why?



# -- In `prediction`, all are still in their priors. To employ the prediction op
#    with the trained posterior, thus, using the `prediction` leads to mistake.
#    Instead, we shall first create a new node of prediction op in the
#    computational graph, with the same topology as `prediction`, but with the
#    trained posterior. The `ed.copy()` helps. It creates a new node of
#    `org_instance` arg in the computational graph of `Inference` class by
#    copying the topology of `org_instance`, but with posterior feeded by the
#    trained via `dict_swap` arg.
prediction_post = ed.copy(org_instance=prediction,
                          dict_swap={w_h: qw_h, b_h: qb_h,
                                     w_a: qw_a, b_a: qb_a})
# -- Now with the feeding, `prediction_post` as a `RandomVariable` instance has
#    the distribution of the trained posterior.
predictions = [prediction_post.eval(feed_dict={x: x_test})
               for i in range(10)]



# PLOT
# Setup an empty figure
fig, ax = plt.subplots(1)
ax.set_title('Shadow Neural Network (hidden: {0})'.format(n_hiddens))

# Draw target function
#ax.plot(x_test, y_test, ls='solid', label='target')

# Draw data points
ax.plot(x_train, y_train, '.', label='data')

# Draw what the trained Bayesian neural network inferences
for pred in predictions:
    ax.plot(x_test, pred.reshape(-1), ls='-')

ax.legend(loc='best', fancybox=True, framealpha=0.5)
plt.show()
