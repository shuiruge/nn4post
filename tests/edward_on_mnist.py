#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description
-----------
Shadow neural network by `Edward ` on MNIST dataset.

This employs the mini-batch and scaling, c.f. [here](http://edwardlib.org/api/\
inference-data-subsampling) and [here](http://edwardlib.org/tutorials/batch-\
training) provides an explicit instance.


Remark
------
Running in `eshell` will raise `UnicodeEncodeError`; however, run it in `bash`
instead ceases this problem.
"""


import numpy as np
import tensorflow as tf
import edward as ed
from edward.models import Normal
import matplotlib.pyplot as plt
import sys
sys.path.append('../sample/')
from sklearn.utils import shuffle
import mnist_loader



ed.set_seed(42)  # for debugging.



class MNIST(object):
    """ Utils of loading, processing, and batch-emitting of MNIST dataset.

    The MNIST are (28, 28)-pixal images.

    Args:
        noise_std:
            `float`, as the standard derivative of Gaussian noise that is to
            add to output `y`.
        batch_size:
            `int`, as the size of mini-batch of training data. We employ no
            mini-batch for test data.
        dtype:
            Numpy `Dtype` object of `float`, optional. As the dtype of output
            data. Default is `np.float32`.
        seed:
            `int` or `None`, optional. If `int`, then set the random-seed of
            the noise in the output data. If `None`, do nothing. This arg is
            for debugging. Default is `None`.

    Attributes:
        x_train:
            Numpy array with shape `(10000, 784, 1)` and dtype `dtype`.
        y_train:
            Numpy array with shape `(10000, 10, 1)` and dtype `dtype`.
        x_test:
            Numpy array with shape `(10000, 784, 1)` and dtype `dtype`.
        y_test:
            Numpy array with shape `(10000,)` and dtype `dtype`.
        XXX


    Methods:
        XXX
    """

    def __init__(self, noise_std, batch_size,
                 dtype=np.float32, seed=None,
                 verbose=True):

        self._dtype = dtype
        self.batch_size = batch_size

        if seed is not None:
            np.random.seed(seed)

        training_data, validation_data, test_data = \
            mnist_loader.load_data_wrapper()

        # Preprocess training data
        x_tr, y_tr = training_data
        x_tr = self._preprocess(x_tr)
        y_tr = self._preprocess(y_tr)
        y_err_tr = noise_std * np.ones(y_tr.shape, dtype=self._dtype)
        self.training_data = (x_tr, y_tr, y_err_tr)

        self.n_data = len(x_tr)
        self.n_batches_per_epoch = int(self.n_data / self.batch_size)

        # Preprocess test data
        x_te, y_te = test_data
        x_te = self._preprocess(x_te)
        y_te = self._preprocess(y_te)
        y_err_te = 0.0 * np.ones(y_te.shape, dtype=dtype)
        self.test_data = (x_te, y_te, y_err_te)


    def _preprocess(self, data):
        """ Preprocessing MNIST data, including converting to numpy array,
            re-arrange the shape and dtype.

        Args:
            data:
                Any element of the tuple as the output of calling
                `mnist_loader.load_data_wrapper()`.

        Returns:
            The preprocessed, as numpy array. (This copies the input `data`,
            so that the input `data` will not be altered.)
        """
        data = np.asarray(data, dtype=self._dtype)
        data = np.squeeze(data)
        return data


    def batch_generator(self):
        """ A generator that emits mini-batch of training data, by acting
            `next()`.

        Returns:
            Tuple of three numpy arraies `(x, y, y_error)`, for the inputs of the
            model, the observed outputs of the model , and the standard derivatives
            of the observation, respectively. They are used for training only.
        """
        x, y, y_err = self.training_data
        batch_size = self.batch_size
        n_data = self.n_data

        while True:
            x, y, y_err = shuffle(x, y, y_err)  # XXX: copy ???

            for k in range(0, n_data, batch_size):
                mini_batch = (x[k:k+batch_size],
                                y[k:k+batch_size],
                                y_err[k:k+batch_size])
                yield mini_batch



mnist = MNIST(noise_std=0.1, batch_size=128)



# MODEL
with tf.name_scope("model"):


    # -- This sets the priors of model parameters. I.e. the :math:`p(\theta)`
    #    in the documentation.
    #
    #    Notice that the priors of the biases shall be uniform on
    #    :math:`\mathbb{R}`; herein we use `Normal()` with large `scale`
    #    (e.g. `100`) to approximate it.
    n_inputs = 28 * 28  # number of input features.
    n_hiddens = 30  # number of perceptrons in the (single) hidden layer.
    n_outputs = 10  # number of perceptrons in the output layer.
    w_h = Normal(loc=tf.zeros([n_inputs, n_hiddens]),
                 scale=tf.ones([n_inputs, n_hiddens]),
                 name="w_h")
    w_a = Normal(loc=tf.zeros([n_hiddens, n_outputs]),
                 scale=tf.ones([n_hiddens, n_outputs]),
                 name="w_a")
    b_h = Normal(loc=tf.zeros([n_hiddens]),
                 scale=tf.ones([n_hiddens]) * 100,
                 name="b_h")
    b_a = Normal(loc=tf.zeros([n_outputs]),
                 scale=tf.ones([n_outputs]) * 100,
                 name="b_a")


    # -- Placeholder for input data.
    x = tf.placeholder(tf.float32, [None, n_inputs],
                       name='x')
    y_error = tf.placeholder(tf.float32, [None, n_outputs],
                             name='y_error')


    # -- The model architecture. This is for getting the likelihood.
    def neural_network(x):
        """
        Args:
            x:
                Tensor object with shape `[None, n_inputs]` and dtype
                `tf.float32`.

        Returns:
            Tensor object with shape `[None, n_outputs]` and dtype `tf.float32`.
        """
        # shape: `[None, n_hiddens]`
        hidden = tf.sigmoid(tf.matmul(x, w_h) + b_h)
        # shape: `[None, n_outputs]`
        activation = tf.nn.softmax(tf.matmul(hidden, w_a) + b_a)
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
        qw_h = Normal(loc=tf.Variable(tf.random_normal([n_inputs, n_hiddens]),
                                      name="loc"),
                      scale=tf.nn.softplus(
                          tf.Variable(tf.random_normal([n_inputs, n_hiddens]),
                                      name="scale")))
    with tf.name_scope("qw_a"):
        qw_a = Normal(loc=tf.Variable(tf.random_normal([n_hiddens, n_outputs]),
                                      name="loc"),
                      scale=tf.nn.softplus(
                          tf.Variable(tf.random_normal([n_hiddens, n_outputs]),
                                      name="scale")))
    with tf.name_scope("qb_h"):
        qb_h = Normal(loc=tf.Variable(tf.random_normal([n_hiddens]),
                                      name="loc"),
                      scale=tf.nn.softplus(
                          tf.Variable(tf.random_normal([n_hiddens]),
                                      name="scale")))
    with tf.name_scope("qb_a"):
        qb_a = Normal(loc=tf.Variable(tf.random_normal([n_outputs]),
                                      name="loc"),
                      scale=tf.nn.softplus(
                          tf.Variable(tf.random_normal([n_outputs]),
                                      name="scale")))



# PLAY
# Set the parameters of training
logdir = '../dat/logs'
n_batch = mnist.batch_size
n_epoch = 1  # test!
n_samples = 10  # test!
scale = {y: mnist.n_batches_per_epoch}
y_ph = tf.placeholder(tf.float32, [None, n_outputs])
inference = ed.KLqp(latent_vars={w_h: qw_h, b_h: qb_h,
                                 w_a: qw_a, b_a: qb_a},
                    data={y: y_ph})
inference.initialize(
    n_iter=n_batch * n_epoch,
    n_samples=n_samples,
    scale=scale,
    logdir=logdir)
tf.global_variables_initializer().run()

for _ in range(inference.n_iter):
    batch_generator = mnist.batch_generator()
    x_batch, y_batch, y_error_batch = next(batch_generator)
    info_dict = inference.update({x: x_batch,
                                  y_ph: y_batch,
                                  y_error: y_error_batch})
    inference.print_progress(info_dict)


# EVALUATE
# -- That is, check your result.
x_test, y_test, y_error_test = mnist.test_data
y_post = ed.copy(y, {w_h: qw_h, b_h: qb_h,
                     w_a: qw_a, b_a: qb_a})
categorical_accuracy = ed.evaluate(
    'sparse_categorical_accuracy',
    data={x: x_test, y_post: y_test, y_error: y_error_test})
print('Categorical accuracy on test data: {0}'\
      .format(categorical_accuracy))
