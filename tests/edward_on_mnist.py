#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description
-----------
Shallow neural network by `Edward ` on MNIST dataset.

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
from edward.models import Normal, NormalWithSoftplusScale
import matplotlib.pyplot as plt
import sys
sys.path.append('../sample/')
from sklearn.utils import shuffle
from tools import get_accuracy
import mnist
import time



ed.set_seed(42)  # for debugging.



# DATA
noise_std = 0.1
batch_size = 16  # test!
mnist_ = mnist.MNIST(noise_std, batch_size)



# MODEL
with tf.name_scope("model"):


    # -- This sets the priors of model parameters. I.e. the :math:`p(\theta)`
    #    in the documentation.
    #
    #    Notice that the priors of the biases shall be uniform on
    #    :math:`\mathbb{R}`; herein we use `Normal()` with large `scale`
    #    (e.g. `100`) to approximate it.
    n_inputs = 28 * 28  # number of input features.
    n_hiddens = 100  # number of perceptrons in the (single) hidden layer.
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


    # -- This sets the :math:`q( \theta \mid \nu )`, where :math:`\nu` is as
    #    the :math:`(a, \mu, \zeta)` in the documentation. The `loc`s and
    #    `scale`s are variables that is to be tuned in every iteration of
    #    optimization (by `Inference.update()`).
    #
    #    CAUTION:
    #        using `ed.models.MultivariateNormalDiag` (the same as the
    #        `tf.contrib.distributions.MultivariateNormalDiag`) makes the
    #        inference extremely slow, comparing with using `ed.models.Normal`.
    #        The reason is unclear.
    with tf.name_scope("qw_h"):
        loc_qw_h = tf.Variable(
            tf.random_normal([n_inputs, n_hiddens]),
            name='loc')
        scale_qw_h = tf.Variable(
            tf.random_normal([n_inputs, n_hiddens]),
            name='scale')
        qw_h = NormalWithSoftplusScale(loc=loc_qw_h, scale=scale_qw_h)
    with tf.name_scope("qw_a"):
        loc_qw_a = tf.Variable(
            tf.random_normal([n_hiddens, n_outputs]),
            name='loc')
        scale_qw_a = tf.Variable(
            tf.random_normal([n_hiddens, n_outputs]),
            name="scale")
        qw_a = NormalWithSoftplusScale(loc=loc_qw_a, scale=scale_qw_a)
    with tf.name_scope("qb_h"):
        loc_qb_h = tf.Variable(
            tf.random_normal([n_hiddens]),
            name="loc")
        scale_qb_h = tf.Variable(
            tf.random_normal([n_hiddens]),
            name="scale")
        qb_h = NormalWithSoftplusScale(loc=loc_qb_h, scale=scale_qb_h)
    with tf.name_scope("qb_a"):
        loc_qb_a = tf.Variable(
            tf.random_normal([n_outputs]),
            name="loc")
        scale_qb_a = tf.Variable(
            tf.random_normal([n_outputs]),
            name="scale")
        qb_a = NormalWithSoftplusScale(loc=loc_qb_a, scale=scale_qb_a)




# PLAY
# Set the parameters of training
n_epochs = 30
n_iter = mnist_.n_batches_per_epoch * n_epochs
n_samples = 100
scale = {y: mnist_.n_data / mnist_.batch_size}
logdir = '../dat/logs'

y_ph = tf.placeholder(tf.float32, [None, n_outputs],
                      name='y')
inference = ed.KLqp(latent_vars={w_h: qw_h, b_h: qb_h,
                                 w_a: qw_a, b_a: qb_a},
                    data={y: y_ph})
inference.initialize(
    n_iter=n_iter,
    n_samples=n_samples,
    scale=scale,
    logdir=logdir)
tf.global_variables_initializer().run()


sess = ed.get_session()


# Add node of posterior to graph
prediction_post = ed.copy(prediction,
                          { w_h: qw_h, b_h: qb_h,
                            w_a: qw_a, b_a: qb_a })

time_start = time.time()
for i in range(inference.n_iter):

    batch_generator = mnist_.batch_generator()
    x_batch, y_batch, y_error_batch = next(batch_generator)
    feed_dict = {x: x_batch,
                    y_ph: y_batch,
                    y_error: y_error_batch}


    _, t, loss = sess.run(
        [ inference.train, inference.increment_t,
            inference.loss ],
        feed_dict)

    # Validation for each epoch
    if i % mnist_.n_batches_per_epoch == 0:

        print('\nFinished the {0}-th epoch'\
                .format(i/mnist_.n_batches_per_epoch))
        print('Elapsed time {0} sec.'.format(time.time()-time_start))

        # Get validation data
        x_valid, y_valid, y_error_valid = mnist_.validation_data
        x_valid, y_valid, y_error_valid = \
            shuffle(x_valid, y_valid, y_error_valid)
        x_valid, y_valid, y_error_valid = \
            x_valid[:128], y_valid[:128], y_error_valid[:128]

        # Get accuracy
        n_models = 100  # number of Monte Carlo neural network models.
        # shape: [n_models, n_test_data, n_outputs]
        softmax_vals = [prediction_post.eval(feed_dict={x: x_valid})
                        for i in range(n_models)]
        # shape: [n_test_data, n_outputs]
        mean_softmax_vals = np.mean(softmax_vals, axis=0)
        # shape: [n_test_data]
        y_pred = np.argmax(mean_softmax_vals, axis=-1)
        accuracy = get_accuracy(y_pred, y_valid)

        print('Accuracy on validation data: {0} %'\
                .format(accuracy/mnist_.batch_size*100))
        time_start = time.time()  # re-initialize.


# EVALUATE
x_test, y_test, y_error_test = mnist_.test_data
n_test_data = len(y_test)
print('{0} test data.'.format(n_test_data))

n_models = 500  # number of Monte Carlo neural network models.
# shape: [n_models, n_test_data, n_outputs]
prediction_vals = [prediction_post.eval(feed_dict={x: x_test})
                    for i in range(n_models)]

# Get accuracy
n_models = 100  # number of Monte Carlo neural network models.
# shape: [n_models, n_test_data, n_outputs]
softmax_vals = [prediction_post.eval(feed_dict={x: x_test})
                for i in range(n_models)]
# shape: [n_test_data, n_outputs]
mean_softmax_vals = np.mean(softmax_vals, axis=0)
# shape: [n_test_data]
y_pred = np.argmax(mean_softmax_vals, axis=-1)
accuracy = get_accuracy(y_pred, y_test)

print('Accuracy on test data: {0} %'\
        .format(accuracy/mnist_.batch_size*100))
time_start = time.time()  # re-initialize.





''' Conclusion:

1   n_hiddens = 30
    n_samples = 100
    n_epochs = 5
    batch_size = 128

    => Accuracy on test data: 92.76 %


2   n_hiddens = 100
    n_samples = 100
    n_epochs = 5
    batch_size = 128

    => Accuracy on test data: 95.68 %


3   n_hiddens = 100
    n_samples = 100
    n_epochs = 10
    batch_size = 128

    => Accuracy on test data: 96.88 %


4   n_hiddens = 100
    n_samples = 100
    n_epochs = 30
    batch_size = 128

    => Accuracy on test data: 97.16 %
'''
