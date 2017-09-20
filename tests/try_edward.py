#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description
-----------
Get familiar with Edward. This is forked and modified from
[here](https://github.com/blei-lab/edward/blob/master/examples/bayesian_nn.py).


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


def build_toy_dataset(N=40, noise_std=0.1):
    D = 1
    X = np.concatenate([np.linspace(0, 2, num=N / 2),
                        np.linspace(6, 8, num=N / 2)])
    y = np.cos(X) + np.random.normal(0, noise_std, size=N)
    X = (X - 4.0) / 4.0
    X = X.reshape((N, D))
    return X, y



ed.set_seed(42)

N = 40  # number of data points
D = 1   # number of features

# DATA
X_train, y_train = build_toy_dataset(N)

# MODEL
with tf.name_scope("model"):
    W_0 = Normal(loc=tf.zeros([D, 10]), scale=tf.ones([D, 10]), name="W_0")
    W_1 = Normal(loc=tf.zeros([10, 10]), scale=tf.ones([10, 10]), name="W_1")
    W_2 = Normal(loc=tf.zeros([10, 1]), scale=tf.ones([10, 1]), name="W_2")
    b_0 = Normal(loc=tf.zeros(10), scale=tf.ones(10), name="b_0")
    b_1 = Normal(loc=tf.zeros(10), scale=tf.ones(10), name="b_1")
    b_2 = Normal(loc=tf.zeros(1), scale=tf.ones(1), name="b_2")

    X = tf.placeholder(tf.float32, [N, D], name="X")

    def neural_network(X):
        h = tf.tanh(tf.matmul(X, W_0) + b_0)
        h = tf.tanh(tf.matmul(h, W_1) + b_1)
        h = tf.matmul(h, W_2) + b_2
        return tf.reshape(h, [-1])
    y = Normal(loc=neural_network(X), scale=0.1 * tf.ones(N), name="y")

# INFERENCE
with tf.name_scope("posterior"):
    with tf.name_scope("qW_0"):
        qW_0 = Normal(loc=tf.Variable(tf.random_normal([D, 10]), name="loc"),
                      scale=tf.nn.softplus(
                          tf.Variable(tf.random_normal([D, 10]), name="scale")))
    with tf.name_scope("qW_1"):
        qW_1 = Normal(loc=tf.Variable(tf.random_normal([10, 10]), name="loc"),
                      scale=tf.nn.softplus(
                          tf.Variable(tf.random_normal([10, 10]), name="scale")))
    with tf.name_scope("qW_2"):
        qW_2 = Normal(loc=tf.Variable(tf.random_normal([10, 1]), name="loc"),
                      scale=tf.nn.softplus(
                          tf.Variable(tf.random_normal([10, 1]), name="scale")))
    with tf.name_scope("qb_0"):
        qb_0 = Normal(loc=tf.Variable(tf.random_normal([10]), name="loc"),
                      scale=tf.nn.softplus(
                          tf.Variable(tf.random_normal([10]), name="scale")))
    with tf.name_scope("qb_1"):
        qb_1 = Normal(loc=tf.Variable(tf.random_normal([10]), name="loc"),
                      scale=tf.nn.softplus(
                          tf.Variable(tf.random_normal([10]), name="scale")))
    with tf.name_scope("qb_2"):
        qb_2 = Normal(loc=tf.Variable(tf.random_normal([1]), name="loc"),
                      scale=tf.nn.softplus(
                          tf.Variable(tf.random_normal([1]), name="scale")))

inference = ed.KLqp({W_0: qW_0, b_0: qb_0,
                     W_1: qW_1, b_1: qb_1,
                     W_2: qW_2, b_2: qb_2}, data={X: X_train, y: y_train})
inference.run(logdir='../dat/log')

#y_post = ed.copy(y, {W_0: qW_0, b_0: qb_0,
#                     W_1: qW_1, b_1: qb_1,
#                     W_2: qW_2, b_2: qb_2})
#
#print("Mean squared error on test data:")
#print(ed.evaluate('mean_squared_error', data={X: X_test, y_post: y_test}))
#
#print("Mean absolute error on test data:")
#print(ed.evaluate('mean_absolute_error', data={X: X_test, y_post: y_test}))
