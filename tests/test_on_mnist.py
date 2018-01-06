#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Example: apply onto MNIST data-set with a shallow neural network."""

import os
import sys
import numpy as np
import tensorflow as tf
# -- `contrib` module in TF 1.3
from tensorflow.contrib.distributions import (
    NormalWithSoftplusScale, Categorical,
)
from sklearn.utils import shuffle

from nn4post import build_nn4post
from nn4post.utils import get_param_shape, get_param_space_dim
from nn4post.utils.posterior import get_log_posterior
from nn4post.utils.prediction import build_prediction
from nn4post.utils.tf_trainer import SimpleTrainer

import mnist


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # turn off the TF noise.


# PARAMETERS
N_C = 1
NOISE_STD = 0.0
BATCH_SIZE = 64


# DATA
mnist_ = mnist.MNIST(NOISE_STD, BATCH_SIZE)


# MODEL
n_inputs = 28 * 28  # number of input features.
n_hiddens = 200  # number of perceptrons in the (single) hidden layer.
n_outputs = 10  # number of perceptrons in the output layer.

with tf.name_scope('data'):
    x = tf.placeholder(shape=[None, n_inputs], dtype=tf.float32, name='x')
    y = tf.placeholder(shape=[None], dtype=tf.int32, name='y')

input_ = {'x': x}
observed = {'y': y}

def model(input_, param):
    """ Shall be implemented by TensorFlow. This is an example, as a shallow
    neural network.

    Args:
        input_:
            `dict`, like `{'x_1': x_1, 'x_2': x_2}, with values Tensors.
        param:
            `dict`, like `{'w': w, 'b': b}, with values Tensors.

    Returns:
        `dict`, like `{'y': Y}`, where `Y` is an instance of
        `tf.distributions.Distribution`.
    """
    # shape: `[None, n_hiddens]`
    hidden = tf.sigmoid(
        tf.matmul(input_['x'], param['w_h']) + param['b_h'])
    # shape: `[None, n_outputs]`
    logits = tf.matmul(hidden, param['w_a']) + param['b_a']

    Y = Categorical(logits=logits)
    return {'y': Y}


# PRIOR
with tf.name_scope('prior'):
    w_h = NormalWithSoftplusScale(
        loc=tf.zeros([n_inputs, n_hiddens]),
        scale=tf.ones([n_inputs, n_hiddens]) * 10,
        name="w_h")
    w_a = NormalWithSoftplusScale(
        loc=tf.zeros([n_hiddens, n_outputs]),
        scale=tf.ones([n_hiddens, n_outputs]) * 10,
        name="w_a")
    b_h = NormalWithSoftplusScale(
        loc=tf.zeros([n_hiddens]),
        scale=tf.ones([n_hiddens]) * 100,
        name="b_h")
    b_a = NormalWithSoftplusScale(
        loc=tf.zeros([n_outputs]),
        scale=tf.ones([n_outputs]) * 100,
        name="b_a")

param_prior = {
    'w_h': w_h, 'w_a': w_a,
    'b_h': b_h, 'b_a': b_a,
}


# POSTERIOR
scale = mnist_.n_data / mnist_.batch_size
log_posterior = get_log_posterior(
    model, input_, observed, param_prior, scale=scale)


# INFERENCE
param_shape = get_param_shape(param_prior)
param_space_dim = get_param_space_dim(param_shape)
print('\n-- Dimension of parameter-space: {}.\n'.format(param_space_dim))

ops, gvs = build_nn4post(N_C, param_space_dim, log_posterior)


# TRAIN
batch_generator = mnist_.batch_generator()
def get_feed_dict_generator():
    while True:
        x_train, y_train, y_err_train = next(batch_generator)
        y_train = np.argmax(y_train, axis=1)
        yield {x: x_train, y: y_train}
trainer = SimpleTrainer(
    loss=ops['loss'],
    gvs=gvs,
    optimizer=tf.train.AdamOptimizer(0.005),
    logdir='../dat/logs/nn4post_advi_on_mnist',
    dir_to_ckpt='../dat/checkpoints/nn4post_advi_on_mnist/',
)
#n_iters = 30000
n_iters = 30  # test!
feed_dict_generator = get_feed_dict_generator()
trainer.train(n_iters, feed_dict_generator)


# PREDICTION

# Get test data of MNIST
x_test, y_test, y_err_test = mnist_.test_data
# Adjust to the eagered form
y_test = y_test.astype('int32')


# Get the trained variables.
var_names = ['a', 'mu', 'zeta']
trained_var = {
    name:
        trainer.sess.run(ops[name])
    for name in var_names
}
print('a: ', trained_var['a'])
print('zeta mean: ', np.mean(trained_var['zeta']))
print('zeta std: ', np.std(trained_var['zeta']))

predictions_dict = build_prediction(
    trained_var, model, param_shape, input_, n_samples=100)
predictions = tf.stack(predictions_dict['y'], axis=0)

with tf.Session() as sess:
    feed_dict = {x: x_test}
    # shape: `[n_samples, n_data]`
    predictions = sess.run(predictions, feed_dict=feed_dict)

def get_most_freq(array):
    index = np.argmax(np.bincount(array))
    return index
# shape `[n_data]`
voted_predictions = np.array([
    get_most_freq(predictions[:,i])
    for i in range(predictions.shape[1])
])

def get_accuracy(xs, ys):
    """
    Args:
        xs:
            Numpy array.
        ys:
            Numpy array with the same shape and dtype as `xs`.

    Returns:
        `float` as the perception of `x == y` in `xs`, where `x` and `y` in
        `xs` and `ys` respectively and one-to-one correspondently.
    """
    assert xs.dtype == ys.dtype
    n_correct = 0
    n_mistake = 0
    for x, y in list(zip(xs, ys)):
        if x == y:
            n_correct += 1
        else:
            n_mistake += 1
    return n_correct / (n_correct + n_mistake)

targets = y_test  # shape `[n_data]`, dtype `int32`.
voted_predictions = voted_predictions.astype('int32')
accuracy = get_accuracy(voted_predictions, targets)
print('Accuracy: ', accuracy)
