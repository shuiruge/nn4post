#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description
-----------
Shallow neural network by `Edward ` on MNIST dataset.
"""


import numpy as np
import tensorflow as tf
# -- `contrib` module in TF 1.3
from tensorflow.contrib.distributions import NormalWithSoftplusScale
from sklearn.utils import shuffle
import pickle

import os
import sys
sys.path.append('../sample/')
sys.path.append('../../')
from tools import Timer
import mnist
from nn4post_advi import build_inference
from tf_trainer import SimpleTrainer



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # turn off the TF noise.


# PARAMETERS
N_C = 3
NOISE_STD = 0.1
BATCH_SIZE = 64


# DATA
mnist_ = mnist.MNIST(NOISE_STD, BATCH_SIZE)
data_x, data_y, data_y_err = mnist_.training_data

print(data_x.shape)
print(data_y.shape)
print(data_y_err.shape)


# MODEL
n_inputs = 28 * 28  # number of input features.
n_hiddens = 10  # number of perceptrons in the (single) hidden layer.
n_outputs = 10  # number of perceptrons in the output layer.

with tf.name_scope('data'):
    x = tf.placeholder(shape=[None, n_inputs], dtype=tf.float32, name='x')
    y = tf.placeholder(shape=[None, n_outputs], dtype=tf.float32, name='y')
    y_err = tf.placeholder(shape=y.shape, dtype=tf.float32, name='y_err')

input_data = {'x': x}
output_data = {'y': (y, y_err)}

def model(inputs, params):
    """ Shall be implemented by TensorFlow. This is an example, as a shallow
    neural network.

    Args:
        inputs:
            `dict`, like `{'x_1': x_1, 'x_2': x_2}, with values Tensors.
        params:
            `dict`, like `{'w': w, 'b': b}, with values Tensors.

    Returns:
        Tensor.
    """
    # shape: `[None, n_hiddens]`
    hidden = tf.sigmoid(
        tf.matmul(inputs['x'], params['w_h']) + params['b_h'])
    # shape: `[None, n_outputs]`
    activation = tf.nn.softmax(
        tf.matmul(hidden, params['w_a']) + params['b_a'])
    return {'y': activation}

with tf.name_scope('prior'):
    w_h = NormalWithSoftplusScale(
        loc=tf.zeros([n_inputs, n_hiddens]),
        scale=tf.ones([n_inputs, n_hiddens]),
        name="w_h")
    w_a = NormalWithSoftplusScale(
        loc=tf.zeros([n_hiddens, n_outputs]),
        scale=tf.ones([n_hiddens, n_outputs]),
        name="w_a")
    b_h = NormalWithSoftplusScale(
        loc=tf.zeros([n_hiddens]),
        scale=tf.ones([n_hiddens]) * 100,
        name="b_h")
    b_a = NormalWithSoftplusScale(
        loc=tf.zeros([n_outputs]),
        scale=tf.ones([n_outputs]) * 100,
        name="b_a")

latent_vars = {
    'w_h': w_h, 'w_a': w_a,
    'b_h': b_h, 'b_a': b_a,
}


# POSTERIOR

#scale = tf.placeholder(shape=[], dtype=tf.float32, name='scale')
scale = mnist_.n_data / mnist_.batch_size

def chi_square(model_outputs):
    """ (Temporally) assume that the data obeys a normal distribution,
    realized by Gauss's limit-theorem.

    Args:
        model_outputs: `dict`.
    Returns:
        Scalar.
    """
    un_scaled_val = 0.0
    for y in output_data.keys():
        y_val, y_err = output_data[y]
        normal = NormalWithSoftplusScale(loc=y_val, scale=y_err)
        un_scaled_val += tf.reduce_sum(normal.log_prob(model_outputs[y]))
    return scale * un_scaled_val


def log_likelihood(params):
    """
    Args:
        params: The same argument in the `model`.
    Returns:
        Scalar.
    """
    return chi_square(model(inputs=input_data, params=params))


def log_prior(params):
    """
    Args:
        params: The same argument in the `model`.
    Returns:
        Scalar.
    """

    log_priors = [qz.log_prob(params[z])
                for z, qz in latent_vars.items()]
    total_log_prior = tf.reduce_sum(
        [tf.reduce_sum(_) for _ in log_priors]
    )
    return total_log_prior


param_names_in_order = sorted(latent_vars.keys())
param_shapes = [latent_vars[z].batch_shape.as_list()
                for z in param_names_in_order]
param_sizes = [np.prod(param_shape) for param_shape in param_shapes]
param_space_dim = sum(param_sizes)
print(' --- Parameter-space Dimension: {0}'.format(param_space_dim))

def parse_params(theta):
    """
    Args:
        theta:
        Tensor with shape `[param_space_dim]`, as one element in the
        parameter-space, obtained by flattening the `params` in the
        arguments of the `model`, and then concatenating by the order
        of the `param_names_in_order`.
    Returns:
        `dict` with keys the same as `latent_vars`, and values Tensors with
        shape the same as the values of `latent_vars`.
    """
    splited = tf.split(theta, param_sizes)
    reshaped = [tf.reshape(flat, shape) for flat, shape
                in list(zip(splited, param_shapes))]
    return {z: reshaped[i] for i, z in enumerate(param_names_in_order)}


def log_posterior(theta):
    """
    Args:
        theta: Tensor with shape `[param_space_dim]`.
    Returns:
        Scalar.
    """
    params = parse_params(theta)
    return log_likelihood(params) + log_prior(params)


ops, gvs = build_inference(N_C, param_space_dim, log_posterior)

#train_op = TrainOp(ops['loss'], tf.train.AdamOptimizer(0.01))
#trainer = Trainer([train_op], tensorboard_dir='../dat/logs')
#
#trainer.fit(
#    {x: data_x, y: data_y, y_err: data_y_err}
#)


batch_generator = mnist_.batch_generator()
def get_feed_dict_generator():
    while True:
        data_x, data_y, data_y_err = next(batch_generator)
        yield {x: data_x, y: data_y, y_err: data_y_err}
trainer = SimpleTrainer(
    loss=ops['loss'],
    gvs=gvs,
    optimizer=tf.train.RMSPropOptimizer(0.03),
    logdir='../dat/logs',
    dir_to_ckpt='../dat/checkpoints/nn4post_advi_on_mnist/')
n_iters = 1000
feed_dict_generator = get_feed_dict_generator()
trainer.train(n_iters, feed_dict_generator)

#with tf.Session() as sess:
#
#    sess.run(tf.global_variables_initializer())
#
#    n_iter = 5000
#    for i in range(n_iter):
#
#        data_x, data_y, data_y_err = next(batch_generator)
#        feed_dict = {x: data_x, y: data_y, y_err: data_y_err}
#        _, loss_val = sess.run([ train_op, ops['loss'] ],
#                               feed_dict=feed_dict)
#
#        if i % 100 == 0:
#            print(i, loss_val)
#
#    print(sess.run(ops['c']))
