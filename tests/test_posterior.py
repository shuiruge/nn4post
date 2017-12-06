#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description
-----------
Main function `get_log_posterior` and its helper `get_param_space_dim`.
"""


import numpy as np
import tensorflow as tf


# LIKELIHOOD

def get_log_likelihood(model, input, observed, scale=None):
    """
    Args:
        model:
            Callable, with
            Args:
                input:
                    `dict`, like `{'x_1': x_1, 'x_2': x_2}, with values
                    Tensors.
                param:
                    `dict`, like `{'w': w, 'b': b}, with values Tensors.
            Returns:
                `dict`, like `{'y': Y}`, where `Y` is an instance of
                `tf.Distribution`.

        input:
            `dict`, like `{'x_1': x_1, 'x_2': x_2}, with values Tensors.

        observed:
            `dict`, like `{'y': y}, with values Tensors.

        scale:
            Scalar with float dtype, or `None`, optional.

    Returns:
        Callable, with
        Args:
            param:
                `dict`, like `{'w': w, 'b': b}, with values Tensors.
        Returns:
            Scalar with float dtype.
    """

    _scale = 1.0 if scale is None else scale

    def log_likelihood(param):
        """:math:`\ln p(\theta \sim y_{\text{obs}}; x)`.

        Args:
            param:
                `dict`, like `{'w': w, 'b': b}, with values Tensors.

        Returns:
            Scalar.
        """

        model_output = model(input=input, param=param)
        assert model_output.keys() == observed.keys()

        output_var_names = model_output.keys()

        log_likelihood_tensor = 0.0
        for y in output_var_names:
            y_obs = observed[y]
            y_dist = model_output[y]
            log_likelihood_tensor += \
                _scale * tf.reduce_sum(y_dist.log_prob(y_obs))

        return log_likelihood_tensor

    return log_likelihood



# PRIOR

def get_log_prior(param_prior):
    """
    Args:
        param_prior:
            `dict`, like `{'w': Nomral(...), 'b': Normal(...)}, with values
            instances of `tf.distributions.Distribution`.

    Returns:
        Callable, with
        Args:
            param: The same argument in the `model`.
        Returns:
            Scalar.
    """

    def log_prior(param):
        """
        Args:
            param: The same argument in the `model`.
        Returns:
            Scalar.
        """

        log_priors = [qz.log_prob(param[z])
                    for z, qz in param_prior.items()]
        total_log_prior = tf.reduce_sum(
            [tf.reduce_sum(_) for _ in log_priors]
        )
        return total_log_prior

    return log_prior



# VECTORIZE

def get_parse_param(param_prior):
    """
    Args:
        param_prior:
            `dict`, like `{'w': Nomral(...), 'b': Normal(...)}, with values
            instances of `tf.distributions.Distribution`.

    Returns:
        Callable, with
        Args:
            theta:
            Tensor with shape `[param_space_dim]`, as one element in the
            parameter-space, obtained by flattening the `param` in the
            arguments of the `model`, and then concatenating by the order
            of the `param_names_in_order`.
        Returns:
            `dict` with keys the same as `param_prior`, and values Tensors with
            shape the same as the values of `param_prior`.
    """


    param_names_in_order = sorted(param_prior.keys())
    param_shapes = [param_prior[z].batch_shape.as_list()
                    for z in param_names_in_order]
    param_sizes = [np.prod(param_shape) for param_shape in param_shapes]

    def parse_param(theta):
        """
        Args:
            theta:
            Tensor with shape `[param_space_dim]`, as one element in the
            parameter-space, obtained by flattening the `param` in the
            arguments of the `model`, and then concatenating by the order
            of the `param_names_in_order`.
        Returns:
            `dict` with keys the same as `param_prior`, and values Tensors with
            shape the same as the values of `param_prior`.
        """
        splited = tf.split(theta, param_sizes)
        reshaped = [tf.reshape(flat, shape) for flat, shape
                    in list(zip(splited, param_shapes))]
        return {z: reshaped[i] for i, z in enumerate(param_names_in_order)}

    return parse_param


def get_param_space_dim(param_prior):
    """
    Args:
        param_prior:
            `dict`, like `{'w': Nomral(...), 'b': Normal(...)}, with values
            instances of `tf.distributions.Distribution`.

    Returns:
        `int` as the dimension of parameter-space.
    """
    param_names_in_order = sorted(param_prior.keys())
    param_shapes = [param_prior[z].batch_shape.as_list()
                    for z in param_names_in_order]
    param_sizes = [np.prod(param_shape) for param_shape in param_shapes]
    param_space_dim = sum(param_sizes)
    return param_space_dim



# POSTERIOR

def get_log_posterior(model, input_data, observed, param_prior, scale=None):

    log_likelihood = get_log_likelihood(
        model, input_data, {'y': y}, scale=scale)
    log_prior = get_log_prior(param_prior)
    parse_param = get_parse_param(param_prior)

    def log_posterior(theta):
        """
        Args:
            theta: Tensor with shape `[param_space_dim]`.
        Returns:
            Scalar.
        """
        param = parse_param(theta)
        return log_likelihood(param) + log_prior(param)

    return log_posterior



if __name__ == '__main__':

    """Test."""

    import os
    # -- `contrib` module in TF 1.3
    from tensorflow.contrib.distributions import NormalWithSoftplusScale
    from sklearn.utils import shuffle

    import sys
    sys.path.append('../sample/')
    from nn4post_advi import build_inference
    import mnist
    sys.path.append('../../')
    from tf_trainer import SimpleTrainer


    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # turn off the TF noise.


    # PARAMETERS
    N_C = 3
    NOISE_STD = 0.1
    BATCH_SIZE = 64


    # DATA
    mnist_ = mnist.MNIST(NOISE_STD, BATCH_SIZE)
    data_x, data_y, data_y_err = mnist_.training_data


    # MODEL
    n_inputs = 28 * 28  # number of input features.
    n_hiddens = 10  # number of perceptrons in the (single) hidden layer.
    n_outputs = 10  # number of perceptrons in the output layer.

    with tf.name_scope('data'):
        x = tf.placeholder(shape=[None, n_inputs], dtype=tf.float32, name='x')
        y = tf.placeholder(shape=[None, n_outputs], dtype=tf.float32, name='y')
        y_err = tf.placeholder(shape=y.shape, dtype=tf.float32, name='y_err')

    input_data = {'x': x, 'y_err': y_err}
    observed = {'y': y}

    def model(input, param):
        """ Shall be implemented by TensorFlow. This is an example, as a shallow
        neural network.

        Args:
            input:
                `dict`, like `{'x_1': x_1, 'x_2': x_2}, with values Tensors.
            param:
                `dict`, like `{'w': w, 'b': b}, with values Tensors.

        Returns:
            `dict`, like `{'y': Y}`, where `Y` is an instance of
            `tf.distributions.Distribution`.
        """
        # shape: `[None, n_hiddens]`
        hidden = tf.sigmoid(
            tf.matmul(input['x'], param['w_h']) + param['b_h'])
        # shape: `[None, n_outputs]`
        activation = tf.nn.softmax(
            tf.matmul(hidden, param['w_a']) + param['b_a'])
        Y = NormalWithSoftplusScale(activation, input['y_err'])
        return {'y': Y}


    # PRIOR
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

    param_prior = {
        'w_h': w_h, 'w_a': w_a,
        'b_h': b_h, 'b_a': b_a,
    }


    # POSTERIOR
    scale = mnist_.n_data / mnist_.batch_size
    log_posterior = get_log_posterior(
        model, input_data, observed, param_prior, scale=scale)


    # INFERENCE
    param_space_dim = get_param_space_dim(param_prior)
    ops, gvs = build_inference(N_C, param_space_dim, log_posterior)


    # TRAIN
    batch_generator = mnist_.batch_generator()
    def get_feed_dict_generator():
        while True:
            data_x, data_y, data_y_err = next(batch_generator)
            yield {x: data_x, y: data_y, y_err: data_y_err}
    trainer = SimpleTrainer(
        loss=ops['loss'],
        gvs=gvs,
        optimizer=tf.train.RMSPropOptimizer(0.005),
        #logdir='../dat/logs',
        #dir_to_ckpt='../dat/checkpoints/nn4post_advi_on_mnist/',
    )
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
