#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description
-----------
Main function `get_log_posterior` and its helper `get_param_space_dim`.

An example locates in the end.
"""


import numpy as np
import tensorflow as tf
from tensorflow.contrib.distributions import Distribution


# LIKELIHOOD

def get_log_likelihood(model, input_, observed, scale=None):
  """
  Args:
    model:
      Callable, implemented by TensorFlow, with
      Args:
        input_:
          `dict`, like `{'x_1': x_1, 'x_2': x_2}, with values Tensors.
        param:
          `dict`, like `{'w': w, 'b': b}, with values Tensors.
      Returns:
        `dict`, like `{'y': Y}`, where `Y` is an instance of `tf.Distribution`.

    input_:
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

    with tf.name_scope('log_likelihood'):

      model_output = model(input_=input_, param=param)
      assert model_output.keys() == observed.keys()

      output_var_names = model_output.keys()

      log_likelihood_tensor = 0.0
      for y in output_var_names:
        y_obs = observed[y]
        y_dist = model_output[y]
        log_likelihood_tensor += _scale \
            * tf.reduce_sum( y_dist.log_prob(y_obs) )

    return log_likelihood_tensor

  return log_likelihood



# PRIOR

def get_log_prior(param_prior):
  """
  Args:
    param_prior:
      `dict`, like `{'w': Nomral(...), 'b': Normal(...)}, with values instances
      of `tf.distributions.Distribution`.

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

    with tf.name_scope('log_prior'):

      log_priors = [qz.log_prob(param[z])
                    for z, qz in param_prior.items()]
      total_log_prior = tf.reduce_sum(
          [tf.reduce_sum(_) for _ in log_priors] )

    return total_log_prior

  return log_prior



# VECTORIZE

def get_size(shape):
  """Get tensor size from tensor shape.

  Args:
    shape:
      List of `int`s, as the shape of tensor.

  Returns:
    `int` as the size of the tensor.
  """
  return np.prod(shape)


def get_param_shape(param_dict):
  """
  Args:
    param_dict:
      `dict`, like `{'w': ..., 'b': ...}, with values being either instances
      of `tf.contrib.distributions.Distribution`, or any objects that have
      `shape` attribute (e.g. numpy array or TensorFlow tensor).

  Returns:
    `dict` with keys the keys in `param_dict` and values the assocated shapes
    (as lists).
  """
  param_shape = {
      name:
          val.batch_shape.as_list() \
          if isinstance(val, Distribution) \
          else val.shape.as_list()
      for name, val in param_dict.items()
  }
  return param_shape


def get_parse_param(param_shape):
  """
  Args:
    param_shape:
      `dict` with keys the keys in `param_dict` and values the assocated shapes
      (as lists).

  Returns:
    Callable, with
    Args:
      theta:
        Tensor with shape `[param_space_dim]`, as one element in the
        parameter-space, obtained by flattening the `param` in the arguments
        of the `model`, and then concatenating by the order of the
        `param_names_in_order`.
    Returns:
      `dict` with keys the same as `param_shape`, and values Tensors with shape
      the same as the values of `param_shape`.
  """

  # Get list of shapes and sizes of parameters ordered by name
  param_names_in_order = sorted(param_shape.keys())
  param_shapes = [param_shape[z] for z in param_names_in_order]
  param_sizes = [get_size(shape) for shape in param_shapes]

  def parse_param(theta):
    """
    Args:
      theta:
        Tensor with shape `[param_space_dim]`, as one element in the
        parameter-space, obtained by flattening the `param` in the arguments
        of the `model`, and then concatenating by the order of the
        `param_names_in_order`.
    Returns:
      `dict` with keys the same as `param_shape`, and values Tensors with shape
       the same as the values of `param_shape`.
    """

    with tf.name_scope('parse_parameter'):

      splited = tf.split(theta, param_sizes)
      reshaped = [tf.reshape(flat, shape) for flat, shape
                  in list(zip(splited, param_shapes))]

    return {z: reshaped[i] for i, z in enumerate(param_names_in_order)}

  return parse_param


def get_param_space_dim(param_shape):
  """
  Args:
    param_shape:
      `dict` with keys the keys in `param_dict` and values the assocated shapes
      (as lists).

  Returns:
    `int` as the dimension of parameter-space.
  """
  param_sizes = [get_size(shape) for shape in param_shape.values()]
  param_space_dim = sum(param_sizes)
  return param_space_dim



# POSTERIOR

def get_log_posterior(model, input_, observed, param_prior,
                      scale=None, base_graph=None):
  """
  Args:
    model:
      Callable, with
      Args:
        input_:
          `dict`, like `{'x_1': x_1, 'x_2': x_2}, with values Tensors.
        param:
          `dict`, like `{'w': w, 'b': b}, with values Tensors.
      Returns:
        `dict`, like `{'y': Y}`, where `Y` is an instance of `tf.Distribution`.
      Shall be implemented by TensorFlow.

    input_:
      `dict`, like `{'x_1': x_1, 'x_2': x_2}, with values Tensors.

    observed:
      `dict`, like `{'y': y}, with values Tensors.

    param_prior:
      `dict`, like `{'w': Nomral(...), 'b': Normal(...)}, with values instances
      of `tf.distributions.Distribution`.

    scale:
      Scalar with float dtype, or `None`, optional.

    base_graph:
      An instance of `tf.Graph`, as the base-graph the "prediction" scope is
      built onto, optional.

  Returns:
    Callable, with
    Args:
      theta:
        Tensor with shape `[param_space_dim]`.
    Returns:
      Scalar.
  """

  if base_graph is None:
    graph = tf.get_default_graph()
  else:
    graph = base_graph

  with graph.as_default():

    log_likelihood = get_log_likelihood(
        model, input_, observed, scale=scale)
    log_prior = get_log_prior(param_prior)
    param_shape = get_param_shape(param_prior)
    parse_param = get_parse_param(param_shape)

    def log_posterior(theta):
      """
      Args:
        theta:
          Tensor with shape `[param_space_dim]`.
      Returns:
        Scalar.
      """

      with tf.name_scope('log_posterior'):

        param = parse_param(theta)
        result = log_likelihood(param) + log_prior(param)

      return result

  return log_posterior



if __name__ == '__main__':

    """Example: apply onto MNIST data-set with a shallow neural network."""

    import os
    # -- `contrib` module in TF 1.3
    from tensorflow.contrib.distributions import (
        Normal, NormalWithSoftplusScale, Categorical,
    )
    from sklearn.utils import shuffle

    import sys
    sys.path.append('../sample/')
    from build_nn4post import build_nn4post
    from build_prediction import build_prediction
    import mnist
    sys.path.append('../../')
    from tf_trainer import SimpleTrainer


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
    n_iters = 30000
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
    print(voted_predictions.shape)

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
    print(accuracy)

