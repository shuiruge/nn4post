#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Helper functions for getting `log_posterior` (as a callable)."""

import tensorflow as tf

from nn4post.utils.vectorization import get_param_shape, get_parse_param




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


