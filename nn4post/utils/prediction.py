#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description
-----------
Main function `build_prediction`.
"""

import tensorflow as tf

from nn4post import get_trained_q
from nn4post.utils.vectorization import get_parse_param


def get_predictions_dict(q, model, param_shape, input_, n_samples):
  """
  Args:
    q:
      An instance of `Mixture`, as the trained inference distribution.

    model:
      Callable, implemented by TensorFlow, with
      Args:
        input_:
          `dict`, like `{'x_1': x_1, 'x_2': x_2}, with values Tensors.
        param:
          `dict`, like `{'w': w, 'b': b}, with values Tensors.
      Returns:
        `dict`, like `{'y': Y}`, where `Y` is an instance of `tf.Distribution`.
      As the model :math:`f(x, \theta)` in the documentation.

    param_shape:
      `dict` with keys the keys in `param_dict` and values the assocated
      shapes (as lists).

    input_:
      `dict`, like `{'x_1': x_1, 'x_2': x_2}, with values Tensors.

    n_samples:
      `int`, as the number of samples in Bayesian inference.

  Returns:
    `dict`, with the same keys as the output of `model`, and values the
    associated predictions (as `Op`) with shape `([n_samples] + batch_shape`
    where for each key, the `batch_shape` is that of the key's value (as an
    instance of `Distribution`) of the output of `model`.
  """

  # Sample models from the inference distribution `q`
  thetas = tf.unstack(q.sample(n_samples), axis=0)
  parse_param = get_parse_param(param_shape)
  params = [parse_param(_) for _ in thetas]

  # type: list of dict with keys output variable names and values
  #       distributions
  model_outputs = [
      # type: `dict`, like `{'y': Y}`
      # with `Y` instance of `Distribution`
      model(input_=input_, param=param)
      for param in params
  ]
  predictions_dict = {}
  output_names = list(model_outputs[0].keys())
  for name in output_names:
    predictions = [dist[name].sample() for dist in model_outputs]
    predictions = tf.stack(predictions, axis=0)
    predictions_dict[name] = predictions

  return predictions_dict


def build_prediction(trained_var, model, param_shape, input_,
                     n_samples=10, base_graph=None):
  """
  Args:
    trained_var:
      `dict` object with keys contains "a", "mu", and "zeta", and values
      being either numpy arraies or TensorFlow tensors (`tf.constant`),
      as the value of the trained value of variables in "nn4post".

    model:
      Callable, implemented by TensorFlow, with
      Args:
        input_:
          `dict`, like `{'x_1': x_1, 'x_2': x_2}, with values Tensors.
        param:
          `dict`, like `{'w': w, 'b': b}, with values Tensors.
      Returns:
        `dict`, like `{'y': Y}`, where `Y` is an instance of `tf.Distribution`.
      As the model :math:`f(x, \theta)` in the documentation.

      param_shape:
        `dict` with keys the keys in `param_dict` and values the assocated
        shapes (as lists).

    input_:
      `dict`, like `{'x_1': x_1, 'x_2': x_2}, with values Tensors.

    n_samples:
      `int`, as the number of samples in Bayesian inference, optional.

    base_graph:
      An instance of `tf.Graph`, as the base-graph the "prediction" scope is
      built onto, optional.

  Returns:
    `dict`, with the same keys as the output of `model`, and values the
    associated predictions (as `Op`) with shape `([n_samples] + batch_shape`
    where for each key, the `batch_shape` is that of the key's value (as an
    instance of `Distribution`) of the output of `model`.
  """

  graph = tf.get_default_graph() if base_graph is None else base_graph

  with graph.as_default():

    with tf.name_scope('prediction'):

      trained_q = get_trained_q(trained_var)
      predictions_dict = get_predictions_dict(
          trained_q, model, param_shape, input_, n_samples)

  return predictions_dict
