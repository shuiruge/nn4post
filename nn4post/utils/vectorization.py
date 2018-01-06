#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Helper functions for vectorization."""

import numpy as np
import tensorflow as tf
from tensorflow.contrib.distributions import Distribution


def get_size(shape):
  """Get tensor size from tensor shape.

  Args:
    shape:
      List of `int`s, as the shape of tensor.

  Returns:
    `int` as the size of the tensor.
  """
  return int(np.prod(shape))


def get_param_shape(param_dict):
  """Parse shapes from instances of `np.array`, `tf.Tensor`, or `Distribution`.

  Args:
    param_dict:
      `dict`, like `{'w': ..., 'b': ...}, with values being instances of
      `np.array`, `tf.Tensor`, or `Distribution`.

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
  """Returns a parser of parameters that parses a tensor of the shape `[n_d]`,
  which is an element in the Euclidean parameter-space :math:`\mathbb{R}^{n_d}`,
  to the tensors of the shapes provided by `param_shape`.

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



def vectorize(param_shape):
  """Returns a decorator that vectorize a function, implemented by TensorFlow,
  on general parameter-space to that on the associated Euclidean parameter-
  space.

  Example:
    ```python:

      param_shape = {'param_1': [2, 5], 'param_2': [3], ...}

      @vectorize(param_shape)
      def fn(param_1, param_2, ...):
          '''
          Args:
              param_1:
                  An instance of `tf.Tensor` with shape `[2, 5]`.

              param_2:
                  An instance of `tf.Tensor` with shape `[3]`.

              ...

          Returns:
              Any.
          '''

          # Your implementation.
    ```

  Args:
    param_shape:
      `dict` with keys the keys in `param_dict` and values the assocated shapes
      (as lists).

  Returns:
    A decorator.
  """

  parse_param = get_parse_param(param_shape)

  def decorator(fn):

    def vectorized_fn(euclidean_param):

      param = parse_param(euclidean_param)

      return fn(**param)

    return vectorized_fn

  return decorator
