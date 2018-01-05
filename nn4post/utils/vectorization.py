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
