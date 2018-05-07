#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf


def convert_to_tensor(x):
  """Converts `x` to tensor, or simply returns `x` if it's an instance of
  `tf.Variable`."""
  if isinstance(x, tf.Variable):
    return x
  else:
    return tf.convert_to_tensor(x)
