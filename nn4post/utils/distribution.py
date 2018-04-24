#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Helper functions for gaining the trained distribution(s)."""

import numpy as np
import tensorflow as tf
from tensorflow.contrib.distributions import (
    Categorical, NormalWithSoftplusScale, Mixture)
try:
    from tensorflow.contrib.distributions import Independent
except:
    from nn4post.utils.independent import Independent
from nn4post.utils.euclideanization import get_parse_param



def get_trained_posterior(trained_var, param_shape):
  """
  Args:
    trained_var:
      `dict` object with keys contains "a", "mu", and "zeta", and values being
      either numpy arraies or TensorFlow tensors (`tf.constant`), as the value
      of the trained value of variables in "nn4post".
	
	param_shape:
      `dict` with keys the parameter-names and values the assocated shapes (as
      lists).

  Returns:
	Dictionary with keys the parameter-names and values instances of `Mixture`
    as the distributions that fit the associated posteriors.
  """

  n_c = trained_var['a'].shape[0]
  cat = Categorical(logits=trained_var['a'])

  parse_param = get_parse_param(param_shape)
  mu_list = [parse_param(trained_var['mu'][i]) for i in range(n_c)]
  zeta_list = [parse_param(trained_var['zeta'][i]) for i in range(n_c)]

  trained_posterior = {}

  for param_name in trained_var.keys():

    components = [
        Independent(NormalWithSoftplusScale(
            mu_list[i][param_name], zeta_list[i][param_name]))
        for i in range(n_c)
    ]
    mixture = Mixture(cat, components)
    trained_posterior[param_name] = mixture

  return trained_posterior
