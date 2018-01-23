#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Helper functions for comparing the results of fitting by `nn4post`."""


import tensorflow as tf
from tensorflow.contrib.distributions import Distribution

from nn4post.utils.distribution import get_trained_q
from nn4post.utils.euclideanization import get_parse_param



def get_kl_divergence_between_inferences(qs,
      n_samples=100, name='kl_divergence_between_inferences'):
  """Returns :math:`KL( q_1 || q_0 )` where the  `qs` as `(q_0, q_1)`."""

  # Check type
  if len(qs) != 2:
    raise TypeError(
        'Argument `qs` should be an iterable of two `Distribution`s.')
  for q in qs:
    if not isinstance(q, Distribution):
      raise TypeError('Elements in argument `qs` should be `Distribution`.')

  with tf.name_scope(name):
  
    # shape: `[n_samples, n_d]`
    thetas = qs[1].sample(n_samples)

    def log_prob_difference(theta):
      """:math:`log[ q_1(x) / q_0(x) ]`"""
      # shape: `[]`
      return qs[1].log_prob(theta) - qs[0].log_prob(theta)
    
    # shape: `[]`
    kl_divergence_between_inferences = tf.reduce_mean(
        # shape: `[n_samples]`
        tf.map_fn(log_prob_difference, thetas))

  return kl_divergence_between_inferences


def get_delta_kl_divergence(log_posterior_upto_const, qs, max_exp_arg=10.0):
  """Compare the two inference-distributions. Explicitly, what we are to
  compute is the KL-divergence between `p(x)` and `q_0(x)`, but with the
  Monte-Carlo integral sampled from `q_1(x)`.

  For this, from ...:math:

        \text{KL-divergence} ( q_0 || p )
      = \int dx q_0(x) log[ q_0(x) / p(x) ] ]
      = E_{q_0}[ log[ q_0(x) / p(x) ] ]

  and ...:math:
          \int dx q_0(x) log[ q_0(x) / p(x) ] ]
        = \int dx q_1(x) q_0(x) / q_1(x) * log[ q_0(x) / p(x) ] ]
        = E_{q_1}[ q_0(x) / q_1(x) * log[ q_0(x) / p(x) ] ]

  we find what we shall compute is ...:math:

          \text{KL-divergence} ( q_0 || p )
        = E_{q_1}[ q_0(x) / q_1(x) * log[ q_0(x) / p(x) ] ]


  Args:
    log_posterior_upto_const:
      Callable from tensor of the shape `[n_d]` to scalar, both with the same
      dtype as the `dtype` argument, as the logorithm of the posterior up to
      a constant.

    qs:
      Tuple of two inference-distributions, i.e. the `(q_0, q_1)`.

    max_exp_arg:
      `float`, for numerical stability in `tf.exp()`, optional.

  Returns:
    Scalar, as the KL-divergence between `log_posterior_upto_const` and `qs[0]`
    but with the Monte-Carlo integral sampled from `qs[1]`.
  """

  # Check type
  if len(qs) != 2:
    raise TypeError(
        'Argument `qs` should be an iterable of two `Distribution`s.')
  for q in qs:
    if not isinstance(q, Distribution):
      raise TypeError('Elements in argument `qs` should be `Distribution`.')

  with tf.name_scope(name):
  
    # shape: `[n_samples, n_d]`
    thetas = qs[1].sample(n_samples)

    def prob_ratio(theta):
      """:math:`q_0(x) / q_1(x) = exp[ log[q_0(x)] - log[q_1(x)] ]`"""
      # shape: `[]`
      exp_arg = qs[0].log_prob(theta) - qs[1].log_prob(theta)
      # For numerical stability
      exp_arg = tf.clip_by_value(exp_arg, -max_exp_arg, max_exp_arg)
      return tf.exp(exp_arg)

    def log_prob_difference(theta):
      """:math:`log[ q_0(x) / p(x) ] ]`"""
      # shape: `[]`
      return qs[0].log_prob(theta) - log_posterior_upto_const(theta)

    def integrand(theta):
      """:math:`q_0(x) / q_1(x) * log[ q_0(x) / p(x) ]`"""
      # shape: `[]`
      return prob_ratio(theta) * log_prob_difference(theta)
    
    # shape: `[]`
    kl_divergence_between_inferences = tf.reduce_mean(
        # shape: `[n_samples]`
        tf.map_fn(log_prob_difference, thetas))

  return kl_divergence_between_inferences


