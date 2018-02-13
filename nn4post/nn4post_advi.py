#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description
-----------
ADVI implementation of "nerual network for posterior" with accurate entropy of
q-distribution.


TF version
----------
Tested on TF 1.4.0.
"""


import numpy as np
import tensorflow as tf
from tensorflow.contrib.distributions import Categorical, Normal, Mixture
try:
    from tensorflow.contrib.distributions import Independent
except:
    from nn4post.utils.independent import Independent
from tensorflow.contrib.framework import is_tensor



def get_gaussian_mixture_log_prob(cat_probs, gauss_mu, gauss_sigma):
  r"""Get the logrithmic p.d.f. of a Gaussian mixture model.

  Args:
    cat_probs:
      `1-D` tensor with unit (reduce) sum, as the categorical probabilities.

    gauss_mu:
      List of tensors, with the length the shape of `cat_probs`, as the `mu`
      values of the Gaussian components. All these tensors shall share the
      same shape (as, e.g., `gauss_mu[0]`)

    gauss_sigma:
      List of tensors, with the length the shape of `cat_probs`, as the `sigma`
      values of the Gaussian components. Thus shall be all positive, and shall
      be all the same shape as `gauss_mu[0]`.

  Returns:
    Callable, mapping from tensor of the shape of `gauss_mu[0]` to scalar, as
    the p.d.f..
  """
  with tf.name_scope('gaussian_mixture_log_prob'):

    n_cats = cat_probs.shape[0]
    cat = Categorical(probs=cat_probs)
    components = [
        Independent( Normal(gauss_mu[i], gauss_sigma[i]) )
        for i in range(n_cats)
    ]
    gaussian_mixture = Mixture(cat=cat, components=components)
    gaussian_mixture_log_prob = gaussian_mixture.log_prob

  return gaussian_mixture_log_prob



def get_wall(wall_position, wall_slope):
  r"""Get a "wall-function" in space of any dimensionality. Along any dimension,
  for any position on the left of wall-position, the hight of the wall is zero,
  otherwise being `wall_slope * (position - wall_position)`, as a wall shall
  be in the world.

  We use softplus to implement the wall.

  Args:
    wall_position:
      Tensor, as the position of the wall. Type `float` is also valid for
      representing scalar.

    wall_slope:
      Tensor of the same shape and dtype as `wall_position`, as the slope of the
      wall. Type `float` is also valid for representing scalar.

  Returns:
    Callable, with
    Args:
      x:
        `Tensor`.
    Returns:
      `Tensor` with the same shape and dtype as `x`, as the hight of the wall
      at position `x`.
  """

  def wall(x):
    with tf.name_scope('wall'):
      wall_val = tf.nn.softplus( wall_slope * (x - wall_position) )
    return wall_val

  return wall



class InferenceBuilder(object):
  r"""Builder for the inference by nn4post. C.f. the documentation in
  "/docs/main.pdf".
   
  Methods:
    make_loss_and_gradients:
      `Python operation` that accepts the `Tensor`s `a`, `mu`, `zeta` as
      arguments, and returns loss and its gradients to the arguments.
  """

  def __init__(self, n_c, n_d, log_posterior_upto_const, n_samples=100,
               r=1.0, beta=1.0, max_a_range=10, wall_slope=10, epsilon=1e-08,
               dtype='float32'):
    r"""Initialize the builder.

    This initialization draws NONE on the TensorFlow graph. So, you can safely
    call `InferenceBuilder(...)` in the outside of `your_graph.as_default()`.

    Args:
      n_c:
        `int`, as the number of categorical probabilities, i.e. the :math:`N_c`
        in the documentation.

      n_d:
        `int`, as the number of dimension, i.e. the :math:`N_d` in the
        documentation.

      log_posterior_upto_const:
        Callable from tensor of the shape `[n_d]` to scalar, both with the same
        dtype as the `dtype` argument, as the logorithm of the posterior up to
        a constant.

      init_var:
        `dict` for setting the initial values of variables. optional. It has
        keys `'a'`, `'mu'`, and `'zeta'`, and values of numpy arraies or tensors
        of the shapes `[n_c]`, `[n_c, n_d]`, and `[n_c, n_d]`, respectively. All
        these values shall be the same dtype as the `dtype` argument.

      n_samples:
        `int` or `tf.placeholder` with scalar shape and `int` dtype, as the
        number of samples in the Monte Carlo integrals, optional.

      r:
        `float` or `tf.placeholder` with scalar shape and `dtype` dtype, as the
        rescaling factor of `a`, optional.

      beta:
        `float` or `tf.placeholder` with scalar shape and `dtype` dtype, as the
        "smooth switcher" :math:`\partial \mathcal{L} / \partial z_i` in the
        documentation, optional.

      max_a_range:
        `float` or `tf.placeholder` with scalar shape and `dtype` dtype, as the
        bound of `max(a) - min(a)`, optional.

      wall_slope:
        `float` or `tf.placeholder` with scalar shape and `dtype` dtype, as the
        slope-parameter in the wall-function in the regularization of loss,
        which bounds the maximum value of the range of `a`, optional.

        NOTE:
          The only restirction to this parameter is that `wall_slope` shall be
          much greater than unit. But when learning-rate of optimizer is not
          small enough (as generally demanded in the early stage of training),
          extremely great value of `wall_slope` will triger `NaN`.

      epsilon:
        `float` or `tf.placeholder` with scalar shape and `dtype` dtype, as the
        :math:`epsilon` in the documentation, optional.

      dtype:
        `str`, as the dtype of floats employed herein, like `float32`, etc.,
        optional.
    """

    self.n_c = n_c
    self.n_d = n_d
    self.log_posterior_upto_const = log_posterior_upto_const

    # Configurations
    self.n_samples = n_samples
    self.r = r
    self.beta = beta
    self.max_a_range = max_a_range
    self.wall_slope = wall_slope
    self.epsilon = epsilon
    self.dtype = dtype


  def check_arguments(self, a, mu, zeta):
    r"""Helper function of the method `self.make_loss_and_gradients`. Checks
    the types, shapes, and dtypes of the arguments `a`, `mu`, and `zeta`.
    Raises `TypeError` if the check is not passed.

    This replaces `tf.convert_to_tensor`, since this function also converts
    `Variable` to non-variable `Tensor`, which is not what we expect.
    """

    with tf.name_scope('check_args'):

      # Check tensors
      msg = 'Arguemnt `{}` should be tensor-like.'
      for _ in (a, mu, zeta):
        if not is_tensor(_):
          raise TypeError(msg.format(_.name))

      # Check dtypes
      msg = 'Argument `{}` should have the dtype {}, but now {}.'
      for _ in (a, mu, zeta):
        if _.dtype not in (self.dtype, self.dtype + '_ref'):
          raise TypeError(msg.format(_.name, self.dtype, _.dtype))

      # Check shapes
      msg = 'Arguemnt `{}` should have the shape {}, but now {}.'
      shape = {
          a: [self.n_c],
          mu: [self.n_c, self.n_d],
          zeta: [self.n_c, self.n_d]
      }
      for _ in (a, mu, zeta):
        if _.shape != shape[_]:
          raise TypeError(msg.format(_.name, shape[_], _.shape))


  def make_loss_and_gradients(self, a, mu, zeta, name=None):
    r"""This function (or say, a `tf.Operation` constructor) implements (c.f.
    the documentation):

    ```tex:
    $$`\mathcal{L} ( a, \mu, \zeta )$$

    and, if denote $z:=(a, \mu, \zeta)$,

    $$\frac{ \partial \mathcal{L} } }{ \partial z } ( a, \mu, \zeta ),$$

    and return them altogether.
    ```

    Args:
      a:
        `Tensor`, with shape `[n_c]` and dtype `dtype`. Expected to be an 
        instance of `tf.Variable`, but can be not so.

      mu:
        `Tensor`, with shape `[n_c, n_d]` and dtype `dtype`. Expected to be
        an instance of `tf.Variable`, but can be not so.

      zeta:
        `Tensor`, with shape `[n_c, n_d]` and dtype `dtype`. Expected to be
        instance of `tf.Variable`, but can be not so.

      name:
        `str` or `None`, as the main name-scope, optional.

    Returns:
      Tuple of two elements. The first is the loss, :math:`\mathcal{L}`;
      and the second is a list of pairs of gradients, with the type as the
      returned by calling `tf.gradients` with arguments `[a, mu, zeta]`.

    Raises:
      TypeError:
        If the arguments do not match their types, shapes, or dtypes.

    Examples:
      >>> n_c = 3
      >>> n_d = 1
      >>> # Gaussian mixture distribution as the :math:`p`
      >>> cat_probs = np.array([0.2, 0.8], dtype='float32')
      >>> gauss_mu = np.array([[-1.0], [1.0]], dtype='float32')
      >>> gauss_sigma = np.array([[1.0], [1.0]], dtype='float32')
      >>> log_p = get_gaussian_mixture_log_prob(
      ...             cat_probs, gauss_mu, gauss_sigma)
      >>> # Then build up the inference
      >>> ib = InferenceBuilder(
      ...          n_c=n_c, n_d=n_d, log_posterior_upto_const=log_p)
      >>> a = tf.Variable(np.zeros([n_c]), dtype='float32')
      >>> mu = tf.Variable(np.zeros([n_c, n_d]), dtype='float32')
      >>> zeta = tf.Variable(np.zeros([n_c, n_d]), dtype='float32')
      >>> loss, gradients = ib.make_loss_and_gradients(a, mu, zeta)
    """

    with tf.name_scope(name, 'nn4post', [a, mu, zeta]):
    
      # Instead of calling `tf.convert_to_tensor`,
      self.check_arguments(a, mu, zeta)

      with tf.name_scope('distributions'):

        with tf.name_scope('categorical'):

          # For gauge fixing. C.f. "/docs/nn4post.tm", section "Gauge
          # Fixing".
          # shape: `[]`
          a_mean = tf.reduce_mean(a, name='a_mean')

          # Rescaling of `a`. C.f. "/docs/nn4post.tm", section "Re-
          # scaling of a".
          # shape: `[n_c]`
          c = tf.nn.softmax(self.r * (a - a_mean), name='c')

        with tf.name_scope('standard_normal'):

          # shape: `[n_c, n_d]`
          sigma = tf.nn.softplus(zeta)

          # shape: `[n_c, n_d]`
          std_normal = Independent(
              Normal(tf.zeros(mu.shape), tf.ones(sigma.shape))
          )

      with tf.name_scope('loss'):

        with tf.name_scope('samples'):

          # shape: `[n_samples, n_c, n_d]`
          eta_samples = std_normal.sample(self.n_samples)

        with tf.name_scope('re_parameter'):

          # shape: `[n_samples, n_c, n_d]`
          theta_samples = eta_samples * sigma + mu

          # shape: `[n_samples * n_c, n_d]`
          flat_theta_samples = tf.reshape(theta_samples, [-1, self.n_d])

        with tf.name_scope('p_part'):

          with tf.name_scope('expect_log_p'):

            def log_p(thetas):
              """Vectorizes `log_posterior_upto_const`.

              Args:
                thetas:
                  Tensor of the shape `[None, n_d]`

              Returns:
                Tensor of the shape `[None]`.
              """
              return tf.map_fn(self.log_posterior_upto_const, thetas)

            # Expectation of :math:`\ln p`
            # shape: `[n_c]`
            expect_log_p = tf.reduce_mean(
                tf.reshape(
                    # shape: `[n_samples * n_c]`
                    log_p(flat_theta_samples),
                    [self.n_samples, self.n_c]),
                axis=0)

          # shape: `[]`
          loss_p_part = - tf.reduce_sum(c * expect_log_p)

        with tf.name_scope('q_part'):

          with tf.name_scope('log_q'):

            gaussian_mixture_log_prob = \
                get_gaussian_mixture_log_prob(c, mu, sigma)

            def log_q(thetas):
              """The vectorized `log_q`.

              Args:
                thetas:
                  Tensor of the shape `[None, n_d]`.

              Returns:
                Tensor of the shape `[None]`.
              """
              return tf.map_fn(gaussian_mixture_log_prob, thetas)

          with tf.name_scope('expect_log_q'):

            # Expectation of :math:`\ln q`
            # shape: `[n_c]`
            expect_log_q = tf.reduce_mean(
                tf.reshape(
                    log_q(flat_theta_samples),  # shape: `[n_samples * n_c]`.
                    [self.n_samples, self.n_c]),
                axis=0)

          # shape: `[]`
          loss_q_part = tf.reduce_sum(c * expect_log_q)

        with tf.name_scope('loss'):

          with tf.name_scope('elbo'):

            elbo = loss_p_part + loss_q_part

          with tf.name_scope('regularization'):

            # NOTE:
            #   Get punished if the range of `a` exceeds `max_a_range`.
            #   Multiplied by `elbo` for automatically setting the order of
            #   punishment.

            with tf.name_scope('a_range'):
              # shape: `[]`, and non-negative.
              a_range = tf.reduce_max(a) - tf.reduce_min(a)

            # Use "wall_function" for regularization.
            wall = get_wall(self.max_a_range, self.wall_slope)
            # shape: `[]`
            regularization = elbo * wall(a_range)

          # shape: `[]`
          loss = elbo + regularization

      with tf.name_scope('gradients'):

        # C.f. "/docs/nn4post.tm", section "Frozen-out Problem".

        with tf.name_scope('bared_gradients'):

          gradient_dict = {
              variable:
                tf.gradients(loss, variable)[0]
              for variable in {a, mu, zeta}
          }

        with tf.name_scope('keep_non_frozen_out'):

          # Notice `tf.truediv` is not broadcastable
          # shape: `[n_c]`
          denominator = tf.pow(c + self.epsilon, self.beta)
          gradient_dict = {
              variable:
                # shape: `[n_c]`
                grad / denominator if variable is a
                # shape: `[n_c, n_d]`
                else grad / tf.expand_dims(denominator, axis=1)
              for variable, grad in gradient_dict.items()
          }

        # Re-arrange as a list of tuples
        gradients = [(g, v) for v, g in gradient_dict.items()]

    return loss, gradients
