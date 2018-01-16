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
    print('WARNING - Your TF < 1.4.0.')
    from nn4post.utils.independent import Independent



def get_gaussian_mixture_log_prob(cat_probs, gauss_mu, gauss_sigma):
  """Get the logrithmic p.d.f. of a Gaussian mixture model.

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

  n_cats = cat_probs.shape[0]
  cat = Categorical(probs=cat_probs)
  components = [
      Independent( Normal(gauss_mu[i], gauss_sigma[i]) )
      for i in range(n_cats)
  ]
  distribution = Mixture(cat=cat, components=components)

  return distribution.log_prob



def get_wall(wall_position, wall_slope):
  """Get a "wall-function" in space of any dimensionality. Along any dimension,
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
      `Tensor` with the same shape and dtype as `x`, as the hight of the wall at
      position `x`.
  """

  def wall(x):
    return tf.nn.softplus( wall_slope * (x - wall_position) )

  return wall



def build_nn4post(
        n_c, n_d, log_posterior_upto_const, init_var=None, base_graph=None,
        n_samples=10, r=1.0, beta=1.0,  max_a_range=10, wall_slope=10,
        epsilon=1e-08, dtype='float32', name='nn4post'):
  r"""Add the name-scope `name` to the graph `base_graph`. This is the
  implementation of 'docs/main.pdf'.

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

    base_graph:
      An instance of `tf.Graph`, optional, as the graph that the scope for
      "nn4post" are added to. If `None`, use the graph returned from
      `tf.get_default_graph()`.

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
        much greater than unit. But when learning-rate of optimizer is not small
        enough (as generally demanded in the early stage of training), extremely
        great value of `wall_slope` will triger `NaN`.

    epsilon:
      `float` or `tf.placeholder` with scalar shape and `dtype` dtype, as the
      :math:`epsilon` in the documentation, optional.

    dtype:
      `str`, as the dtype of floats employed herein, like `float32`, `float64`,
      etc., optional.

    name:
      `str`, as the main name-scope.

  Returns:
    A tuple of two elements. The first is a `dict` for useful `tensor`s (for
    convinence), with keys `'a'`, `'mu'`, `'zeta'`, and `'loss'`, and with
    their associated tensors as values. The second is a list of tupes of
    gradient and its associated variable, as the argument of the method
    `tf.train.Optimizer.apply_gradients()`.
  """

  graph = tf.get_default_graph() if base_graph is None else base_graph


  with graph.as_default():


    with tf.name_scope(name):


      with tf.name_scope('variables'):

        if init_var is None:
          init_a = np.zeros([n_c], dtype=dtype)

          if n_c == 1:
            init_mu = np.random.normal(size=[n_c, n_d])
          else:
            # Because of the curse of dimensionality
            init_mu = np.random.normal(size=[n_c, n_d]) * np.sqrt(n_d)
          init_mu = init_mu.astype(dtype)

          init_zeta = np.ones([n_c, n_d], dtype=dtype)

        else:
          init_a = init_var['a']
          init_mu = init_var['mu']
          init_zeta = init_var['zeta']

        # shape: `[n_c]`
        a = tf.Variable(init_a, name='a')
        # shape: `[n_c, n_d]`
        mu = tf.Variable(init_mu, name='mu')
        # shape: `[n_c, n_d]`
        zeta = tf.Variable(init_zeta, name='zeta')


      with tf.name_scope('distributions'):


        with tf.name_scope('categorical'):

          # For gauge fixing. C.f. "/docs/nn4post.tm", section "Gauge
          # Fixing".
          # shape: `[]`
          a_mean = tf.reduce_mean(a, name='a_mean')

          # Rescaling of `a`. C.f. "/docs/nn4post.tm", section "Re-
          # scaling of a".
          # shape: `[n_c]`
          c = tf.nn.softmax(r * (a - a_mean), name='c')

          # Replaced by clipping the gradient of `a`, c.f. `name_scope`
          # `'gradients/clipping_grad_a'`.
          ## Additionally clip `c` by a minimal value
          ## shape: `[n_c]`
          #c = tf.clip_by_value(c, epsilon, 1, name='c_clipped')


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
          eta_samples = std_normal.sample(n_samples)


        with tf.name_scope('re_parameter'):

          # shape: `[n_samples, n_c, n_d]`
          theta_samples = eta_samples * sigma + mu

          # shape: `[n_samples * n_c, n_d]`
          flat_theta_samples = tf.reshape(theta_samples, [-1, n_d])


        with tf.name_scope('p_part'):


          with tf.name_scope('expect_log_p'):

            def log_p(thetas):
              """Vectorize `log_posterior_upto_const`.

              Args:
                thetas:
                  Tensor of the shape `[None, n_d]`

              Returns:
                Tensor of the shape `[None]`.
              """
              return tf.map_fn(log_posterior_upto_const, thetas)

            # Expectation of :math:`\ln p`
            # shape: `[n_c]`
            expect_log_p = tf.reduce_mean(
                tf.reshape(
                    log_p(flat_theta_samples),  # shape `[n_samples * n_c]`.
                    [n_samples, n_c]),
                axis=0)

          # shape: `[]`
          loss_p_part = - tf.reduce_sum(c * expect_log_p)


        with tf.name_scope('q_part'):


          with tf.name_scope('log_q'):

            gaussian_mixture_log_prob = \
                get_gaussian_mixture_log_prob(c, mu, sigma)

            def log_q(thetas):
              """Vectorize `log_q`.

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
                    [n_samples, n_c]),
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

            # shape: `[]`, and non-negative.
            a_range = tf.reduce_max(a) - tf.reduce_min(a)
            # Use "wall_function" for regularization.
            wall = get_wall(max_a_range, wall_slope)
            # shape: `[]`
            regularization = elbo * wall(a_range)

          # shape: `[]`
          loss = elbo + regularization


      with tf.name_scope('gradients'):

        # C.f. "/docs/nn4post.tm", section "Frozen-out Problem".

        with tf.name_scope('bared_gradients'):

          gradient = {
              variable:
                tf.gradients(loss, variable)[0]
              for variable in {a, mu, zeta}
          }


        with tf.name_scope('keep_non_frozen_out'):

          # Notice `tf.truediv` is not broadcastable
          denominator = tf.pow(c + epsilon, beta)  # `[]`
          gradient = {
              variable:
                grad / denominator if variable is a  # `[n_c]`
                else grad / tf.expand_dims(denominator, axis=1)  # `[n_c, n_d]`
              for variable, grad in gradient.items()
          }


        # Re-arrange as a list of tuples
        grads_and_vars = [(grad, var_) for var_, grad in gradient.items()]


    # -- Collections
    collection = {
        'a': a,
        'mu': mu,
        'zeta': zeta,
        'c': c,
        'loss': loss,
    }

    if isinstance(r, tf.Tensor):
      collection['r'] = r
    if isinstance(beta, tf.Tensor):
      collection['beta'] = beta
    if isinstance(n_samples, tf.Tensor):
      collection['n_samples'] = n_samples

    for name, tensor in collection.items():
      graph.add_to_collection(name, tensor)

  return (collection, grads_and_vars)
