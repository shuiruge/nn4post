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
from tensorflow.contrib.distributions import (
    Normal, Categorical, Mixture
)
try:
    from tensorflow.contrib.distribution import Independent
except:
    # Your TF < 1.4.0
    from independent import Independent



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



def build_inference(n_c, n_d, log_posterior, init_vars=None,
                    base_graph=None, n_samples=10, a_rescale_factor=1.0,
                    dtype='float32', verbose=True):
  r"""Add the scope of inference to the graph `base_graph`.

  CAUTION:
    This function will MODIFY the `base_graph`, or the graph returned from
    `tf.get_default_graph()` if the `base_graph` is `None`. (Pure functional
    approach is suppressed, since it's memory costy.)

  Args:
    n_c:
      `int`, as the number of categorical probabilities, i.e. the :math:`N_c`
      in the documentation.

    n_d:
      `int`, as the number of dimension, i.e. the :math:`N_d` in the
      documentation.

    log_posterior:
      Callable from tensor of the shape `[n_d]` to scalar, both with the same
      dtype as the `dtype` argument.

    init_vars:
      `dict` for setting the initial values of variables. optional. It has
      keys `'a'`, `'mu'`, and `'zeta'`, and values of numpy arraies or tensors
      of the shapes `[n_c]`, `[n_c, n_d]`, and `[n_c, n_d]`, respectively. All
      these values shall be the same dtype as the `dtype` argument.

    base_graph:
      An instance of `tf.Graph`, optional, as the graph that the scope for
      inference are added to. If `None`, use the graph returned from
      `tf.get_default_graph()`.

    n_samples:
      `int` or `tf.placeholder` with scalar shape and `int` dtype, as the
      number of samples in the Monte Carlo integrals, optional.

    a_rescale_factor:
      `float` or `tf.placeholder` with scalar shape and `dtype` dtype, as the
      rescaling factor of `a`, optional.

    dtype:
      `str`, as the dtype of floats employed herein, like `float32`, `float64`,
      etc., optional.

    verbose:
      `bool`.
  """

  graph = tf.get_default_graph() if base_graph is None else base_graph


  if verbose:
    msg = ( 'INFO - Function `building_inference()` will MODIFY the graph {0}.'
          + ' (Pure functional approach is suppressed, being memory costy.)' )
    print(msg.format(graph))


  with graph.as_default():


    with tf.name_scope('inference'):


      with tf.name_scope('variables'):

        if init_vars is None:
          init_a = np.array([0.0 for i in range(n_c)],
                            dtype=dtype)
          init_mu = np.array(
              [np.random.normal(size=[n_d]) * 5.0 for i in range(n_c)],
              dtype=dtype)
          init_zeta = np.array([np.ones([n_d]) * 5.0 for i in range(n_c)],
                               dtype=dtype)
        else:
          init_a = init_vars['a']
          init_mu = init_vars['mu']
          init_zeta = init_vars['zeta']

        # shape: `[n_c]`
        a = tf.Variable(init_a, name='a')
        # shape: `[n_c, n_d]`
        mu = tf.Variable(init_mu, name='mu')
        # shape: `[n_c, n_d]`
        zeta = tf.Variable(init_zeta, name='zeta')


      with tf.name_scope('distributions'):


        with tf.name_scope('categorical'):

          # shape: `[n_c]`
          c = tf.nn.softmax( (a - tf.reduce_mean(a)) \
                             * a_rescale_factor,
                            name='c')


        with tf.name_scope('standard_normal'):

          # shape: `[n_c, n_d]`
          sigma = tf.nn.softplus(zeta)

          # shape: `[n_c, n_d]`
          std_normal = \
              Independent(
                  Normal(tf.zeros(mu.shape), tf.ones(sigma.shape)))


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
              """Vectorize `log_posterior`.

              Args:
                thetas:
                  Tensor of the shape `[None, n_d]`

              Returns:
                Tensor of the shape `[None]`.
              """
              return tf.map_fn(log_posterior, thetas)

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

          loss = loss_p_part + loss_q_part


    # -- Collections
    ops = {
        'a': a,
        'mu': mu,
        'zeta': zeta,
        'c': c,
        'loss': loss,
    }

    if isinstance(a_rescale_factor, tf.Tensor):
      ops['a_rescale_factor'] = a_rescale_factor
    if isinstance(n_samples, tf.Tensor):
      ops['n_samples'] = n_samples

    for op_name, op in ops.items():
      graph.add_to_collection(op_name, op)

  return ops




if __name__ == '__main__':

  """Test and Trail on Gaussian Mixture Distribution."""

  from tensorflow.contrib.distributions import NormalWithSoftplusScale
  from tools import Timer


  # For testing (and debugging)
  SEED = 123
  tf.set_random_seed(SEED)
  np.random.seed(SEED)


  # -- Parameters
  TARGET_N_C = 3  # shall be fixed.
  N_D = 10**5
  N_C = 3  # shall be varied.
  N_SAMPLES = 10
  A_RESCALE_FACTOR = 1.0
  N_ITERS = 1 * 10**4
  LR = 0.05
  #OPTIMIZER = tf.train.AdamOptimizer(LR)
  OPTIMIZER = tf.train.RMSPropOptimizer(LR)
  DTYPE = 'float32'
  SKIP_STEP = 50



  # -- Gaussian Mixture Distribution
  with tf.name_scope('posterior'):

    target_c = tf.constant([0.05, 0.25, 0.70])
    target_mu = tf.stack([
          tf.ones([N_D]) * (i - 1) * 3
          for i in range(TARGET_N_C)
        ], axis=0)
    target_zeta_val = np.zeros([TARGET_N_C, N_D])
    #target_zeta_val[1] = np.ones([N_D]) * 5.0
    target_zeta = tf.constant(target_zeta_val, dtype='float32')

    cat = Categorical(probs=target_c)
    components = [
        Independent(
            NormalWithSoftplusScale(target_mu[i], target_zeta[i])
        ) for i in range(TARGET_N_C)
      ]
    p = Mixture(cat, components)

    def log_posterior(theta):
        return p.log_prob(theta)

  # test!
  init_vars = None
  mu_val = np.array(np.random.uniform(low=-1, high=1, size=[N_C, N_D]) * 10.0,
                     dtype=DTYPE)
  mu_val[0] = np.ones([N_D]) * 1.5
  init_vars = {
    'a':
      np.zeros([N_C], dtype=DTYPE),
    'mu':
      mu_val,
    'zeta':
      np.ones([N_C, N_D], dtype=DTYPE) * (-5.0),
  }
  init_vars = {
    'a':
      np.zeros([N_C], dtype=DTYPE),
    'mu':
      np.array([np.ones([N_D]) * (i - 1) * 3 for i in range(N_C)],
               dtype=DTYPE) \
      + np.array(np.random.normal(size=[N_C, N_D]) * 0.5,
                 dtype=DTYPE),
    'zeta':
      np.array(np.random.normal(size=[N_C, N_D]) * 5.0,
               dtype=DTYPE),
  }

  a_rescale_factor = tf.placeholder(shape=[], dtype=DTYPE)

  ops = build_inference(N_C, N_D, log_posterior,
            init_vars=init_vars, n_samples=N_SAMPLES,
            a_rescale_factor=a_rescale_factor)

  train_op = OPTIMIZER.minimize(ops['loss'])


  # -- Training
  with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    # Display Targets
    print(target_c.eval())
    print(target_mu.eval())
    print(target_zeta.eval())
    print()

    # Display Initialized Values
    print(ops['mu'].eval())
    print(ops['zeta'].eval())
    print()

    # Optimizing
    with Timer():
      for i in range(N_ITERS):

        a_rescale_factor_val = A_RESCALE_FACTOR if i > 500 else 0.0

        _, loss_val, a_val, c_val, mu_val, zeta_val = \
            sess.run([
                train_op, ops['loss'], ops['a'],
                ops['c'], ops['mu'], ops['zeta']
            ], feed_dict={ops['a_rescale_factor']: a_rescale_factor_val})

        # Display Trained Values
        if i % SKIP_STEP == 0:
          print('--- {0:5}  | {1}'.format(i, loss_val))
          print('c:\n', c_val)
          print('a:\n', a_val)
          print('mu (mean):\n', np.mean(mu_val, axis=1))
          print('mu (std):\n', np.std(mu_val, axis=1))
          print('zeta (mean):\n', np.mean(zeta_val, axis=1))
          print('zeta (std):\n', np.std(zeta_val, axis=1))
          print()

      print('--- SUMMARY ---')
      print()
      print('-- Parameters')
      print('n_d: ', N_D)
      print('n_c:', N_C)
      print('n_samples: ', N_SAMPLES)
      print('a_rescale_factor: ', A_RESCALE_FACTOR)
      print('n_iters: ', N_ITERS)
      print('learning-rate: ', LR)
      print('dtype: ', DTYPE)
      print()
      print('-- Result')
      print('c:\n', c_val)
      print('a:\n', a_val)
      print('mu (mean):\n', np.mean(mu_val, axis=1))
      print('mu (std):\n', np.std(mu_val, axis=1))
      print('zeta (mean):\n', np.mean(zeta_val, axis=1))
      print('zeta (std):\n', np.std(zeta_val, axis=1))
      print()



'''Trial

### Trial 1
Given `TARGET_N_C = 3` and `N_D = 100`, we find that

  # -- Parameters
  N_C = 10  # shall be varied.
  N_SAMPLES = 10
  A_RESCALE_FACTOR = 0.1
  N_ITERS = 2 * 10**3
  LR = 0.03
  OPTIMIZER = tf.train.RMSPropOptimizer(LR)
  DTYPE = 'float32'

is enough to reach the goal, with elapsed time 355 secs, and cost memory 160M.


### Trial 2
Given `TARGET_N_C = 3` and `N_D = 1000` with the `mu` initially locates on the
peaks of the given Gaussian mixture model, but with equal `c` and random `zeta`,
we find that

  # -- Parameters
  N_C = 3  # shall be varied.
  N_SAMPLES = 10
  A_RESCALE_FACTOR = 0.1
  N_ITERS = 4 * 10**3
  LR = 0.03
  OPTIMIZER = tf.train.RMSPropOptimizer(LR)
  DTYPE = 'float32'

gives a perfect result. However, if set `A_RESCALE_FACTOR = 1.0` instead, we
will get a terrible result of `c` with `c[1]` equals `1.0` almost.
'''
