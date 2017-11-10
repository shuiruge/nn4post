#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description
-----------
ADVI implementation of "nerual network for posterior" with accurate entropy of
q-distribution.

TF version
----------
Tested on TF 1.4.0
"""


import numpy as np
import tensorflow as tf
from tensorflow.contrib.distributions import (
    Normal, Categorical, Mixture, Independent
)


# For testing (and debugging)
SEED = 123456
tf.set_random_seed(SEED)
np.random.seed(SEED)


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



def build_inference(n_c, n_d, log_posterior, init_vars=None, optimizer=None,
                    base_graph=None, dtype='float32', verbose=True):
  r"""Add the blocks of inference to the graph `base_graph`.

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

    optimizer:
      A inherit class of `tf.train.Optimizer`, optional. If `None`, use
      `tf.train.AdamOptimizer`.

    base_graph:
      An instance of `tf.Graph`, optional, as the graph that the blocks for
      inference are added to. If `None`, use the graph returned from
      `tf.get_default_graph()`.

    dtype:
      `str`, like `float32`, `float64`, etc., optional.

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

        a = [
            tf.Variable(init_a[i], name='a_{}'.format(i))  # shape: `[]`.
            for i in range(n_c)
        ]
        mu = [
            tf.Variable(init_mu[i], name='mu_{}'.format(i))  # `[n_d]`.
            for i in range(n_c)
        ]
        zeta = [
            tf.Variable(init_zeta[i], name='zeta_{}'.format(i))  # `[n_d]`.
            for i in range(n_c)
        ]


      with tf.name_scope('distributions'):


        with tf.name_scope('categorical'):

          a_rescale_factor = tf.placeholder(shape=[], dtype=dtype,
                                            name='a_rescale_factor')
          c = tf.nn.softmax(a_rescale_factor * tf.stack(a), name='c')  # `[n_c]`.


        with tf.name_scope('standard_normal'):

          sigma = [
              tf.nn.softplus(zeta[i])  # `[n_d]`.
              for i in range(n_c)
          ]

          std_normal = [
              Independent(
                  Normal(tf.zeros(mu[i].shape), tf.ones(sigma[i].shape))
              ) for i in range(n_c)
          ]


      with tf.name_scope('loss'):


        with tf.name_scope('samples'):

          n_samples = tf.placeholder(shape=[], dtype=tf.int32,
                                      name='n_samples')
          eta_samples = [
              std_normal[i].sample(n_samples)  # `[n_samples, n_d]`
              for i in range(n_c)
          ]


        with tf.name_scope('re_parameter'):

          theta_samples = [
              eta_samples[i] * sigma[i] + mu[i]  # `[n_samples, n_d]`.
              for i in range(n_c)
          ]


        with tf.name_scope('p_part'):


          with tf.name_scope('expect_log_p'):

            def log_p(theta_samples):
              """Vectorize `log_posterior`."""
              return tf.map_fn(log_posterior, theta_samples)

            # Expectation of :math:`\ln p`
            expect_log_p = [
                tf.reduce_mean( log_p(theta_samples[i]) )
                for i in range(n_c)
            ]

          loss_p_part = - tf.add_n(
              [ c[i] * expect_log_p [i] for i in range(n_c) ]
          )


        with tf.name_scope('q_part'):


          with tf.name_scope('log_q'):

            gaussian_mixture_log_prob = \
                get_gaussian_mixture_log_prob(c, mu, sigma)

            def log_q(theta_samples):
              """Vectorize `log_q`."""
              return tf.map_fn(gaussian_mixture_log_prob, theta_samples)


          with tf.name_scope('expect_log_q'):

            expect_log_q = [
                tf.reduce_mean( log_q(theta_samples[i]) )
                for i in range(n_c)
            ]


          loss_q_part = tf.add_n(
              [ c[i] * expect_log_q [i] for i in range(n_c) ]
          )


        with tf.name_scope('loss'):

          loss = loss_p_part + loss_q_part


      with tf.name_scope('optimization'):

        learning_rate = tf.placeholder(
            shape=[], dtype=dtype, name='learning_rate')

        if optimizer is None:
          _optimizer = tf.train.AdamOptimizer(learning_rate)
        else:
          _optimizer = optimizer(learning_rate)

        # Redefine the gradients of `a`
        with tf.name_scope('redefine_grad_a'):

          grad_and_var_list = _optimizer.compute_gradients(loss)
          grad_a = {v: g for g, v in grad_and_var_list if v in a}
          average_grad_a = tf.reduce_mean(
              [ g for g in grad_a.values() ]
          )
          new_grad_a = {
              v:
                g - average_grad_a
              for v, g in grad_a.items()
          }

        new_grad_and_var_list = [
            (new_grad_a[v], v) if v in a else (g, v)
            for g, v in grad_and_var_list
        ]
        train_op = _optimizer.apply_gradients(new_grad_and_var_list)


    # -- Collections
    ops = {
        'vars': {
            'a': tf.stack(a, axis=0),
            'mu': tf.stack(mu, axis=0),
            'zeta': tf.stack(zeta, axis=0),
            'c': c,
        },
        'feed': {
            'a_rescale_factor': a_rescale_factor,
            'learning_rate': learning_rate,
            'n_samples': n_samples,
        },
        'loss': {
            'loss': loss,
        },
        'train': {
            'train_op': train_op,
        },
    }
    for class_name, op_dict in ops.items():
      for op_name, op in op_dict.items():
        graph.add_to_collection(op_name, op)

  return ops




if __name__ == '__main__':

  """Test and Trail on Gaussian Mixture Distribution."""

  from tensorflow.contrib.distributions import NormalWithSoftplusScale
  from tools import Timer


  # -- Parameters
  N_C = 5  # shall be varied.
  N_D = 2
  N_SAMPLES = 10
  TARGET_N_C = 3  # shall be fixed.
  A_RESCALE_FACTOR = 1.0
  LR = 0.03
  N_ITERS = 10**3
  SKIP_STEP = 50
  #OPTIMIZER = tf.contrib.opt.NadamOptimizer
  OPTIMIZER = tf.train.RMSPropOptimizer
  #OPTIMIZER = tf.train.GradientDescentOptimizer  # test!
  DTYPE = 'float32'



  # -- Gaussian Mixture Distribution
  with tf.name_scope('posterior'):

    target_c = tf.constant([0.05, 0.25, 0.70])
    target_mu = tf.stack([
          tf.ones([N_D]) * (i - 1) * 3
          for i in range(TARGET_N_C)
        ], axis=0)
    target_zeta = tf.zeros([TARGET_N_C, N_D])

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
  init_vars = {
    'a':
      np.zeros([N_C], dtype=DTYPE),
    'mu':
      np.array([np.ones([N_D]) * (i - 1) * 3 for i in range(TARGET_N_C)],
               dtype=DTYPE),
    'zeta':
      np.zeros([TARGET_N_C, N_D], dtype=DTYPE),
  }
  init_vars = {
    'a':
      np.zeros([N_C], dtype=DTYPE),
    'mu':
      np.array([np.ones([N_D]) * (i + 1) * 3 for i in range(N_C)],
               dtype=DTYPE),
    'zeta':
      np.array(np.random.normal(size=[N_C, N_D]) * 5.0,
               dtype=DTYPE),
  }
  init_vars = None

  ops = build_inference(N_C, N_D, log_posterior,
                        optimizer=OPTIMIZER, init_vars=init_vars)


  # -- Training
  with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    # Display Targets
    print(target_c.eval())
    print(target_mu.eval())
    print(target_zeta.eval())
    print()

    # Display Initialized Values
    var_ops = ops['vars']
    print(var_ops['mu'].eval())
    print(var_ops['zeta'].eval())
    print()

    # Optimizing
    with Timer():
      for i in range(N_ITERS):

        feed_ops = ops['feed']
        feed_dict = {
            feed_ops['a_rescale_factor']: A_RESCALE_FACTOR,
            feed_ops['learning_rate']: LR,
            feed_ops['n_samples']: N_SAMPLES,
        }

        train_op = ops['train']['train_op']
        loss = ops['loss']['loss']

        _, loss_val, a_val, c_val, mu_val, zeta_val = \
            sess.run([ train_op, loss, var_ops['a'],
                       var_ops['c'], var_ops['mu'], var_ops['zeta'] ],
                     feed_dict)

        # Display Trained Values
        if i % SKIP_STEP == 0:
          print('--- {0:5}  | {1}'.format(i, loss_val))
          print('c:\n', c_val)
          print('a:\n', a_val)
          print('mu:\n', mu_val)
          print('zeta:\n', zeta_val)
          print()


'''Trial

### Trail 1
With `N_C = 5` and `N_D = 2`, we find that `N_ITERS = 10**3` with `LR = 0.03`
(`SEED = 123`) is enough for finding out all three peaks of the target
Gaussian mixture distribution, as well as their variance, with high accuracy.

Comparing to the same ADVI implementation but employing entropy lower bound,
herein, the accuracy of `c` is much pretty.

The RAM cost is quite small (~100 M).


### Trail 2
With `N_C = 1` and `N_D = 10**5`, we find that `N_ITERS = 10**3` with `LR = 0.03`
and `N_SAMPLES = 20` (`SEED = 123`) it is hard to find the greatest peak of the
Gaussian mixture distribution. However, once the center of the peak is given,
the variance can be found soon, while the center is kept invariant, even when
the initial variance is great enough so that other peaks can be percived in the
training.

The RAM cost is quite small (~200 M).


### Trail 3
With `N_C = 5` and `N_D = 2`, we find that `N_ITERS = 10**3` with `LR = 0.03`
and `N_SAMPLES = 10` (`SEED = 123`) makes perfect. It finds out all peaks
with accurate categorical probabilities, centers, and variances.

The RAM cost is quite small (~250 M)

'''
