#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import numpy as np
# -- `contrib` module in TF 1.3
from tensorflow.contrib.distributions import Normal
from tools import ensure_directory, Timer
from independent import Independent



# For testing (and debugging)
SEED = 12334023
tf.set_random_seed(SEED)
np.random.seed(SEED)



def get_gaussian_entropy(sigma):
  """Get the entropy of a multivariate Gaussian distribution with
  ALL DIMENSIONS INDEPENDENT.

  C.f. eq.(8.7) of [here](http://www.biopsychology.org/norwich/isp/\
  chap8.pdf).

  NOTE:
    Gaussian entropy is independent of its center `mu`.

  Args:
    sigma:
      Tensor of shape `[None]`.

  Returns:
    Scalar.
  """
  n_dims = np.prod(sigma.get_shape().as_list())
  return 0.5 * n_dims * tf.log(2. * np.pi * np.e) \
         + tf.reduce_sum(tf.log(sigma))



def build_inference(n_c, n_d, log_posterior,
    init_vars=None, optimizer=None, base_graph=None):
  """XXX"""

  graph = tf.get_default_graph() if base_graph is None else base_graph


  with graph.as_default():

    with tf.name_scope('inference'):

      with tf.name_scope('variables'):

        if init_vars is None:
          init_a = np.array([0.0 for i in range(n_c)],
                            dtype='float32')
          init_mu = np.array(
              [np.random.normal(size=[n_d]) * 5.0 for i in range(n_c)],
              dtype='float32')
          init_zeta = np.array([np.ones([n_d]) * 1.0 for i in range(n_c)],
                              dtype='float32')
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

        a_rescale_factor = tf.placeholder(shape=[], dtype=tf.float32,
                                          name='a_rescale_factor')
        c = tf.nn.softmax(a_rescale_factor * tf.stack(a), name='c')  # `[n_c]`.


        with tf.name_scope('log_p_part'):

          n_samples = tf.placeholder(shape=[], dtype=tf.int32, name='n_samples')


          with tf.name_scope('samples'):

            eta_samples = [
                std_normal[i].sample(n_samples)  # `[n_samples, n_d]`
                for i in range(n_c)
            ]


          with tf.name_scope('reparameterize'):

            theta_samples = [
                eta_samples[i] * sigma[i] + mu[i]
                for i in range(n_c)
            ]


          def log_p(theta_samples):
            """Vectorize `log_posterior`."""
            return tf.map_fn(log_posterior, theta_samples)

          expectation = [
              tf.reduce_mean( log_p(theta_samples[i]) )
              for i in range(n_c)
          ]

          log_p_mean = sum([ c[i] * expectation[i] for i in range(n_c) ])


        with tf.name_scope('q_entropy'):

          with tf.name_scope('gaussian_entropy'):

            gaussian_entropy = [
                get_gaussian_entropy(sigma[i])
                for i in range(n_c)
            ]

          with tf.name_scope('entropy_lower_bound'):

            entropy_lower_bound = \
                sum([ c[i] * gaussian_entropy[i] for i in range(n_c) ])

          q_entropy = entropy_lower_bound


        with tf.name_scope('loss'):

          loss = - ( log_p_mean + q_entropy )


      with tf.name_scope('optimization'):

        learning_rate = tf.placeholder(
            shape=[], dtype=tf.float32, name='learning_rate')

        if optimizer is None:
          _optimizer = tf.train.AdamOptimizer
        else:
          _optimizer = optimizer

        train_op = _optimizer(learning_rate).minimize(loss)


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

  return (graph, ops)




if __name__ == '__main__':

  """Test and Trail on Gaussian Mixture Distribution."""

  from tensorflow.contrib.distributions import (
      Categorical, NormalWithSoftplusScale, Mixture
  )


  # -- Parameters
  N_C = 1  # shall be varied.
  N_D = 10 ** 5
  N_SAMPLES = 20
  TARGET_N_C = 3  # shall be fixed.
  LOG_ACCURATE_LOSS = True
  PROFILING = False
  DEBUG = False
  A_RESCALE_FACTOR = 0.01
  LR = 0.05
  N_ITERS = 10 ** 3 * 2
  SKIP_STEP = 50
  #LOG_DIR = '../dat/logs/gaussian_mixture_model/{0}_{1}'\
  #          .format(TARGET_N_C, N_C)
  #DIR_TO_CKPT = '../dat/checkpoints/gaussian_mixture_model/{0}_{1}'\
  #              .format(TARGET_N_C, N_C)
  LOG_DIR = None
  DIR_TO_CKPT = None
  if DIR_TO_CKPT is not None:
    ensure_directory(DIR_TO_CKPT)
  #OPTIMIZER = tf.contrib.opt.NadamOptimizer
  OPTIMIZER = tf.train.RMSPropOptimizer



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
      np.zeros([N_C], dtype='float32'),
    'mu':
      np.array([np.ones([N_D]) * (i - 1) * 3 for i in range(TARGET_N_C)],
               dtype='float32'),
    'zeta':
      np.zeros([TARGET_N_C, N_D], dtype='float32'),
  }
  init_vars = None
  init_vars = {
    'a':
      np.zeros([N_C], dtype='float32'),
    'mu':
      np.array([np.ones([N_D]) * (i + 1) * 3 for i in range(N_C)],
               dtype='float32'),
    'zeta':
      np.array(np.random.normal(size=[N_C, N_D]) * 5.0,
               dtype='float32'),
  }

  graph, ops = build_inference(N_C, N_D, log_posterior,
                               optimizer=OPTIMIZER, init_vars=init_vars)


  # -- Training
  with tf.Session(graph=graph) as sess:

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

        loss_val, _ = sess.run([loss, train_op], feed_dict)

        # Display Trained Values
        if i % SKIP_STEP == 0:
          print('--- {0:5}  | {1}'.format(i, loss_val))
          print(var_ops['c'].eval(
            {feed_ops['a_rescale_factor']: A_RESCALE_FACTOR}))
          print(var_ops['mu'].eval())
          print(var_ops['zeta'].eval())
          print()


'''Trial

### Trail 1
With `N_C = 5` and `N_D = 2`, we find that `N_ITERS = 10**3` with `LR = 0.03`
(`SEED = 123`) is enough for finding out all three peaks of the target
Gaussian mixture distribution, as well as their variance, with high accuracy.

The only trouble is the value of `a`. It has correct order relation between the
values of its components. But the values are not correct. I GUESS that this is
caused by the numerical instability of softmax.

The RAM cost is quite small (~100 M).


### Trail 2
With `N_C = 1` and `N_D = 10**5`, we find that `N_ITERS = 10**3` with `LR = 0.03`
(`SEED = 123`) it is hard to find the greatest peak of the Gaussian mixture
distribution. However, once the center of the peak is given, the variance can be
found soon, while the center is kept invariant, even when the initial variance
is great enough so that other peaks can be percived in the training.

The RAM cost is quite small (~200 M).
'''
