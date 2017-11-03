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
SEED = 123
tf.set_random_seed(SEED)
np.random.seed(SEED)




def get_gaussian_entropy(sigma):
  """Entropy of a multivariate Gaussian distribution with
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
          init_c = np.array([ 1 / n_c for i in range(n_c) ],
                            dtype='float32')
          init_mu = np.array(
              [ np.random.normal(size=[n_d]) * 5.0 for i in range(n_c) ],
              dtype='float32')
          init_zeta = np.array([ np.ones([n_d]) * 5.0 for i in range(n_c) ],
                               dtype='float32')
        else:
          init_c = init_vars['a']
          init_mu = init_vars['mu']
          init_zeta = init_vars['zeta']

        assert sum(init_c) == 1

        c = [
            tf.Variable(init_c[i], name='c_{}'.format(i))  # shape: `[]`.
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

            entropy_lower_bound = sum(
                [ c[i] * gaussian_entropy[i] for i in range(n_c) ]
            )

          q_entropy = entropy_lower_bound


        with tf.name_scope('loss'):

          loss = - ( log_p_mean + q_entropy )


      with tf.name_scope('optimization'):

        learning_rate = tf.placeholder(
            shape=[], dtype=tf.float32, name='learning_rate')

        if optimizer is None:
          _optimizer = tf.train.AdamOptimizer(learning_rate)
        else:
          _optimizer = optimizer(learning_rate)

        with tf.name_scope('grad_c'):

          gradients = _optimizer.compute_gradients(loss)
          # XXX

        train_op = _optimizer.apply_gradients(gradients)


    # -- Collections
    ops = {
        'vars': {
            'c': tf.stack(c, axis=0),
            'mu': tf.stack(mu, axis=0),
            'zeta': tf.stack(zeta, axis=0),
        },
        'feed': {
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

    print(ops['vars']['mu'].shape)

  return (graph, ops)




if __name__ == '__main__':

  """Test and Trail on Gaussian Mixture Distribution."""

  from tensorflow.contrib.distributions import (
      Categorical, NormalWithSoftplusScale, Mixture
  )


  # -- Parameters
  N_C = 5  # shall be varied.
  N_D = 2
  N_SAMPLES = 100
  TARGET_N_C = 3  # shall be fixed.
  LOG_ACCURATE_LOSS = True
  PROFILING = False
  DEBUG = False
  N_ITERS = 10 ** 3
  SKIP_STEP = 10
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

    target_a = tf.constant([-1., 0., 1.])
    target_mu = tf.stack([
          tf.ones([N_D]) * (i - 1) * 3
          for i in range(TARGET_N_C)
        ], axis=0)
    target_zeta = tf.zeros([TARGET_N_C, N_D])

    cat = Categorical(logits=target_a)
    components = [
        Independent(
            NormalWithSoftplusScale(target_mu[i], target_zeta[i])
        ) for i in range(TARGET_N_C)
      ]
    p = Mixture(cat, components)

    def log_posterior(theta):
        return p.log_prob(theta)

  graph, ops = build_inference(N_C, N_D, log_posterior,
                               optimizer=OPTIMIZER)


  # -- Training
  with tf.Session(graph=graph) as sess:

    sess.run(tf.global_variables_initializer())

    # Display Targets
    print(target_a.eval())
    print(target_mu.eval())
    print(target_zeta.eval())
    print()

    # Display Initialized Values
    var_ops = ops['vars']
    print(var_ops['a'].eval())
    print(var_ops['mu'].eval())
    print(var_ops['zeta'].eval())
    print()

    # Optimizing
    with Timer():
      for i in range(N_ITERS):

        feed_ops = ops['feed']
        feed_dict = {
            feed_ops['learning_rate']: 0.03,
            feed_ops['n_samples']: N_SAMPLES,
        }

        train_op = ops['train']['train_op']
        loss = ops['loss']['loss']

        loss_val, _ = sess.run([loss, train_op], feed_dict)

        # Display Trained Values
        if i % SKIP_STEP == 0:
          print('--- {0:5}  | {1}'.format(i, loss_val))
          print(var_ops['a'].eval())
          print(var_ops['mu'].eval())
          print(var_ops['zeta'].eval())
          print()


'''Trial

With `N_C = 5`, we find that `N_ITERS = 10**3` with `LR = 0.03` is enough for
finding out all three peaks of the target Gaussian mixture distribution, as
well as their variance, with high accuracy.

The only trouble is the value of `a`. It has correct order between the values of
its components. But the values are not correct. I GUESS that this is caused by
the numerical instability of softmax.
'''
