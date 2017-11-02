#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import numpy as np
# -- `contrib` module in TF 1.3
from tensorflow.contrib.distributions import Normal
from tools import ensure_directory, Timer, TimeLiner
from independent import Independent



# For testing (and debugging)
tf.set_random_seed(42)
np.random.seed(42)


def gaussian_entropy(sigma):
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



def build_inference(n_c, n_d, log_posterior, base_graph=None):

  graph = tf.get_default_graph() if base_graph is None else base_graph


  with graph.as_default():

    with tf.name_scope('inference'):

      with tf.name_scope('variables'):

        init_a = np.array([0.0 for i in range(n_c)],
                          dtype='float32')
        init_mu = np.array([np.random.normal([n_d]) for i in range(n_c)],
                           dtype='float32')
        init_zeta = np.array([np.zeros([n_d]) for i in range(n_c)],
                             dtype='float32')
        # Or using the values in the previous calling of this script.

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

        c = tf.nn.softmax(tf.stack(a), name='c')  # `[n_c, n_d]`.


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

          log_p_mean = sum([
              c[i] * tf.reduce_mean( log_p(theta_samples[i]) )
              for i in range(n_c)
          ])


        with tf.name_scope('q_entropy'):

          with tf.name_scope('entropy_lower_bound'):

            entropy_lower_bound = sum([
                c[i] * gaussian_entropy(sigma[i])
                for i in range(n_c)
            ])


          q_entropy = entropy_lower_bound


        with tf.name_scope('approximate_loss'):
          approximate_loss = - ( log_p_mean + q_entropy )


        with tf.name_scope('accurate_loss'):
          pass


      with tf.name_scope('optimization'):

        optimizer = tf.contrib.opt.NadamOptimizer
        learning_rate = tf.placeholder(shape=[], dtype=tf.float32,
                                      name='learning_rate')
        train_op = optimizer(learning_rate).minimize(approximate_loss)
        #train_op = optimizer(learning_rate).minimize(accurate_loss)  # test!


    # -- Collections
    ops = {
        'vars': {
            'a': tf.stack(a, axis=-1),
            'mu': tf.stack(mu, axis=-1),
            'zeta': tf.stack(zeta, axis=-1),
        },
        'feed': {
            'learning_rate': learning_rate,
            'n_samples': n_samples,
        },
        'loss': {
            'approximate_loss': approximate_loss,
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

  """Test."""

  from tensorflow.contrib.distributions import (
      Categorical, NormalWithSoftplusScale, Mixture
  )


  N_CATS = 1  # shall be varied.
  N_SAMPLES = 100
  PARAM_SPACE_DIM = 2
  TARGET_N_CATS = 3  # shall be fixed.
  LOG_ACCURATE_LOSS = True
  PROFILING = False
  DEBUG = False
  SKIP_STEP = 20
  #LOG_DIR = '../dat/logs/gaussian_mixture_model/{0}_{1}'\
  #          .format(TARGET_N_CATS, N_CATS)
  #DIR_TO_CKPT = '../dat/checkpoints/gaussian_mixture_model/{0}_{1}'\
  #              .format(TARGET_N_CATS, N_CATS)
  LOG_DIR = None
  DIR_TO_CKPT = None
  if DIR_TO_CKPT is not None:
    ensure_directory(DIR_TO_CKPT)


  with tf.name_scope('posterior'):

    target_a = tf.constant([-1., 0., 1.])
    target_mu = tf.stack([
          tf.ones([PARAM_SPACE_DIM]) * (i - 1) * 3
          for i in range(TARGET_N_CATS)
        ], axis=0)
    target_zeta = tf.zeros([TARGET_N_CATS, PARAM_SPACE_DIM])

    cat = Categorical(logits=target_a)
    components = [
        Independent(
            NormalWithSoftplusScale(target_mu[i], target_zeta[i])
        ) for i in range(TARGET_N_CATS)
      ]
    p = Mixture(cat, components)

    def log_posterior(theta):
        return p.log_prob(theta)

  graph, ops = build_inference(1, PARAM_SPACE_DIM, log_posterior)


  with tf.Session(graph=graph) as sess:

    sess.run(tf.global_variables_initializer())

    n_iter = 3  # test!

    for i in range(n_iter):

      feed_ops = ops['feed']
      feed_dict = {
          feed_ops['learning_rate']: 0.01,
          feed_ops['n_samples']: N_SAMPLES,
      }

      train_op = ops['train']['train_op']
      loss = ops['loss']['approximate_loss']

      loss_val, _ = sess.run([loss, train_op], feed_dict)

      print(i, loss_val)
      # TODO: display `a`, `mu`, and `zeta`, as well as their target values.



