#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import numpy as np
# -- `contrib` module in TF 1.3
from tensorflow.contrib.distributions import (
    Categorical, NormalWithSoftplusScale,
    MultivariateNormalDiagWithSoftplusScale, Mixture
)
from tensorflow.contrib.bayesflow import entropy
# -- To be changed in TF 1.4
from mixture_same_family import MixtureSameFamily
from tools import ensure_directory, Timer, TimeLiner
import mnist
import pickle
import time
from tensorflow.python.client import timeline
from independent import Independent



# For testing (and debugging)
tf.set_random_seed(42)
np.random.seed(42)


def build_inference(base_graph, n_cats, log_p, d_log_p=None, log_accurate_loss=False):

  with tf.name_scope('inference'):

    with tf.name_scope('variables'):

      with tf.name_scope('initial_values'):

        init_cat_logits = tf.zeros([n_cats])
        init_locs = [tf.random_normal([param_space_dim])
                     for i in range(n_cats)]
        init_softplus_scales = [tf.zeros([param_space_dim])
                                for i in range(n_cats)]
        # Or using the values in the previous calling of this script.

      cat_logits = tf.Variable(init_cat_logits, name='cat_logits')
      locs = [
          tf.Variable(init_loc[i], name='loc_{}'.format(i))
          for i in range(n_cats)
      ]
      softplus_scales = [
          tf.Variable(init_softplus_scales[i],
                      name='softplus_scale_{}'.format(i))
          for i in range(n_cats)
      ]

    with tf.name_scope('distributions'):

      with tf.name_scope('cat'):

        cat_probs = tf.nn.softmax(cat_logits)
        cat = Categorical(prob=cat_probs)

      with tf.name_scope('normal'):

        scales = [tf.nn.softplus(softplus_scales[i]) for i in range(n_cats)]

        # Used for computing entropy lower bound
        normals = [
            Independent(Normal(locs[i], scales[i]))
            for i in range(n_cats)
        ]
        # Used for sampling
        std_normals = [
            Independent(Normal(tf.zeros(shape=locs[i].shape),
                               tf.ones(shape=scales[i].shape)))
            for i in range(n_cats)
        ]

      with tf.name_scope('q'):

        q = Mixture(cat, normals)


    with tf.name_scope('loss'):

      with tf.name_scope('log_p_term'):

        n_samples = tf.placeholder(shape=[], dtype=tf.int32, name='n_samples')

        with tf.name_scope('param_samples'):

          etas = [# shape: `[n_samples, param_space_dim]`
                  std_normals[i].sample(n_samples)
                  for i in range(n_cats)]

        with tf.name_scope('reparam'):

          thetas = [tf.matmul(etas[i], scales[i]) + tf.expand_dims(loc[i], 0)
                    for i in range(n_cats)]

        def log_p(thetas):
          """Vectorize `log_posterior`."""
          return tf.map_fn(log_posterior, thetas)

        cat_prob_comp = tf.unstack(cat_probs)

        log_p_mean = sum([
            cat_prob_comp[i] * tf.reduce_mean(log_p(thetas[i]))
            for i in range(n_cats)
        ])

      with tf.name_scope('q_entropy'):
        q_entropy = q.entropy_lower_bound()

      with tf.name_scope('approximate_loss'):
        approximate_loss = - ( log_p_mean + q_entropy )

      with tf.name_scope('accurate_loss'):
        accurate_loss = - entropy.elbo_ratio(log_p, q, n=n_samples)


    with tf.name_scope('optimization'):

      optimizer = tf.contrib.opt.NadamOptimizer
      learning_rate = tf.placeholder(shape=[], dtype=tf.float32,
                                     name='learning_rate')
      train_op = optimizer(learning_rate).minimize(approximate_loss)
      #train_op = optimizer(learning_rate).minimize(accurate_loss)  # test!

  # -- Collections
  ops = {
      'vars': {
          'cat_logits': cat_logits,
          'locs': locs,
          'softplus_scales': softplus_scales,
      },
      'feed': {
          'learning_rate': learning_rate,
          'n_samples': n_samples,
      },
      'loss': {
          'approximate_loss': approximate_loss,
          'accurate_loss': accurate_loss,
      },
      'train': {
          'train_op': train_op,
      },
  }
  for class_name, op_dict in ops.items():
    for op_name, op in op_dict.items():
      graph.add_to_collection(op_name, op)

  return (graph, ops)



