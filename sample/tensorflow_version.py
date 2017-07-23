#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow version. For the principle and notations, c.f. '../docs/nn4post.tm'.
"""

import tensorflow as tf
from tensorflow.contrib.distributions import \
    Categorical, Mixture, MultivariateNormalDiag



NUM_PEAKS = 5
DIM = 100
NUM_SAMPLES = 10

TEST_NUM_PEAKS = 3


# -- `tf.Variable`s
a = tf.Variable(tf.random_uniform([NUM_PEAKS]),
                dtype=tf.float32)
a_square = tf.square(a)
w = tf.Variable(tf.random_uniform([DIM, NUM_PEAKS]),
                dtype=tf.float32)
w_square = tf.square(w)
b = tf.Variable(tf.random_uniform([DIM, NUM_PEAKS]),
                dtype=tf.float32)

w_square_list = tf.unstack(w_square, axis=1)
b_list = tf.unstack(b, axis=1)
assert len(w_square_list) == len(b_list)


# -- q(theta) as a distribution
cat = Categorical(probs=a_square)
components = [
    MultivariateNormalDiag(loc=b_list[i] / w_square_list[i], 
                           scale_diag=1 / w_square_list[i])
    for i, _ in enumerate(b_list)
    ]
dist_q = Mixture(cat=cat, components=components)
        

def log_q(theta):
    """ ln q (theta)
    
    Args:
        theta: tf.Tensor(shape=[None, DIM], dtype=tf.float32)
            where `None` for batch_size.
    
    Returns:
        tf.Tensor(shape=[None], dtype=tf.float32)
    """
    
    beta = tf.log(a_square) \
         + tf.reduce_sum(
               -0.5 * tf.square((tf.matmul(theta, w_square) + b)) \
               +0.5 * tf.log(w_square / (2 * 3.14)),
               axis=0)
     
    beta_max = tf.reduce_max(beta)
    delta_beta = beta - beta_max
    top_beta, _ = tf.nn.top_k(delta_beta, k=3)
    
    return beta_max + tf.log(tf.reduce_sum(tf.exp(top_beta)))  # tested.


def kl_divergence(log_p, log_q, dist_q):
    
    theatae = dist_q.sample(sample_shape=[NUM_SAMPLES])  # tested.
    
    return log_p(theatae) - log_q(theatae)
    
