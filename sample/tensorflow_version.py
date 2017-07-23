#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow version.
"""

import tensorflow as tf
from tensorflow.contrib.distributions import \
    Categorical, Mixture, MultivariateNormalDiag



NUM_PEAKS = 5
DIM = 100
NUM_SAMPLES = 10

TEST_NUM_PEAKS = 3


with tf.name_scope('inputs'):
    theta = tf.placeholder(shape=[None, DIM],
                           dtype=tf.float32)  # shape=[batch_size, DIM]
    
    
with tf.name_scope('q'):
    # Distribution that fits the posterior by minimizing KL-divergence
    
    with tf.name_scope('variables'):
        
        a = tf.Variable(tf.random_uniform([NUM_PEAKS]),
                        dtype=tf.float32)
        a_square = tf.square(a)
        w = tf.Variable(tf.random_uniform([DIM, NUM_PEAKS]),
                        dtype=tf.float32)
        w_square = tf.square(w)
        b = tf.Variable(tf.random_uniform([DIM, NUM_PEAKS]),
                        dtype=tf.float32)
        
    with tf.name_scope('log_q'):
        # -- Or in a function as below???
        
        beta = tf.log(a_square) \
             + tf.reduce_sum(
                   -0.5 * tf.square((tf.matmul(theta, w_square) + b)) \
                   +0.5 * tf.log(w_square / (2 * 3.14)),
                   axis=0)
         
        beta_max = tf.reduce_max(beta)
        delta_beta = beta - beta_max
        top_beta, _ = tf.nn.top_k(delta_beta, k=3)
        
        log_q = beta_max + tf.log(tf.reduce_sum(tf.exp(top_beta)))
    
    with tf.name_scope('dist_q'):
        
        w_square_list = tf.unstack(w_square, axis=1)
        b_list = tf.unstack(b, axis=1)
        assert len(w_square_list) == len(b_list)
        
        cat = Categorical(probs=a_square)
        components = [
            MultivariateNormalDiag(loc=b_list[i] / w_square_list[i], 
                                   scale_diag=1 / w_square_list[i])
            for i, _ in enumerate(b_list)
            ]
        dist_q = Mixture(cat=cat, components=components)
        

def log_q(theta):
    
    beta = tf.log(a_square) \
         + tf.reduce_sum(
               -0.5 * tf.square((tf.matmul(theta, w_square) + b)) \
               +0.5 * tf.log(w_square / (2 * 3.14)),
               axis=0)
     
    beta_max = tf.reduce_max(beta)
    delta_beta = beta - beta_max
    top_beta, _ = tf.nn.top_k(delta_beta, k=3)
    
    return beta_max + tf.log(tf.reduce_sum(tf.exp(top_beta)))


def kl_divergence(log_p, log_q, dist_q):
    
    theatae = dist_q.sample(sample_shape=[NUM_SAMPLES])  # tested.
    
    return log_p(theatae) - log_q(theatae)
    