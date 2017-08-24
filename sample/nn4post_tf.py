#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow version.
"""


import tensorflow as tf
from tensorflow.contrib.distributions import \
    Categorical, Mixture, MultivariateNormalDiag
from tensorflow.contrib.distributions import softplus_inverse
from tensorflow.contrib.bayesflow import entropy


class PostNN(object):
    """
    TODO: complete it.
    TODO: write `self.fit()`.
    """
    
    def __init__(self, num_peaks, dim, num_samples=1000):
        self._num_peaks = num_peaks
        self._dim = dim
        self._num_samples = num_samples
        
    
    def compile(self, log_post, learning_rate):
        """ Set up the computation-graph.
        
        Args:
            log_post: Map(tf.Tensor(shape=[None, self.get_dim()],
                                    dtype=float32),
                          tf.Tensor(shape=[None],
                                    dtype=float32))
                where the two `None`s shall be the same number in practice,
                indicating the number of samples of parameters ("theta").
            learning_rate: float
        """
        
        self.graph = tf.Graph()

        with self.graph.as_default():
        
            with tf.name_scope('trainable_variables'):
                self.a = tf.Variable(
                    initial_value=tf.ones([self.get_num_peaks()]),
                    dtype=tf.float32,
                    name='a')
                self.mus = [
                    tf.Variable(
                        initial_value=tf.zeros([self.get_dim()]),
                        dtype=tf.float32,
                        name='mu_{0}'.format(i))
                    for i in range(self.get_num_peaks())]
                self.zetas = [
                    tf.Variable(
                        initial_value=\
                            softplus_inverse(tf.ones([self.get_dim()])),
                        dtype=tf.float32,
                        name='zeta_{0}'.format(i))
                    for i in range(self.get_num_peaks())]
            
            with tf.name_scope('CGMD_model'):
                with tf.name_scope('model_parameters'):
                    self.weights = tf.nn.softmax(self.a, name='weights')
                    self.sigmas = [
                        tf.nn.softplus(zeta, name='sigma_{0}'.format(i))
                        for i, zeta in enumerate(self.zetas)]
                with tf.name_scope('categorical'):
                    cat = Categorical(probs=self.weights)
                with tf.name_scope('Gaussian'):
                    components = [
                        MultivariateNormalDiag(
                            loc=self.mus[i],
                            scale_diag=self.sigmas[i])
                        for i in range(self.get_num_peaks())]
                with tf.name_scope('mixture'):
                    self.cgmd = Mixture(cat=cat, components=components,
                                        name='CGMD')
        
            with tf.name_scope('loss'):
                elbo = entropy.elbo_ratio(log_post, self.cgmd, n=100)
                # In TensorFlow, ELBO is defined as E_q [log(p / q)], rather
                # than as E_q [log(q / p)] as WikiPedia. So, the loss would be
                # `-1 * elbo`
                self.loss = -1 * elbo
        
            with tf.name_scope('optimize'):
                optimizer = tf.train.GradientDescentOptimizer(learning_rate)
                self.optimize = optimizer.minimize(self.loss)
            
#            with tf.name_scope('summary'):
#                tf.summary.scalar('loss', self.loss)
#                tf.summary.histogram('histogram_loss', self.loss)
#                self.summary = tf.summary.merge_all()
                
    
    # --- Get-Functions ---
                
    def get_num_peaks(self):
        return self._num_peaks
    
    def get_dim(self):
        return self._dim
    
    def get_num_samples(self):
        return self._num_samples
