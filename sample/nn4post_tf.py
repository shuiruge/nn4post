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
        
        self.graph = tf.Graph()
        
    
    def compile(self, model, learning_rate):
        """ Set up the (TensorFlow) computational graph.
        
        Args:
            model:
                Callable, mapping from `(x, theta)` to `y`; wherein `x` is a
                `Tensor` representing the input data to the `model`, having
                shape `[batch_size, ...]` (`...` is for any sequence of `int`)
                and any dtype, so is the `y`; however, the `theta` must have
                the shape `[self.get_dim()]` and dtype `tf.float32`.

            learning_rate:
                `float`, the learning rate of optimizer `self.optimize`.
        """
        
        with self.graph.as_default():
            
            # shape: [batch_size, *model_input]
            self.x = tf.placeholder(tf.float32, name='x')
            # shape: [batch_size, *model_output]
            self.y = tf.placeholder(tf.float32, name='y')
            # shape: [batch_size, *model_output]
            self.y_error = tf.placeholder(tf.float32, name='y_error')
            
            
            def log_p_nv(theta):
                """ Chi-square.
                
                Args:
                    theta: `Tensor` with shape `[self.get_dim()]` and dtype
                           `tf.float32`.
                    
                Returns:
                    `Tensor` with shape `[]` and dtype `tf.float32`.
                """
                
                noise = tf.subtract(self.y, model(self.x, theta),
                                    name='noise')
                return tf.reduce_sum(-0.5 * tf.square(noise/self.y_error),
                                     name='log_p')
            
                
            def log_p(thetas):
                """ Chi-square.
                
                Args:
                    thetas: `Tensor` with shape `[None, self.get_dim()]` and
                            dtype `tf.float32`.
                    
                Returns:
                    `Tensor` with shape `[None]` and dtype `tf.float32`. The
                    two `None` shall both be the same value (e.g. both being
                    `self.get_num_samples()`).
                """
                
                return tf.map_fn(log_p_nv, thetas, name='vectorized_log_p')
            
        
            with tf.name_scope('trainable_variables'):
                
                a_shape = [self.get_num_peaks()]
                self.a = tf.Variable(
                    initial_value=tf.ones(a_shape),
                    dtype=tf.float32,
                    name='a')
                
                mu_shape = [self.get_num_peaks(), self.get_dim()]
                self.mu = tf.Variable(
                    initial_value=tf.zeros(mu_shape),
                    dtype=tf.float32,
                    name='mu')
                
                zeta_shape = [self.get_num_peaks(), self.get_dim()]
                self.zeta = tf.Variable(
                    initial_value=softplus_inverse(tf.ones(zeta_shape)),
                    dtype=tf.float32,
                    name='zeta')
                
            
            with tf.name_scope('CGMD_model'):
                
                with tf.name_scope('model_parameters'):
                    
                    self.weight = tf.nn.softmax(self.a, name='weight')
                    self.sigma = tf.nn.softplus(self.zeta, name='sigma')
                    
                with tf.name_scope('categorical'):
                    
                    cat = Categorical(probs=self.weight, name='cat')
                    
                with tf.name_scope('Gaussian'):
                    
                    mu_list = tf.unstack(self.mu, name='mu_list')
                    sigma_list = tf.unstack(self.sigma, name='sigma_list')
                    
                    components = [
                        MultivariateNormalDiag(
                            loc=mu_list[i],
                            scale_diag=sigma_list[i],
                            name='Gaussian_{0}'.format(i))
                        for i in range(self.get_num_peaks())]
                    
                with tf.name_scope('mixture'):
                    
                    self.cgmd = Mixture(cat=cat,
                                        components=components,
                                        name='CGMD')
                    
        
            with tf.name_scope('loss'):
                
                theta_samples = self.cgmd.sample(self.get_num_samples(),
                                          name='theta_samples')
                elbo = entropy.elbo_ratio(log_p, self.cgmd, z=theta_samples,
                                          name='ELBO')
                
                # In TensorFlow, ELBO is defined as E_q [log(p / q)], rather
                # than as E_q [log(q / p)] as WikiPedia. So, the loss would be
                # `-1 * elbo`
                self.loss = tf.multiply(-1.0, elbo, name='loss')

        
            with tf.name_scope('optimize'):
                
                # Using `GradientDescentOptimizer` will increase the loss and
                # raise an ERROR after several training-steps. However, e.g.
                # `AdamOptimizer` and `RMSPropOptimizer` naturally saves this.
                optimizer = tf.train.RMSPropOptimizer(learning_rate)
                self.optimize = optimizer.minimize(self.loss)
            
            
            with tf.name_scope('summary'):
                
                tf.summary.scalar('loss', self.loss)
                tf.summary.histogram('histogram_loss', self.loss)
                self.summary = tf.summary.merge_all()
                
    
    # --- Get-Functions ---
                
    def get_num_peaks(self):
        return self._num_peaks
    
    def get_dim(self):
        return self._dim
    
    def get_num_samples(self):
        return self._num_samples
