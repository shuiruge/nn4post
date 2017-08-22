# -*- coding: utf-8 -*-
"""
TensorFlow version.
"""


import tensorflow as tf
from tensorflow.contrib.distributions import \
    Categorical, Mixture, MultivariateNormalDiag
from tensorflow.contrib.distributions import softplus_inverse


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
            log_post: Map(np.array(shape=[None, self.get_dim()],
                                   dtype=float32),
                          float)
            learning_rate: float
        """
        
        self.graph = tf.Graph()

        with self.graph.as_default():
        
            with tf.name_scope('trainable_variables'):
                self.a = tf.Variable(tf.ones([self.get_num_peaks()]),
                                dtype=tf.float32,
                                name='a')
                self.mu = tf.Variable(tf.zeros([self.get_num_peaks(), self.get_dim()]),
                                 dtype=tf.float32,
                                 name='mu')
                self.zeta = tf.Variable(softplus_inverse(
                                       tf.ones([self.get_num_peaks(), self.get_dim()])),
                                   dtype=tf.float32,
                                   name='zeta')
            
            with tf.name_scope('CGMD_model'):
                with tf.name_scope('model_parameters'):
                    self.weights = tf.nn.softmax(self.a, name='weights')
                    self.sigma = tf.nn.softplus(self.zeta, name='sigma')
                with tf.name_scope('categorical'):
                    cat = Categorical(probs=self.weights)
                with tf.name_scope('Gaussian'):
                    components = [
                        MultivariateNormalDiag(
                            loc=self.mu[i],
                            scale_diag=self.sigma[i])
                        for i in range(self.get_num_peaks())]
                with tf.name_scope('mixture'):
                    self.cgmd = Mixture(cat=cat, components=components, name='CGMD')
                
            with tf.name_scope('sampling'):
                thetae = self.cgmd.sample(self.get_num_samples(), name='thetae')
                    
            with tf.name_scope('ELBO'):
                mc_integrand = self.cgmd.log_prob(thetae) - log_post(thetae)
                self.elbo = tf.reduce_mean(mc_integrand, name='ELBO')
        
            with tf.name_scope('optimize'):
                optimizer = tf.train.GradientDescentOptimizer(learning_rate)
                self.optimize = optimizer.minimize(self.elbo)
            
            with tf.name_scope('summary'):
                tf.summary.scalar('ELBO', self.elbo)
                tf.summary.histogram('histogram_ELBO', self.elbo)
                self.summary = tf.summary.merge_all()
                
    
    # --- Get-Functions ---
                
    def get_num_peaks(self):
        return self._num_peaks
    
    def get_dim(self):
        return self._dim
    
    def get_num_samples(self):
        return self._num_samples