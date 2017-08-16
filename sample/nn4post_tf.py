# -*- coding: utf-8 -*-
"""
TensorFlow version.
"""

from tools import Timer
import numpy as np
import tensorflow as tf
from tensorflow.contrib.distributions import \
    Categorical, Mixture, MultivariateNormalDiag
from tensorflow.contrib.distributions import softplus_inverse
    
    
NUM_PEAKS = 1
DIM = 1
NUM_SAMPLES = 1000


graph = tf.Graph()
    
def log_p(thetae):  # -- Test
    return (-0.5 * tf.reduce_sum(tf.square(thetae-100), axis=1)
            -0.5 * tf.log(2 * np.pi) * DIM)

with graph.as_default():

    with tf.name_scope('trainable_variables'):
        a = tf.Variable(tf.ones([NUM_PEAKS]),
                        dtype=tf.float32,
                        name='a')
        mu = tf.Variable(tf.zeros([NUM_PEAKS, DIM]),
                         dtype=tf.float32,
                         name='mu')
        zeta = tf.Variable(softplus_inverse(
                               tf.ones([NUM_PEAKS, DIM])),
                           dtype=tf.float32,
                           name='zeta')
    
    with tf.name_scope('CGMD_model'):
        with tf.name_scope('model_parameters'):
            weights = tf.nn.softmax(a, name='weights')
            sigma = tf.nn.softplus(zeta, name='sigma')
        with tf.name_scope('categorical'):
            cat = Categorical(probs=weights)
        with tf.name_scope('Gaussian'):
            components = [
                MultivariateNormalDiag(
                    loc=mu[i],
                    scale_diag=sigma[i])
                for i in range(NUM_PEAKS)]
        with tf.name_scope('mixture'):
            cgmd = Mixture(cat=cat, components=components, name='CGMD')
        
    with tf.name_scope('sampling'):
        thetae = cgmd.sample(NUM_SAMPLES, name='thetae')
            
    with tf.name_scope('ELBO'):
        mc_integrand = cgmd.log_prob(thetae) - log_p(thetae)
        elbo = tf.reduce_mean(mc_integrand, name='ELBO')

    with tf.name_scope('optimize'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        optimize = optimizer.minimize(elbo)
    
    with tf.name_scope('summary'):
        tf.summary.scalar('ELBO', elbo)
        tf.summary.histogram('histogram_ELBO', elbo)
        summary = tf.summary.merge_all()


# -- Test
with tf.Session(graph=graph) as sess:
    
    writer = tf.summary.FileWriter('../dat/graphs', graph)
    sess.run(tf.global_variables_initializer())
    
    with Timer():
        
        for step in range(6000):
       
           elbo_val, _, summary_val = sess.run([elbo, optimize, summary])
           writer.add_summary(summary_val, global_step=step)

           if step % 100 == 0:
               print('step: {0}'.format(step))
               print('elbo: {0}'.format(elbo_val))
               print('theta sample: {0}'.format(sess.run(thetae)[0]))
               print('-----------------------\n')
               
    weights_val, mu_val, sigma_val = sess.run([weights, mu, sigma])
    print('weights: {0}'.format(weights_val))
    print('mu: {0}'.format(mu_val))
    print('sigma: {0}'.format(sigma_val))
