#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple test.
"""

import sys
sys.path.append('../sample/')
from nn4post_tf import PostNN
from tools import Timer
import tensorflow as tf
import numpy as np

NUM_PEAKS = 3
DIM = 1
NUM_SAMPLES = 1000


# --- Generate Posterior from Generic Model `f` ---

def log_phi(mu, sigma):
    return (-0.5 * np.log(2 * np.pi)
            - np.log(sigma)
            - 0.5 * np.square(mu / sigma))
    
# -- For instance
def f(x, theta):
    return x * theta

num_data = 100
xs = np.linspace(-1, 1, num_data)
ys = xs
y_sigmas = np.square(np.random.normal(scale=0.1, size=num_data))
data = (xs, ys, y_sigmas)

def log_likelihood(theta):
    noises = [ys[i] - f(xs[i], theta) for i in range(num_data)]
    return np.sum([log_phi(noises[i], y_sigmas[i]) for i in range(num_data)])
    
def log_post(thetae):
    num_theta = thetae.shape[0]
    return np.sum([log_likelihood(thetae[i,0]) for i in range(num_theta)])

#def log_post(thetae):
#    return -0.5 * np.square(thetae - 100)
    


# -- Test

pnn = PostNN(NUM_PEAKS, DIM)
print('haha')
pnn.compile(log_post, learning_rate=0.01)
print('hahaha')

with tf.Session(graph=pnn.graph) as sess:
    
    writer = tf.summary.FileWriter('../dat/graphs', pnn.graph)
    sess.run(tf.global_variables_initializer())
    
    with Timer():
        
        print('Start training ......')
        
        for step in range(6000):
       
           elbo_val, _, summary_val = sess.run([pnn.elbo, pnn.optimize, pnn.summary])
           writer.add_summary(summary_val, global_step=step)

           if step % 100 == 0:
               print('step: {0}'.format(step))
               print('elbo: {0}'.format(elbo_val))
               print('-----------------------\n')
               
    weights_val, mu_val, sigma_val = sess.run([pnn.weights, pnn.mu, pnn.sigma])
    print('weights: {0}'.format(weights_val))
    print('mu: {0}'.format(mu_val))
    print('sigma: {0}'.format(sigma_val))
