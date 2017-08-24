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

NUM_PEAKS = 10
DIM = 1
NUM_SAMPLES = 10 ** 2


# --- Generate Posterior from Generic Model `f` ---

def log_phi(mu, sigma):
    return - 0.5 * tf.square(mu / sigma)
    

# -- For instance
def f(x, theta):
    return tf.multiply(x, theta)


num_data = 100
noise_scale = 0.05

xs = np.linspace(-1, 1, num_data)
xs.astype(np.float32)

ys = 2 * xs
ys += noise_scale * np.random.normal(size=num_data)
ys.astype(np.float32)

y_sigmas = noise_scale * np.ones(shape=(num_data))
y_sigmas.astype(np.float32)

data = (xs, ys, y_sigmas)


def log_post_nv(theta):  # "nv" for "not vectorized".
    
    xs_t = tf.constant(xs, dtype=tf.float32)
    ys_t = tf.constant(ys, dtype=tf.float32)
    y_sigmas_t = tf.constant(y_sigmas, dtype=tf.float32)
    
    noises = tf.unstack(ys_t - f(xs_t, theta))
    sigmas = tf.unstack(y_sigmas_t)
    
    return tf.reduce_sum(tf.stack(
               [-0.5 * tf.square(noises[i] / sigmas[i])
                for i in range(num_data)]))


#def log_post_nv(theta):
#    return -0.5 * tf.square(theta - 100)

def log_post(thetas):
    return tf.map_fn(log_post_nv, thetas)
    



# -- Test

pnn = PostNN(NUM_PEAKS, DIM)
print('Model setup')
with Timer():
    pnn.compile(log_post, learning_rate=0.01)
    print('Model compiled.')

with tf.Session(graph=pnn.graph) as sess:
    
    writer = tf.summary.FileWriter('../dat/graphs', pnn.graph)
    sess.run(tf.global_variables_initializer())
    
    mean_theta = tf.reduce_mean(pnn.cgmd.sample(pnn.get_num_samples()),
                                axis=0)
    
    with Timer():
        
        print('Start fitting ......')
        
        for step in range(6000):
       
           loss_val, _ = sess.run([pnn.loss, pnn.optimize])
           #writer.add_summary(summary_val, global_step=step)

           if step % 10 == 0:
               print('step: {0}'.format(step))
               print('loss: {0}'.format(loss_val))
               print('theta: {0}'.format(mean_theta.eval()))
               print('-----------------------\n')
               
#    weights_val, mu_val, sigma_val = sess.run([pnn.weights, pnn.mu, pnn.sigma])
#    print('weights: {0}'.format(weights_val))
#    print('mu: {0}'.format(mu_val))
#    print('sigma: {0}'.format(sigma_val))
