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
NUM_SAMPLES = 10 ** 4


    

# -- For instance
def model(x, theta):
    return tf.multiply(x, theta)


num_data = 100
noise_scale = 0.05

xs = np.linspace(-1, 1, num_data)
xs = np.expand_dims(xs, -1)
xs.astype(np.float32)

ys = 2 * xs
ys += noise_scale * np.random.normal(size=[num_data, 1])
ys.astype(np.float32)

y_errors = noise_scale * np.ones(shape=([num_data, 1]))
y_errors.astype(np.float32)



# -- Test

pnn = PostNN(NUM_PEAKS, DIM)
print('Model setup')


with Timer():
    learning_rate=0.05
    pnn.compile(model, learning_rate=learning_rate)
    print('Model compiled.')
    

with tf.Session(graph=pnn.graph) as sess:
    
    writer = tf.summary.FileWriter('../dat/graphs', pnn.graph)
    sess.run(tf.global_variables_initializer())
    
    mean_theta = tf.reduce_mean(pnn.cgmd.sample(pnn.get_num_samples()),
                                axis=0)
    
    with Timer():
        
        print('Start fitting ......')
        
        feed_dict = {
            pnn.x: xs,
            pnn.y: ys,
            pnn.y_error: y_errors,
            }
        
        for step in range(300):
        
            _, loss_val, summary_val = sess.run(
                    [pnn.optimize, pnn.loss, pnn.summary],
                    feed_dict=feed_dict)
            writer.add_summary(summary_val, global_step=step)

            if step % 1 == 0:
                print('step: {0}'.format(step))
                print('loss: {0}'.format(loss_val))
                print('theta: {0}'.format(mean_theta.eval()))
                print('-----------------------\n')
               
#    weights_val, mu_val, sigma_val = sess.run([pnn.weights, pnn.mu, pnn.sigma])
#    print('weights: {0}'.format(weights_val))
#    print('mu: {0}'.format(mu_val))
#    print('sigma: {0}'.format(sigma_val))
