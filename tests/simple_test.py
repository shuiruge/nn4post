#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple test.
"""

import sys
sys.path.append('../sample/')
from nn4post import PostNN
from tools import Timer
import tensorflow as tf
import numpy as np


# --- Parameters ---

NUM_PEAKS = 100
#NUM_PEAKS = 1  # reduce to mean-field variational inference.
NUM_SAMPLES = 10 ** 4

SKIP_STEPS = 1


# --- Model ---
## -- For instance 1
#DIM = 1
#def model(x, theta):
#    return theta * x


## -- For instance 2
#DIM = 2
#def model(x, theta):
#    a, b = tf.unstack(theta)
#    return a * x + b * tf.square(x, 2)


## -- For instance 3
#DIM = 3
#def model(x, theta):
#    a, b, c = tf.unstack(theta)
#    return a * x + b * tf.pow(x, 2) + c * tf.pow(x, 3)


# -- For instance 3
DIM = 3
def model(x, theta):
    a, b, c = tf.unstack(theta)
    return a * x + tf.tanh(b * tf.pow(x, 2) + c * tf.pow(x, 3))



# --- Data ---

num_data = 100
noise_scale = 0.1

xs = np.linspace(-10, 10, num_data)
xs = np.expand_dims(xs, -1)
xs.astype(np.float32)

ys = 2 * xs
ys += noise_scale * np.random.normal(size=[num_data, 1])
ys.astype(np.float32)

y_errors = noise_scale * np.ones(shape=([num_data, 1]))
y_errors.astype(np.float32)



# --- Test ---

pnn = PostNN(NUM_PEAKS, DIM)
print('Model setup')


with Timer():
    learning_rate=0.05
    pnn.compile(model=model,
                learning_rate=learning_rate)
    print('Model compiled.')
    

sess = tf.Session(graph=pnn.graph)

# test! for debug
#from tensorflow.python import debug as tf_debug
#sess = tf_debug.LocalCLIDebugWrapperSession(sess)
#sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

with sess:
    
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

            if step % SKIP_STEPS == 0:
                print('step: {0}'.format(step))
                print('loss: {0}'.format(loss_val))
                print('theta: {0}'.format(mean_theta.eval()))
                print('-----------------------\n')
