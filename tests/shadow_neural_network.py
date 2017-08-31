#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description
-----------

Test on a shadow neural network.
"""

import sys
sys.path.append('../sample/')
from nn4post import PostNN
from tools import Timer
import tensorflow as tf
import numpy as np


# For testing (and debugging)
tf.set_random_seed(1234)
np.random.seed(1234)


# --- Model ---

NUM_HIDDEN = 10
DIM = NUM_HIDDEN * 3 + 1

def parse_params(params):
    
    w_h, w_a, b_h, b_a = tf.split(
        value=params,
        num_or_size_splits=[NUM_HIDDEN, NUM_HIDDEN, NUM_HIDDEN, 1])
    
    # shape: [1, num_hidden]
    w_h = tf.expand_dims(w_h, axis=0)
    # shape: [num_hidden, 1]
    w_a = tf.expand_dims(w_a, axis=1)
    
    return w_h, w_a, b_h, b_a
    
def shadow_neural_network(x, params):
    
    w_h, w_a, b_h, b_a = parse_params(params)
    
    # shape: [None, num_hidden]
    h = tf.nn.relu(tf.add(tf.matmul(x, w_h), b_h))
    # shape: [None, 1]
    a = tf.nn.relu(tf.add(tf.matmul(h, w_a), b_a))
    
    return a



# --- Data ---

def target_func(x):
    return np.sin(x)

num_data = 100
noise_scale = 0.1

x = np.linspace(-10, 10, num_data)
x = np.expand_dims(x, -1)  # shape: [num_data, 1]
x.astype(np.float32)

y = target_func(x)
y += noise_scale * np.random.normal(size=[num_data, 1])
y.astype(np.float32)

y_error = noise_scale * np.ones(shape=([num_data, 1]))
y_error.astype(np.float32)

class BatchGenerator(object):
    
    def __init__(self, x, y, y_error):
        
        self._x = x
        self._y = y
        self._y_error = y_error
        
    def __next__(self):
        
        return (self._x, self._y, self._y_error)
    
    
batch_generator = BatchGenerator(x, y, y_error)



# --- Test ---

#NUM_PEAKS = 1  # reduce to mean-field variational inference.
#NUM_PEAKS = 10
NUM_PEAKS = 100


pnn = PostNN(NUM_PEAKS, DIM, model=shadow_neural_network)
print('Model setup')


with Timer():
    pnn.compile(learning_rate=0.05)
    print('Model compiled.')
    

with Timer():
    
    pnn.fit(batch_generator, 300, verbose=True, skip_steps=10)    
        