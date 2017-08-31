#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description
-----------

Simple test.
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


# --- Parameters ---

NUM_PEAKS = 100
#NUM_PEAKS = 1  # reduce to mean-field variational inference.
NUM_SAMPLES = 10 ** 4



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
def model(x, params):
    a, b, c = tf.unstack(params)
    return a * x + tf.tanh(b * tf.pow(x, 2) + c * tf.pow(x, 3))



# --- Data ---

def target_func(x):
    return 2 * x

num_data = 100
noise_scale = 0.1

x = np.linspace(-10, 10, num_data)
x = np.expand_dims(x, -1)
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

pnn = PostNN(NUM_PEAKS, DIM, model=model)
print('Model setup')


with Timer():
    pnn.compile(learning_rate=0.05)
    print('Model compiled.')
    

with Timer():
    
    pnn.fit(batch_generator, 300, verbose=True, skip_steps=10)    
        
