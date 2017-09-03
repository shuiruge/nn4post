#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description
-----------
Test on a shadow neural network.

TODO: Needs further test on the lower limit of loss for each `NUM_PEAKS`.
      However, this can be estabilish only  after finishing the `Trainer()`.
"""

import sys
sys.path.append('../sample/')
from nn4post import PostNN
from tools import Timer
import tensorflow as tf
import numpy as np


# For testing (and debugging)
tf.set_random_seed(42)
np.random.seed(42)


# --- Model ---

num_hidden_1 = 10
num_hidden_2 = 10

split_shapes = [
    num_hidden_1, num_hidden_1 * num_hidden_2, num_hidden_2,  # `w`s.
    num_hidden_1, num_hidden_2, 1,  # `b`s.
]


def parse_params(params):

    w_1, w_2, w_a, b_1, b_2, b_a = tf.split(
        value=params,
        num_or_size_splits=split_shapes)

    # shape: [1, num_hidden_1]
    w_1 = tf.reshape(w_1, [1, num_hidden_1])
    # shape: [num_hidden_1, num_hidden_2]
    w_2 = tf.reshape(w_2, [num_hidden_1, num_hidden_2])
    # shape: [num_hidden_2, 1]
    w_a = tf.reshape(w_a, [num_hidden_2, 1])

    return w_1, w_2, w_a, b_1, b_2, b_a


def shadow_neural_network(x, params):
    """
    Args:
        x: `Tensor` with shape `[None, 1]`
        params: `Tensor`
    Returns:
        `Tensor` with shape `[None, 1]`.
    """

    w_1, w_2, w_a, b_1, b_2, b_a = parse_params(params)

    # -- Hidden Layer 1
    # shape: [None, num_hidden_1]
    h_1 = tf.tanh(tf.matmul(x, w_1) + b_1)

    # -- Hidden Layer 2
    # shape: [None, num_hidden_2]
    h_2 = tf.tanh(tf.matmul(h_1, w_2) + b_2)

    # -- Output Layer
    # shape: [None, 1]
    a = tf.tanh(tf.matmul(h_2, w_a) + b_a)

    return a


DIM = int(sum(split_shapes))  # dimension of parameter-space.


def log_prior(theta):
    """
    ```math

    p(\theta) = \prod_i^d \exp \left( -1/2 theta_i^2 \right)
    ````

    Args:
        theta: `Tensor` with the shape `[None]`.

    Returns:
        `Tensor` with the shape `[]`.
    """

    return -0.5 * tf.reduce_sum(tf.square(theta))



# --- Data ---

def target_func(x):
    return np.sin(x) * 0.5

num_data = 100
noise_scale = 0.1

x = np.linspace(-7, 7, num_data)
x = np.expand_dims(x, -1)  # shape: [num_data, 1]
x.astype(np.float32)

y = target_func(x)
y += noise_scale * np.random.normal(size=[num_data, 1])
y.astype(np.float32)

y_error = noise_scale * np.ones(shape=([num_data, 1]))
y_error.astype(np.float32)


class BatchGenerator(object):

    def __init__(self, x, y, y_error, batch_size):

        self._x = x
        self._y = y
        self._y_error = y_error
        self._num_data = x.shape[0]
        self._batch_size = batch_size

    def __next__(self):

        if self._batch_size is None:
            return(self._x, self._y, self._y_error)

        else:
            ids = np.random.randint(0, self._num_data-1, size=self._batch_size)
            x = np.array([self._x[i] for i in ids])
            y = np.array([self._y[i] for i in ids])
            y_error = np.array([self._y_error[i] for i in ids])
            return (x, y, y_error)


batch_generator = BatchGenerator(x, y, y_error, batch_size=None)
#batch_generator = BatchGenerator(x, y, y_error, batch_size=int(num_data/3))



# --- Test ---

#NUM_PEAKS = 1  # reduce to mean-field variational inference.
#NUM_PEAKS = 5
NUM_PEAKS = 25


pnn = PostNN(num_peaks=NUM_PEAKS,
             dim=DIM,
             model=shadow_neural_network,
             log_prior=log_prior)
print('Model setup')


with Timer():
    learning_rate = 0.03
    pnn.compile(learning_rate=learning_rate, optimizer=tf.train.AdamOptimizer)
    print('Model compiled.')


print('\n--- Parameters:\n\t--- NUM_PEAKS: {0},  learning_rate: {1}\n'
      .format(NUM_PEAKS, learning_rate))


with Timer():
    pnn.fit(batch_generator, 3000, verbose=True, skip_steps=10)


predicted = pnn.predict(x)

import matplotlib.pyplot as plt
plt.plot(x, target_func(x), '-')
plt.plot(x, predicted.reshape(-1), '--')
plt.plot(x, y, '.')
plt.show()


pnn.finalize()
