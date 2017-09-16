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
from nn4post import Nn4post
from tools import Timer
import tensorflow as tf
import numpy as np
import mnist_loader


# For testing (and debugging)
tf.set_random_seed(42)
np.random.seed(42)


# --- Model ---

num_input_feature = 28 * 28
num_hidden_1 = 100
num_output_feature = 10


split_shapes = [

    # -- `w`s.
    num_input_feature*num_hidden_1, num_hidden_1*num_output_feature,

    # -- `b`s.
    num_hidden_1, num_output_feature,

]


def parse_params(params):

    w_1, w_a, b_1, b_a = tf.split(
        value=params,
        num_or_size_splits=split_shapes)

    # shape: [num_input_feature, num_hidden_1]
    w_1 = tf.reshape(w_1, [num_input_feature, num_hidden_1])
    # shape: [num_hidden_1, num_output_feature]
    w_a = tf.reshape(w_a, [num_hidden_1, num_output_feature])

    return w_1, w_a, b_1, b_a


def shadow_neural_network(x, params):
    """
    Args:
        x: `Tensor` with shape `[None, 1]`
        params: `Tensor`
    Returns:
        `Tensor` with shape `[None, 1]`.
    """

    w_1, w_a, b_1, b_a = parse_params(params)

    # -- Hidden Layer 1
    # shape: [None, num_hidden_1]
    h_1 = tf.sigmoid(tf.matmul(x, w_1) + b_1)

    # -- Output Layer
    # shape: [None, num_output_feature]
    a = tf.nn.softmax(tf.matmul(h_1, w_a) + b_a)

    return a


DIM = int(sum(split_shapes))  # dimension of parameter-space.
print('DIM: {0}'.format(DIM))


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


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()


# -- Training Data
x, y = training_data
x = np.asarray(x, dtype=np.float32)
x = np.squeeze(x, -1)
y = np.asarray(y, dtype=np.float32)
y = np.squeeze(y, -1)

print(x.shape, x.dtype)
print(y.shape, y.dtype)

noise_scale = 0.1
y_error = noise_scale * np.ones(y.shape, dtype=np.float32)

# -- Testing Data
x_test = [_[0].astype(np.float32) for _ in test_data]
y_test = [_[1] for _ in test_data]


# -- Batch Generator

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


#batch_generator = BatchGenerator(x, y, y_error, batch_size=None)
batch_size = 256
batch_generator = BatchGenerator(x, y, y_error, batch_size)



# --- Test ---

#NUM_PEAKS = 1  # reduce to mean-field variational inference.
#NUM_PEAKS = 2
NUM_PEAKS = 5
#NUM_PEAKS = 10


nn4post = Nn4post(num_peaks=NUM_PEAKS,
             dim=DIM,
             model=shadow_neural_network,
             log_prior=log_prior,
             float_='float32')
print('Model setup')


with Timer():
    #optimizer = tf.train.RMSPropOptimizer
    optimizer = tf.train.AdamOptimizer
    nn4post.compile(optimizer=optimizer)
    print('Model compiled.')


learning_rate = 0.01
print('\n--- Parameters:\n\t--- NUM_PEAKS: {0},  learning_rate: {1}\n'
      .format(NUM_PEAKS, learning_rate))


with Timer():
    nn4post.fit(batch_generator=batch_generator,
        epochs=1000,
        #epochs=3,  # test!
        learning_rate=learning_rate,
        batch_ratio=1.0,
        logdir='../dat/graph/on_mnist_{0}_{1}'\
            .format(NUM_PEAKS, num_hidden_1),
        dir_to_ckpt='../dat/checkpoint/on_mnist_{0}_{1}'\
            .format(NUM_PEAKS, num_hidden_1),
        skip_steps=50,
    )


## Test
#num_data = x_test.shape[0]
#predicted = nn4post.predict(x_test)
#predicted = [np.argmax(predicted[i]) for i in range(num_data)]
#
#
#num_correct = 0
#for i in range(num_data):
#    if int(y_test[i]) == predicted[i]:
#        num_correct += 1
#print('Acc: {0}'.format(num_correct/num_data))



nn4post.finalize()
