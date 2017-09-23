#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description
-----------
Test on a shallow neural network.

TODO: Needs further test on the lower limit of loss for each `N_PEAKS`.
      However, this can be estabilish only  after finishing the `Trainer()`.
"""

import sys
sys.path.append('../sample/')
from nn4post_tf import Nn4post, error
from tools import Timer
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
import mnist_loader



# For testing (and debugging)
tf.set_random_seed(42)
np.random.seed(42)


# --- Model ---

n_input_features = 28 * 28
n_hidden = 100
n_output_features = 10


split_shapes = [

    # -- `w`s.
    n_input_features*n_hidden, n_hidden*n_output_features,

    # -- `b`s.
    n_hidden, n_output_features,

]


def parse_params(params):

    w_1, w_a, b_1, b_a = tf.split(
        value=params,
        num_or_size_splits=split_shapes)

    # shape: [n_input_features, n_hidden]
    w_1 = tf.reshape(w_1, [n_input_features, n_hidden])
    # shape: [n_hidden, n_output_features]
    w_a = tf.reshape(w_a, [n_hidden, n_output_features])

    return w_1, w_a, b_1, b_a


def shallow_neural_network(x, params):
    """
    Args:
        x: `Tensor` with shape `[None, 1]`
        params: `Tensor`
    Returns:
        `Tensor` with shape `[None, 1]`.
    """

    w_1, w_a, b_1, b_a = parse_params(params)

    # -- Hidden Layer 1
    # shape: [None, n_hidden]
    h_1 = tf.sigmoid(tf.matmul(x, w_1) + b_1)

    # -- Output Layer
    # shape: [None, n_output_features]
    a = tf.nn.softmax(tf.matmul(h_1, w_a) + b_a)

    return a


DIM = int(sum(split_shapes))  # dimension of parameter-space.
print('DIM: {0}'.format(DIM))


lambda_ = 1.0
def log_prior(theta, lambda_=lambda_):
    """ C.f. section 2.1.1 of Neal (1995), wherein the priors of weights are
        averaged, while the priors of biases are uniform.

    ```math

    \ln p(w, b) = \left( 1/N_w \sum_{i}^{N_w} \right)
                  \left(     -1/2 w_i^2       \right)
    ````

    Args:
        theta: `Tensor` with the shape `[None]`.

        lambda_:
            `float`, optional, as the regularization factor introduced in
            section 2.3.1 of Nealson.

    Returns:
        `Tensor` with the shape `[]`.
    """

    w_1, w_a, b_1, b_a = parse_params(theta)

    prior_from_weights = -0.5 * tf.reduce_mean(tf.square(w_1)) \
                         -0.5 * tf.reduce_mean(tf.square(w_a))

    return lambda_ * prior_from_weights



# --- Data ---


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()


# -- Training Data
x, y = training_data
x = np.asarray(x, dtype=np.float32)
x = np.squeeze(x, -1)
y = np.asarray(y, dtype=np.float32)
y = np.squeeze(y, -1)

print('x:', x.shape, x.dtype)
print('y:', y.shape, y.dtype)

noise_scale = 0.1
y_error = noise_scale * np.ones(y.shape, dtype=np.float32)

# -- Testing Data
x_test = [_[0].astype(np.float32) for _ in test_data]
y_test = [_[1] for _ in test_data]


# -- Batch Generator

def get_batch_generator(x, y, y_error, batch_size):

    n_data = len(x)

    while True:

        if batch_size is None:
            yield x, y, y_error

        else:
            x, y, y_error = shuffle(x, y, y_error)  # XXX: copy ???
            batches = [
                (  x[k:k+batch_size],
                   y[k:k+batch_size],
                   y_error[k:k+batch_size],
                )
                for k in range(0, n_data, batch_size)
            ]
            yield batches


batch_size = 128
batch_generator = get_batch_generator(x, y, y_error, batch_size)



# --- Test ---

#N_PEAKS = 1  # reduce to mean-field variational inference.
#N_PEAKS = 2
N_PEAKS = 5
#N_PEAKS = 10


nn4post = Nn4post(n_peaks=N_PEAKS,
             dim=DIM,
             model=shallow_neural_network,
             log_prior=log_prior,
             float_='float32')
print('Model setup')


with Timer():
    #optimizer = tf.train.RMSPropOptimizer
    optimizer = tf.train.AdamOptimizer
    metrics = [error]
    nn4post.compile(
        optimizer=optimizer,
        metrics=metrics,
        summarize_variables=True)
    print('Model compiled.')


learning_rate = 0.1
print('\n--- Parameters:\n\t--- N_PEAKS: {0},  learning_rate: {1}\n'
      .format(N_PEAKS, learning_rate))


with Timer():
    nn4post.fit(batch_generator=batch_generator,
        epochs=5,
        #epochs=3,  # test!
        learning_rate=learning_rate,
        batch_ratio=1.0,
        logdir='../dat/graph/on_mnist_{0}_{1}'\
            .format(N_PEAKS, n_hidden),
        dir_to_ckpt='../dat/checkpoint/on_mnist_{0}_{1}'\
            .format(N_PEAKS, n_hidden),
        skip_steps=50,
    )


# Test
n_data = x_test.shape[0]
predicted = nn4post.predict(x_test)
print(predicted.shape, y_test[0].shape)
predicted = [np.argmax(predicted[i]) for i in range(n_data)]


n_correct = 0
for i in range(n_data):
    if int(y_test[i]) == predicted[i]:
        n_correct += 1
print('Acc: {0}'.format(n_correct/n_data))



nn4post.finalize()
