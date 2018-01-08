
# -*- coding: utf-8 -*-
"""Test and Trail on Gaussian Mixture Distribution."""


import sys
import numpy as np
import tensorflow as tf
from tensorflow.contrib.distributions import (
    Categorical, NormalWithSoftplusScale, Mixture)

from nn4post import build_nn4post
try:
    from tensorflow.contrib.distributions import Independent
except:
    print('WARNING - Your TF < 1.4.0.')
    from nn4post.utils.independent import Independent



# For testing (and debugging)
SEED = 123456
tf.set_random_seed(SEED)
np.random.seed(SEED)


# -- Parameters
TARGET_N_C = 3  # shall be fixed.
N_D = 10**4
N_C = 5  # shall be varied.
N_SAMPLES = 10
A_RESCALE_FACTOR = 1.0
N_ITERS = 1 * 10**4
LR = tf.placeholder(shape=[], dtype='float32')
#OPTIMIZER = tf.train.AdamOptimizer(LR)
OPTIMIZER = tf.train.RMSPropOptimizer(LR)
DTYPE = 'float32'
SKIP_STEP = 50



# -- Gaussian Mixture Distribution
with tf.name_scope('posterior'):

  target_c = tf.constant([0.05, 0.25, 0.70])
  target_mu = tf.stack([
        tf.ones([N_D]) * (i - 1) * 3
        for i in range(TARGET_N_C)
      ], axis=0)
  target_zeta_val = np.zeros([TARGET_N_C, N_D])
  #target_zeta_val[1] = np.ones([N_D]) * 5.0
  target_zeta = tf.constant(target_zeta_val, dtype='float32')

  cat = Categorical(probs=target_c)
  components = [
      Independent(
          NormalWithSoftplusScale(target_mu[i], target_zeta[i])
      ) for i in range(TARGET_N_C)
    ]
  p = Mixture(cat, components)

  def log_posterior(theta):
    return p.log_prob(theta)

# test!
# test 1
init_var = {
    'a':
        np.zeros([N_C], dtype=DTYPE),
    'mu':
        np.array([np.ones([N_D]) * (i - 1) * 3 for i in range(N_C)],
                 dtype=DTYPE) \
        + np.array(np.random.normal(size=[N_C, N_D]) * 0.5,
                   dtype=DTYPE),
    'zeta':
        np.array(np.random.normal(size=[N_C, N_D]) * 5.0,
                 dtype=DTYPE),
}
# test 2
init_var = {
    'a':
        np.zeros([N_C], dtype=DTYPE),
    'mu':
        np.array( np.random.uniform(-1, 1, size=[N_C, N_D]) * 5.0 \
                  * np.sqrt(N_D),
                dtype=DTYPE),
    'zeta':
        np.ones([N_C, N_D], dtype=DTYPE) * 5.0,
}

n_samples = tf.placeholder(shape=[], dtype='float32', name='n_samples')
beta = tf.placeholder(shape=[], dtype='float32', name='beta')

ops, gvs = build_nn4post(N_C, N_D, log_posterior, init_var=init_var,
                         n_samples=N_SAMPLES, beta=beta)

train_op = OPTIMIZER.apply_gradients(gvs)


# -- Training
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    # Display Targets
    print(target_c.eval())
    print(target_mu.eval())
    print(target_zeta.eval())
    print()

    # Display Initialized Values
    print(ops['mu'].eval())
    print(ops['zeta'].eval())
    print()

    # Optimizing
    for i in range(N_ITERS):

        # C.f. the section "Re-scaling of a" of "/docs/nn4post.tm".
        # Even though, herein, `r` is tuned manually, it can be tuned
        # automatically (and smarter-ly), and should be so.
        # And learning-rate `LR` decays as usual.
        if i < 1000:
            lr_val = 0.5
            n_samples_val = 5
            beta_val = 1.0
        elif i < 3000:
            lr_val = 0.1
            n_samples_val = 5
            beta_val = 1.0
        elif i < 5000:
            lr_val = 0.02
            n_samples_val = 10
            beta_val = 1.0
        else:
            lr_val = 0.001
            n_samples_val = 100
            beta_val = 0.0

        _, loss_val, a_val, c_val, mu_val, zeta_val = \
            sess.run(
                [ train_op, ops['loss'], ops['a'],
                  ops['c'], ops['mu'], ops['zeta'] ],
                feed_dict={beta: beta_val, LR: lr_val}
            )

        # Display Trained Values
        if i % SKIP_STEP == 0:
            print('--- Step {0:5}  |  Loss {1}'.format(i, loss_val))
            print('c:\n', c_val)
            print('a:\n', a_val)
            print('mu (mean):\n', np.mean(mu_val, axis=1))
            print('mu (std):\n', np.std(mu_val, axis=1))
            print('zeta (mean):\n', np.mean(zeta_val, axis=1))
            print('zeta (std):\n', np.std(zeta_val, axis=1))
            print()

    print('--- SUMMARY ---')
    print()
    print('-- Parameters')
    print('n_d: ', N_D)
    print('n_c:', N_C)
    print('n_samples: ', N_SAMPLES)
    print('r: ', A_RESCALE_FACTOR)
    print('n_iters: ', N_ITERS)
    print('learning-rate: ', LR)
    print('dtype: ', DTYPE)
    print()
    print('-- Result')
    print('c:\n', c_val)
    print('a:\n', a_val)
    print('mu (mean):\n', np.mean(mu_val, axis=1))
    print('mu (std):\n', np.std(mu_val, axis=1))
    print('zeta (mean):\n', np.mean(zeta_val, axis=1))
    print('zeta (std):\n', np.std(zeta_val, axis=1))
    print()
