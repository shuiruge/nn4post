# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 21:57:19 2017

@author: pengxu.jiang
"""

from tools import Timer
import numpy as np
import tensorflow as tf
from tensorflow.contrib.distributions import \
    Categorical, Mixture, MultivariateNormalDiag
    
    
    
NUM_PEAKS = 5
DIM = 2
NUM_SAMPLES = 100


# Set trainable variables (parameters) of CGMD
a = tf.Variable(tf.random_uniform([NUM_PEAKS]),
                dtype=tf.float32)
weights = tf.nn.softmax(a)
mu = tf.Variable(tf.random_uniform([NUM_PEAKS, DIM]),
                 dtype=tf.float32)

zeta = tf.Variable(tf.random_uniform([NUM_PEAKS, DIM]),
                   dtype=tf.float32)
sigma = tf.nn.softplus(zeta)

# Construct CGMD as a `tf.distributions.Mixture` instance
cat = Categorical(probs=weights)
components = [
    MultivariateNormalDiag(
        loc=mu[i],
        scale_diag=sigma[i])
    for i in range(NUM_PEAKS)]
cgmd = Mixture(cat=cat, components=components)


#def elbo(log_p):
#    """
#    Args:
#        log_p: Map(tf.Tensor(shape=[None, DIM], dtype=tf.float32),
#                   tf.Tensor(shape=[], dtype=tf.float32))
#    Returns:
#        tf.Tensor(shape=[], dtype=tf.float32)
#    
#    XXX:
#        Or returns`tf.Op`???
#    """
#    thetae = cgmd.sample(NUM_SAMPLES)
#    return tf.reduce_mean(cgmd.log_prob(thetae) - log_p(thetae))
#
#
#
## -- Test
#def log_p(thetae):
#    return (-0.5 * tf.reduce_mean(tf.square(thetae), axis=1)
#            -0.5 * tf.log(2 * np.pi) * DIM)
#
#sess = tf.Session()
#init = tf.global_variables_initializer()
#sess.run(init)
#
##with sess:
##    
##    with Timer():
##        elbo_val = sess.run(elbo(log_p))
#            
## So far so good.
#
#
#elbo_op = elbo(log_p)
#grads_0 = tf.gradients(elbo_op, [a, mu, zeta])
#grads = [grads_0[i] + elbo_op for i in range(3)]
#
#
#with sess:
#    
#    elbo_val = sess.run(elbo_op)
#    grads_0_val = sess.run(grads_0)
#    grads_val = sess.run(grads)
#    
#    print(grads_val[0] - grads_0_val[0] - elbo_val)
#    # NOT correct.
    

# -- Try another way
thetae = cgmd.sample(NUM_SAMPLES)
def log_p(thetae):  # -- Test
    return (-0.5 * tf.reduce_mean(tf.square(thetae), axis=1)
            -0.5 * tf.log(2 * np.pi) * DIM)
elbo = tf.reduce_mean(cgmd.log_prob(thetae) - log_p(thetae))
grads_0 = tf.gradients(elbo, [a, mu, zeta])
grads = [grads_0[i] + elbo for i in range(3)]




sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

with sess:
    
    with Timer():
        elbo_val, grads_0_val, grads_val = sess.run([elbo, grads_0, grads])
    
    for i in range(3):
        print(grads_val[i] - grads_0_val[i] - elbo_val)  # shall print zeros.

''' NOTE:
    
    As shown in ..:code:
        
        elbo_val, grads_0_val, grads_val = sess.run([elbo, grads_0, grads])
        
        for i in range(3):
            print(grads_val[i] - grads_0_val[i] - elbo_val)  # shall print zeros.
        
        >>> [ 0.  0.  0.  0.  0.]
            [[ 0.  0.]
             [ 0.  0.]
             [ 0.  0.]
             [ 0.  0.]
             [ 0.  0.]]
            [[  0.00000000e+00   0.00000000e+00]
             [  0.00000000e+00   0.00000000e+00]
             [  0.00000000e+00  -5.96046448e-08]
             [  0.00000000e+00   0.00000000e+00]
             [  0.00000000e+00   0.00000000e+00]]
            
    `elbo`, `grads_0` and `grads` shall be run in one `sess.run()`, rather than
    ..:code:
        
        elbo_val = sess.run(elbo)
        grads_0_val = sess.run(grads_0)
        grads_val = sess.run(grads)
        
        for i in range(3):
            print(grads_val[i] - grads_0_val[i] - elbo_val)
            # shall print zeros, but NOT so.
    
    The reason is that any time in one `sess.run()` calling any quantity that
    depends one `thetae`, the `cgmd` will do sampling once. So, to keep `thetae`
    overall in one epoch, all that depends on `thetae` shall be called in one
    `sess.run()`.
'''
