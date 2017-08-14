# -*- coding: utf-8 -*-
"""
"""

from tools import Timer
import numpy as np
import tensorflow as tf
from tensorflow.contrib.distributions import \
    Categorical, Mixture, MultivariateNormalDiag
    
    
    
NUM_PEAKS = 5
DIM = 2
NUM_SAMPLES = 1000


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
    return (-0.5 * tf.reduce_mean(tf.square(thetae-100), axis=1)
            -0.5 * tf.log(2 * np.pi) * DIM)
elbo = tf.reduce_mean(cgmd.log_prob(thetae) - log_p(thetae))
grads_0 = tf.gradients(elbo, [a, mu, zeta])
grads = [grads_0[i] + elbo for i in range(3)]




''' NOTE:
    
    As shown in ..:code:
        
        with tf.Session() as sess:
            
            sess.run(initializer)
        
            elbo_val, grads_0_val, grads_val = sess.run([elbo, grads_0, grads])
        
            for i in range(3):
                
                # Shall print zeros
                print(grads_val[i] - grads_0_val[i] - elbo_val)
        
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


## --- WITHOUT Modification of Gradients ---
#
## Un-comment this block if test WITHOUT modification of gradients
#
#optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
#optimize = optimizer.minimize(elbo)
#
## -- Test
#with tf.Session() as sess:
#    
#    sess.run(tf.global_variables_initializer())
#    
#    with Timer():
#        
#        for i in range(50000):
#       
#           elbo_val, _ = sess.run([elbo, optimize])
#
#           if i % 100 == 0:
#               print('step: {0}'.format(i))
#               print('elbo: {0}'.format(elbo_val))
#               print('theta instance: {0}'.format(sess.run(thetae)[0]))



# --- WITH Modification of Gradients ---

## Un-comment this block if test WITH modification of gradients

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
direct_grads_to_vars = optimizer.compute_gradients(elbo)
real_grads_to_vars = [
    (grad + elbo, var)
    if grad is not None else (None, var)
    for grad, var in direct_grads_to_vars]
optimize = optimizer.apply_gradients(real_grads_to_vars)

# -- Test
with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    
    with Timer():
        
        for i in range(50000):
       
           elbo_val, _ = sess.run([elbo, optimize])
           
           if i % 100 == 0:
               print('step: {0}'.format(i))
               print('elbo: {0}'.format(elbo_val))
               print('theta instance: {0}'.format(sess.run(thetae)[0]))
               
               
''' Conclusion:
    
    Without the modification of gradients in `optimizer` for ELBO, the
    optimization works surprsingly great. However, also surprisingly, when with
    the modification, the optimization blow up quickly.
    
    Can it be possible that TensorFlow has handed the modification for ELBO???
    
    TODO: To answer this, we need a double-check by numpy (i.e. `nn4post_np`).
'''
