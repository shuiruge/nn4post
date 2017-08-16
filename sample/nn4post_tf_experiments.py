# -*- coding: utf-8 -*-
"""
TensorFlow version. For experiments.
"""

from tools import Timer
import numpy as np
import tensorflow as tf
from tensorflow.contrib.distributions import \
    Categorical, Mixture, MultivariateNormalDiag
from tensorflow.contrib.distributions import softplus_inverse
    
    
NUM_PEAKS = 3
DIM = 2
NUM_SAMPLES = 1000


# -- Set Trainable Variables (Parameters) of CGMD ---
a = tf.Variable(tf.ones([NUM_PEAKS]),
                dtype=tf.float32,
                name='a')
weights = tf.nn.softmax(a)

mu = tf.Variable(tf.zeros([NUM_PEAKS, DIM]),
                 dtype=tf.float32,
                 name='mu')

zeta = tf.Variable(softplus_inverse(
                       tf.ones([NUM_PEAKS, DIM])),
                   dtype=tf.float32,
                   name='zeta')
sigma = tf.nn.softplus(zeta)

# --- Construct CGMD as a `tf.distributions.Mixture` Instance ---
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
    return (-0.5 * tf.reduce_sum(tf.square(thetae-100), axis=1)
            -0.5 * tf.log(2 * np.pi) * DIM)
mc_integrand = cgmd.log_prob(thetae) - log_p(thetae)
print('hahaha', mc_integrand.shape)
elbo = tf.reduce_mean(mc_integrand)

grads_0 = tf.gradients(elbo, [a, mu, zeta])
# TODO: read the documentation of `tf.gradients()`.
print('hahaha', [_.shape for _ in grads_0])
#grads = [grads_0[i] + elbo for i in range(3)]

grad_mci = tf.gradients([mc_integrand[i] for i in range(mc_integrand.shape[0])],
                        [a, mu, zeta])


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


# --- WITHOUT Modification of Gradients ---

# Un-comment this block if test WITHOUT modification of gradients

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
optimize = optimizer.minimize(elbo)

# -- Test
with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    
    gvs = optimizer.compute_gradients(elbo)
    print(gvs)
    print('\n\n')
    
    grads_by_optimizer = [
        (g.eval().tolist(), v)
        for g, v in gvs if g is not None]
    grads_by_autodiff = [_.eval() for _ in grads_0]
    
    for i in range(3):
        print(np.mean([grads_by_optimizer[i][0][0] for _ in range(1000)], axis=0))
        print(np.mean([grads_by_autodiff[i][0] for _ in range(1000)], axis=0))
        print()
    # -- Conclusion:
    #    Not the same, even after averaging over many samples of thetae.
    
    
#    with Timer():
#        
#        for i in range(20000):
#       
#           elbo_val, _ = sess.run([elbo, optimize])
#
#           if i % 100 == 0:
#               print('step: {0}'.format(i))
#               print('elbo: {0}'.format(elbo_val))
#               print('theta sample: {0}'.format(sess.run(thetae)[0]))
#               print('-----------------------\n')
#               
#    a_val, mu_val, zeta_val = sess.run([a, mu, zeta])
#    print('a: {0}'.format(a_val))
#    print('mu: {0}'.format(mu_val))
#    print('zeta: {0}'.format(zeta_val))



## --- WITH Modification of Gradients ---
#
## Un-comment this block if test WITH modification of gradients
#
#optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
#direct_grads_to_vars = optimizer.compute_gradients(elbo)
#real_grads_to_vars = [
#    (grad + elbo, var)
#    if grad is not None else (None, var)
#    for grad, var in direct_grads_to_vars]
#optimize = optimizer.apply_gradients(real_grads_to_vars)
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
#               print('theta sample: {0}'.format(sess.run(thetae)[0]))
#               print('-----------------------\n')
#
#    a_val, mu_val, zeta_val = sess.run([a, mu, zeta])
#    print('a: {0}'.format(a_val))
#    print('mu: {0}'.format(mu_val))
#    print('zeta: {0}'.format(zeta_val))
               
               
''' Conclusion:
    
    Without the modification of gradients in `optimizer` for ELBO, the
    optimization works surprsingly great (implied by ELBO as well as by the
    sample of `theta`). However, also surprisingly, when with the modification,
    the optimization blow up quickly.
    
    Can it be possible that TensorFlow has handed the modification for ELBO???
    
    TODO: To answer this, we need a double-check by numpy (i.e. `nn4post_np`).
    
    TODO: Try pure gradient descent optimizer and see what happened in the
          optimization process by TensorBoard ("graph" section).
'''




''' Checking with Hand-Calculated Gradients in the Sample Instance:
    
Use the codes in `'nn4post_tf.py'`:
    
    # Add to `with graph.as_default():`
    with tf.name_scope('momenta'):
        momentum_1 = tf.reduce_mean(thetae,
                                    axis=0,
                                    name='momentum_1')
        momentum_2 = tf.reduce_mean(tf.pow(thetae, 2),
                                    axis=0,
                                    name='momentum_2')
        momentum_3 = tf.reduce_mean(tf.pow(thetae, 3),
                                    axis=0,
                                    name='momentum_3')
    
    # Add to `with tf.Session(graph=graph) as sess:`

    grads_by_optimizer, m1, m2, m3 = sess.run([
        [g for g, v in gvs if g is not None],
        momentum_1,
        momentum_2,
        momentum_3])
    m0 = 1
    momenta = [m0, m1[0], m2[0], m3[0]]
    
    for i in range(3):
        print('grad', i, grads_by_optimizer[i])
    print()
    for i in range(4):
        print('momentum', i, momenta[i])

        
    def cal_grad_by_mu(momenta):
        return -100 * momenta[2] + 5001 * momenta[1]
    def cal_grad_by_zeta(momenta):
        return (np.e-1)/np.e * (-100*momenta[3] + 5001*momenta[2]
                                + 100*momenta[1] - 5001*momenta[0])
    
    print(cal_grad_by_mu(momenta))
    print(cal_grad_by_zeta(momenta))
    
Conclusion:
    
    Even though the momenta are not consistent with the result of gradients
    computed by TensorFlow, but the EFFECT is that the result of gradients
    from TensorFlow is surprisingly more accurate to the theoritical
    calculation result of gradients. I do not know why and where TensorFlow
    makes better than bared computation by my derived formulae.
'''
