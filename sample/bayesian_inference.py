#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A Bayesian inference illustrated in '../docs/nn4post.tm', section 1.
"""


import numpy as np
import matplotlib.pyplot as plt


def log_post(data, theta, f, prior_is_unknown=True):
    """ Logrithmic posterior, up to :math:`ln p(D)`.
    
    Notations:
        C.f. '../docs/nn4post.tm', section 1.
    
    Args:
        data: (x: np.array(shape=[num_data, x_dim]),
               y: np.array(shape=[num_data, y_dim]))
            with length of list `num_data`.
        theta: np.array(shape=[theta_dim], dtype=float32)
        f: Map((theta: np.array(shape=[theta_dim], dtype=float32),
                x: np.array(shape=[x_dim])),
               np.array(shape=[y_dim]))
    
    Returns:
        float
    """

    x, y = data  # shape: [num_data, x_dim] and [num_data, y_dim]
    num_data, x_dim = x.shape
    num_data, y_dim = y.shape
    theta_dim = theta.shape

    y_sigma=np.ones(shape=[num_data, y_dim]) * 1
    theta_sigma = np.ones(shape=theta_dim) * 1000
    
    f_array = np.array([f(theta, x[i,:]) for i in range(num_data)])
    
    log_likelihood = -0.5 * (
        np.sum(np.square((y - f_array) / y_sigma))
        + np.sum(np.log(np.square(y_sigma)))
        + np.log(2 * np.pi) * num_data
        )
    log_prior = -0.5 * (
        np.sum(np.log(np.square(theta / theta_sigma)))
        + np.log(2 * np.pi) + np.sum(np.log(np.square(theta_sigma)))
        )
    
    if prior_is_unknown:
        return log_likelihood
    else:
        return log_likelihood + log_prior


if __name__ == '__main__':
    """ Test the code by linear regression instance. `y_dim = 1`. """
    
    # --- Test `log_post()` ---
    x_dim = 1
    y_dim = 1
    theta_dim = x_dim

    num_data = 100
    
    def f(theta, x):
        return np.array([np.sum(theta * x)])
    
    x = np.array([np.linspace(-1, 1, num_data) for i in range(x_dim)])
    x = x.T  # shape: [num_data, x_dim]
    target_theta = np.ones(shape=[x_dim])
    @np.vectorize
    def target_f(x):
        return np.sum(target_theta * x)
    y = target_f(x) \
      + np.random.normal(loc=0.0, scale=0.0, size=[num_data, y_dim])
    data = (x, y)
    print(log_post(data, target_theta, f))

    
    
    # --- Test nn4post ---
    
    from nn4post_np import FGMD, performance, gradient_descent
    
    DIM = theta_dim
    NUM_PEAKS = 1
    
    fgmd = FGMD(DIM, NUM_PEAKS)
    def log_p(theta):
        return log_post(data, theta, f)
    
    # --- Before gradient descent
    old_performance = performance(log_p, fgmd)
    print('performance: {0}'.format(old_performance))
    
    # --- Making gradient descent
    import tools
    with tools.Timer():
        
        num_epochs = 1000
        performance_log = []
        performance_log.append(np.log(old_performance))
        
        for epoch in range(num_epochs):
            
            gradient_descent(log_p, fgmd, num_steps=10, learning_rate=0.0001)
            new_performance = performance(log_p, fgmd)
            performance_log.append(np.log(new_performance))
    
    # --- Improvement by gradient descent
    new_performance = performance(log_p, fgmd)
    print('\nupdated performance:\n\t{0}  -->  {1}\n'.format(
            old_performance, new_performance))
    
    plt.plot(performance_log)
    plt.show() 
    
    # --- `theta_hat` ---
    
    bi_num_samples = 100
    thetae = fgmd.sample(bi_num_samples)
    
    x = np.random.uniform(size=[x_dim])
    y_target = target_f(x)
    y_hat = np.mean([f(thetae[i,:], x) for i in range(bi_num_samples)], axis=0)
    print(y_target, y_hat)
