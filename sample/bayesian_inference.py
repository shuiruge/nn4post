#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A Bayesian inference illustrated in '../docs/nn4post.tm', section 1.
"""


import numpy as np
import matplotlib.pyplot as plt


def log_posterior(data, theta, f, prior_is_unknown=True):
    """ Logrithmic posterior, up to :math:`ln p(D)`.
    
    Notations:
        C.f. '../docs/nn4post.tm', section 1.
    
    Args:
        data: (x: np.array(shape=[num_data, x_dim]),
               y: np.array(shape=[num_data, y_dim]))
            with length of list `num_data`.
        theta: np.array(shape=[theta_dim], dtype=float32)
        f: Map(**{theta: np.array(shape=[theta_dim], dtype=float32),
                  x: np.array(shape=[x_dim])},
               np.array(shape=[y_dim]))
    
    Returns:
        float
    """

    x, y = data  # shape: [num_data, x_dim] and [num_data, y_dim]
    num_data, x_dim = x.shape
    num_data, y_dim = y.shape

    y_sigma=np.ones(shape=[num_data, y_dim]) * 1
    theta_sigma = np.ones(shape=theta.shape) * 1
    
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

    num_data = 1000
    num_batchs = 50
    assert num_data % num_batchs == 0
    
    def f(theta, x):
        return np.array([np.sum(theta * x)])
    
    x = np.array([np.linspace(-1, 1, num_data) for i in range(x_dim)])
    x = x.T  # shape: [num_data, x_dim]
    target_theta = np.ones(shape=[x_dim])
    def target_f(x):
        return np.sum(target_theta * x)
    y = target_f(x) \
      + np.random.normal(loc=0.0, scale=0.0, size=[num_data, y_dim])
    np.random.RandomState(seed=12345).shuffle(x)
    np.random.RandomState(seed=12345).shuffle(y)
    x_batchs = np.split(x, num_batchs)
    y_batchs = np.split(y, num_batchs)
    data_batchs = list(zip(x_batchs, y_batchs))
    print(log_posterior(data_batchs[0], target_theta, f))

    
    
    # --- Test nn4post ---
    
    from nn4post_np import CGMD, performance, sgd
    
    DIM = theta_dim
    NUM_PEAKS = 10
    
    cgmd = CGMD(DIM, NUM_PEAKS)
    def log_p(data, theta):
        return log_posterior(data, theta, f)
    
    # --- Before gradient descent
    data = data_batchs[0]
    old_performance = performance(lambda theta: log_p(data, theta), cgmd)
    print('performance: {0}'.format(old_performance))
    
    # --- Making gradient descent
    import tools
    with tools.Timer():
        
        num_epochs = 30
        performance_log = []
        performance_log.append(np.log(old_performance))
        
        for epoch in range(num_epochs):
            
            sgd(log_p, data_batchs, cgmd, learning_rate=0.001, num_samples=10**2)
            new_performance = performance(lambda theta: log_p(data, theta), cgmd)
            performance_log.append(np.log(new_performance))
    
    # --- Improvement by gradient descent
    new_performance = performance(lambda theta: log_p(data, theta), cgmd)
    print('\nupdated performance:\n\t{0}  -->  {1}\n'.format(
            old_performance, new_performance))
    
    plt.plot(performance_log)
    plt.show()
    
    # --- Plot the result out
    #     **Valid only when `DIM = 1`**
    assert DIM == 1
    boundary = 5
    num_x = boundary * 10
    x = np.linspace(-boundary, boundary, num_x)
    plt.plot(x, np.array([log_p(data, np.array([_])) for _ in x]))
    plt.plot(x, cgmd.log_pdf(np.array([[_] for _ in x])), '--')
    plt.show()

    
#    # --- `theta_hat` ---
#    
#    thetae = cgmd.sample(100)
#    print(np.mean(thetae, axis=0))
#    
#    x = np.random.uniform(size=[x_dim])
#    y_target = target_f(x)
#    y_hat = np.mean([f(thetae[i,:], x) for i in range(100)], axis=0)
#    print(y_target, y_hat)
