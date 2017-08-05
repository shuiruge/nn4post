#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A Bayesian inference illustrated in '../docs/nn4post.tm', section 1.
"""


import numpy as np


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
    
    # shape: [num_data, x_dim] and [num_data, y_dim]
    x, y = data
    num_data, x_dim = x.shape
    num_data, y_dim = y.shape

    y_sigma=np.ones(shape=y.shape) * 1
    theta_sigma = np.ones(shape=theta.shape) * 1
    
    # shape: [num_data, y_dim]
    f_array = np.array([f(theta, x[i,:]) for i in range(num_data)])
    
    log_likelihood = -0.5 * (
          np.sum(np.square((y - f_array) / y_sigma))
        + np.sum(np.log(2 * np.pi * np.square(y_sigma)))
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
    
    from nn4post_np import CGMD, performance, sgd, vectorize
    import tools
    import matplotlib.pyplot as plt

    
    # --- Test `log_post()` ---
    x_dim = 1
    y_dim = 1
    theta_dim = x_dim

    num_data = 1000
    num_batchs = 100
    assert num_data % num_batchs == 0

    def f(theta, x):
        return np.dot(x, theta)

    x = np.array([np.linspace(-1, 1, num_data) for i in range(x_dim)])
    x = x.T  # shape: [num_data, x_dim]
    
    target_theta = np.ones(shape=[x_dim, y_dim])
    target_theta *= 10
    def target_f(x):
        return f(target_theta, x)
    
    np.random.shuffle(x)
    x_batchs = np.split(x, num_batchs)
    y_batchs = [target_f(x) for x in x_batchs]
    data_batchs = list(zip(x_batchs, y_batchs))
    for data in data_batchs:
        x, y = data
        plt.plot(x, y)
    plt.show()
    
    print([log_posterior(data_batchs[i], target_theta, f)
           for i in range(num_batchs)])
    
    # --- Plot `log_posterior` out
    #     **Valid only when `DIM = 1`**
    thetae = [np.array([[theta]]) for theta in np.linspace(-20, 20, 20)]
    plt.plot(
        [theta[0,0] for theta in thetae],
        [log_posterior(data_batchs[0], theta, f) for theta in thetae])
    plt.show()
    


    
    # --- Test nn4post ---
    
    
    DIM = theta_dim
    NUM_PEAKS = 1
    
    cgmd = CGMD(DIM, NUM_PEAKS)
    def log_p(data, theta):
        return log_posterior(data, theta, f)
    
    # --- Plot `log_p` out
    #     **Valid only when `DIM = 1`**
    thetae = [np.array([[theta]]) for theta in np.linspace(-20, 20, 20)]
    plt.plot(
        [theta[0,0] for theta in thetae],
        [log_p(data_batchs[0], theta) for theta in thetae])
    plt.show()
    
    
    # --- Before gradient descent
    data = data_batchs[0]
    old_performance = performance(lambda theta: log_p(data, theta), cgmd)
    print('performance: {0}'.format(old_performance))
    
    # --- Making gradient descent
    performance_log = [old_performance]
    a_log = [cgmd.get_a()]
    b_log = [cgmd.get_b()]
    w_log = [cgmd.get_w()]

    with tools.Timer():
        
        num_epochs = 5

        for epoch in range(num_epochs):
            
            sgd(log_p, data_batchs, cgmd, learning_rate=0.001, num_samples=10**2)
            new_performance = performance(lambda theta: log_p(data, theta), cgmd)
            
            performance_log.append(np.log(new_performance))
            a_log.append(cgmd.get_a())
            b_log.append(cgmd.get_b())
            w_log.append(cgmd.get_w())
    
    # --- Improvement by gradient descent
    new_performance = performance(lambda theta: log_p(data, theta), cgmd)
    print('\nupdated performance:\n\t{0}  -->  {1}\n'.format(
            old_performance, new_performance))
    
    plt.plot(performance_log)
    plt.show()
    
    # --- Plot the result out
    #     **Valid only when `DIM = 1`**
    assert DIM == 1
    boundary = 20
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
