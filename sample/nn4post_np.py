#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Numpy Version.
"""

import numpy as np


_DEFAULT_NUM_SAMPLES = 10 ** 2
_DEFAULT_EPSILON = np.exp(-10)
#_DEFAULT_CLIP_LIMIT = np.exp(7)
_DEFAULT_CLIP_LIMIT = np.inf  # test!


def no_underflow(epsilon, x):
    """ If x underflow, then return `epsilon`, as protection in `1/x`.
        Args:
            x: float
        Returns:
            float
    """
    return np.where(x==0, epsilon, x)


class CGMD(object):
    """ Categorical Gaussian Mixture Distribution.
    
    C.f. WikiPedia_.
    
    .. _WikiPedia: https://en.wikipedia.org/wiki/Mixture_distribution#Finite_\
                   and_countable_mixtures
    
    Notations:
        cat:
            (Parameters of) categorical distribution, i.e. the probability for
            each category.
        components:
            (List of parameters of) Gaussian distributions as components, i.e.
            list of the :math:`\mu` and :math:`\sigma` of each multivariate
            Gaussian distribution.
        
        For other notations employed herein, c.f. `'../docs/nn4post.tm'`.
        
    NOTE:
        `a` is initialized by `np.ones`; `b` and `w` are both initialized
        randomly by standard Gaussian.
                   
    Args:
        dim: int
        num_peaks: int
        epsilon: float
            Employed in `no_underflow()`.
        
    Methods:
        log_pdf, sample, 
        get_a, set_a,
        get_b, set_b,
        get_w, set_w,
        get_cat, get_components
    """
    
    def __init__(self, dim, num_peaks, epsilon=_DEFAULT_EPSILON):
        
        self._dim = dim
        self._num_peaks = num_peaks
        self._epsilon = epsilon
        
        self._a = np.ones(shape=[num_peaks])
        self._b = np.random.normal(size=[dim, num_peaks])
        self._w = np.random.normal(size=[dim, num_peaks])
        
        # --- `prob` in the finite categorical distribution ---
        a = self.get_a()
        a_square = np.square(a)
        self._cat = a_square / np.sum(a_square)

        # --- :math:`\mu` and :math:`\sigma` for each Gaussian component ---
        # C.f. the docstring of `self.get_components()`
        b = self.get_b()
        w = self.get_w()
        w = no_underflow(self.get_epsilon(), w)
        self._components = [
            np.array([-b[:,i] / w[:,i], 1 / np.abs(w[:,i])])
            for i in range(num_peaks)]
            
    
    def beta(self, theta):
        """ The :math:`\beta` in `'../docs/nn4post.tm'`.
        
        Args:
            theta: np.array(shape=[None, self.get_dim()], dtype=float32)
            
        Returns:
            (
                beta_max:
                    np.array(shape=[None], dtype=float32)
                delta_beta:
                    np.array(shape=[None, self.get_num_peaks()],
                             dtype=float32)
            )
        """
        
        cat = self.get_cat()
        w = self.get_w()
        b = self.get_b()
        
        # shape: [1, self._num_peaks]
        log_cat = np.expand_dims(
            np.log(cat),
            axis=0)
        # shape: [None, self._num_peaks]
        log_gaussian = np.sum(
            (   -0.5 * np.square(np.expand_dims(theta, axis=2) * w + b)
                +0.5 * np.log(np.square(w) / (2 * np.pi))
            ),
            axis=1)
        
        # shape: [None, self._num_peaks]
        beta0 = log_cat + log_gaussian
        
        # shape: [None]
        beta_max = np.max(beta0, axis=1)
        # shape: [None, self._num_peaks]
        delta_beta = beta0 - np.expand_dims(beta_max, axis=1)
    
        return (beta_max, delta_beta)
        

    def log_pdf(self, theta):
        """ :math:`\ln q (\theta; a, b, w)`, where `theta` is argument and
        `(a, b, w)` is parameter.
        
        Args:
            theta: np.array(shape=[None, self.get_dim()],
                            dtype=float32)
            
        Returns:
            np.array(shape=[None], dtype=float32)
        """
        
        beta_max, delta_beta = self.beta(theta)
        
        return beta_max + np.log(np.sum(np.exp(delta_beta), axis=1))  # shape: [None]
   

    def sample(self, num_samples):
        """ Randomly sample `num_samples` samples from CGMD.
        
        Args:
            num_samples: int
            
        Returns:
            np.array(shape=[num_samples, self.get_dim()],
                     dtype=float32)
        """
        
        num_peaks = self.get_num_peaks()
        dim = self.get_dim()
        cat = self.get_cat()
        components = self.get_components()
        
        def generate_sample():
            """ Generate one sample from CGMD. """
            
            index = np.random.choice(num_peaks, p=cat)
            samp = np.random.normal(*components[index])
            
            return samp
        
        return np.array([generate_sample() for _ in range(num_samples)])
    
    
    # --- Get-Functions ---
    
    def get_dim(self):
        return self._dim
    
    def get_num_peaks(self):
        return self._num_peaks
    
    def get_a(self):
        return self._a
    
    def get_b(self):
        return self._b
    
    def get_w(self):
        return self._w
    
    def get_cat(self):
        """
        Returns:
            np.array(shape=[self._get_num_peaks()], dtype=float32)
        """
        return self._cat
    
    def get_components(self):
        """
        Returns:
            np.array(shape=[self.get_num_peaks(), 2, self.get_dim()],
                     dtype=float32)
            wherein the first in the `2` is the value of "mu", and the second
            of "sigma".
        """
        return self._components
    
    def get_epsilon(self):
        return self._epsilon
    
    
    # --- Set-Functions ---
    
    def _update_cat(self):
        """ Helper of `self.set_a()`. """ 
        a = self.get_a()
        a_square = np.square(a)
        self._cat = a_square / np.sum(a_square)
        return None
    
    def _update_components(self):
        """ Helper of `self.set_b()` and `self.set_w().
        """
        num_peaks = self.get_num_peaks()
        b = self.get_b()
        w = self.get_w()
        w = no_underflow(self.get_epsilon(), w)
        self._components = [
            np.array([-b[:,i] / w[:,i], 1 / np.abs(w[:,i])])
            for i in range(num_peaks)]
        return None
    
    def set_a(self, value):
        self._a = value
        self._update_cat()
        return None
    
    def set_b(self, value):
        self._b = value
        self._update_components()
        return None
    
    def set_w(self, value):
        self._w = value
        self._update_components()
        return None
    
    def copy(self):
        cgmd = CGMD(self.get_dim(),
                    self.get_num_peaks(),
                    self.get_epsilon())
        cgmd.set_a(self.get_a())
        cgmd.set_b(self.get_b())
        cgmd.set_w(self.get_w())
        return fgmc


def performance(log_p, cgmd, num_samples=_DEFAULT_NUM_SAMPLES):
    """ Use KL-divergence as performance of fitting a posterior `log_p` by a
        finite Gaussian mixture distribution (CGMD).
        
    NOTE:
        Even though KL-divergence is principly non-negative, the `log_p` and
        `cgmd.log_pdf` may not be normalized, so that the KL-divergence is
        "non-negative up to an overall constant".
    
    Args:
        log_p: Map(**{thetae: np.array(shape=[None, cgmd.get_dim()],
                                       dtype=float32)},
                   float)
        cgmd: CGMD
        num_sample: int
            For Monte Carlo integration.
            
    Returns:
        float
    """
    
    thetae = cgmd.sample(num_samples)

    kl_divergence = np.mean(cgmd.log_pdf(thetae) - log_p(thetae))
    return kl_divergence


def nabla_perfm(log_p, cgmd, epsilon, clip_limit, num_samples):
    """ :math:`\nabla_z \textrm(KL) (q(z) || p_D)`, where :math:`z := (a, b, w)`,
        :math:`q` is the PDF of `cgmd` (whose logrithm is `cgmd.log_pdf`), and
        `log_p` is the logrithm of PDF of :math:`p`.
        
        Helper function for `gradient_descent()`.
        
    Numerical Treatments:
        Sometimes the gradients shall be clipped, since there's `1/a`, `1/w`
        in the gradients, which may lead to an overflow (thus returns `nan`)
        when `a` or `w` becomes small enough.
        
        And the epsilon is added to the denominator of `1/a` andb '1/w` in the
        computation of gradients to avoid `nan`.
        
    Args:
        log_p: Map(**{thetae: np.array(shape=[None, cgmd.get_dim()],
                                       dtype=float32)},
                   float)
            Vectorized, as `None` hints.
        cgmd: CGMD
        epsilon: float
            Positive. Added to the denominator of `1/a` andb '1/w` in the
            computation of gradients to avoid `nan`.
        clip_limit: float or None
            Clipping each element in the gradients of `a`, `b`, and `w` by
            `clip_limit`. If `None`, then no clipping to be established.
        num_samples: int
            Number of samples in the Monte Carlo integration herein.
            
    Returns:
        [
            nabla_perfm_by_a: np.array(shape=cgmd.get_a().shape,
                                       dtype=float32),
            nabla_perfm_by_b: np.array(shape=cgmd.get_b().shape,
                                       dtype=float32),
            nabla_perfm_by_w: np.array(shape=cgmd.get_w().shape,
                                       dtype=float32),
        ]
    """

    log_q = cgmd.log_pdf
    a = cgmd.get_a()
    b = cgmd.get_b()
    w = cgmd.get_w()

    thetae = cgmd.sample(num_samples)

    beta_max, delta_beta = cgmd.beta(thetae)
    # shape: [num_samples, num_peaks]
    proportion = np.exp(delta_beta) \
               / np.expand_dims(np.sum(np.exp(delta_beta), axis=1), axis=1)
             
    def _nabla_perfm_sub(nabla_beta):
        """ Helper of `nabla_perfm()`.
        
        Args:
            nabla_beta_i: np.array(shape=?, dtype=float32)
                :math:`\frac{\partial \beta}{\partial z}`
        Returns:
            np.array(shape=?, dtype=float32)
        """
        
        # shape: [num_samples, num_peaks]
        x = np.expand_dims(log_q(thetae) - log_p(thetae) - 1, axis=1) \
            * proportion
            
        if len(nabla_beta.shape) == 1:  # like `a`, with shape: [num_peaks].
            
            return np.mean(x * nabla_beta, axis=0)
        
        else:  # like `b` and `w`, with shape: [num_samples, dim, num_peaks].
        
            assert len(nabla_beta.shape) == 3
            
            return np.mean(
                np.expand_dims(x, axis=1) * nabla_beta,
                axis=0)
    
    def _no_underflow(x):
        return no_underflow(epsilon, x)
    
    # :math:`\frac{\partial beta_i}{\partial a_i} = \frac{2}{a_i}`
    # shape: [num_samples, num_peaks]
    nabla_beta_by_a = 2 / (_no_underflow(a))
    # :math:`\frac{\partial beta_i}{\partial b_{ji}} = -(\theta_j w_{ji} + b_{ji})`
    # shape: [num_samples, dim, num_peaks]
    nabla_beta_by_b = (- np.expand_dims(thetae, axis=2) * np.expand_dims(w, axis=0)
                       + np.expand_dims(b, axis=0))
    # :math:`\frac{\partial beta_i}{\partial b_{ji}} = -(\theta_j w_{ji} + b_{ji}) \theta_j + \frac{1}{w_{ji}}`
    # shape: [num_samples, dim, num_peaks]
    nabla_beta_by_w = (- np.expand_dims(thetae, axis=2) * np.expand_dims(w, axis=0)
                       + np.expand_dims(b, axis=0)) * np.expand_dims(thetae, axis=2) \
                      + 1 / np.expand_dims(_no_underflow(w), axis=0)

    gradients = [
        _nabla_perfm_sub(nabla_beta_by_a),  # shape: [num_peaks]
        _nabla_perfm_sub(nabla_beta_by_b),  # shape: [dim, num_peaks]
        _nabla_perfm_sub(nabla_beta_by_w),  # shape: [dim, num_peaks]
        ]

    # Clip the gradients
    for i, grad in enumerate(gradients):
        gradients[i] = np.clip(grad, -clip_limit, clip_limit)
        
    return gradients


def gradient_descent(
        log_p,
        cgmd,
        learning_rate,
        epsilon=_DEFAULT_EPSILON,
        clip_limit=_DEFAULT_CLIP_LIMIT,
        num_samples=_DEFAULT_NUM_SAMPLES):
    """ Update `cgmd` **once** by using gradient descent method which
        **minimizes** the KL-divergence (as the performance) between `log_p`
        and `cgmd.log_pdf`.
        
    Numerical Treatments:
        Sometimes the gradients shall be clipped, since there's `1/a`, `1/w`
        in the gradients, which may lead to an overflow (thus returns `nan`)
        when `a` or `w` becomes small enough.
        
        And the epsilon is added (BUGS HEREIN) to the denominator of `1/a` and
        '1/w` in the computation of gradients to avoid `nan`.
        
    Args:
        log_p: Map(**{thetae: np.array(shape=[None, cgmd.get_dim()],
                                       dtype=float32)},
                   float)
            Vectorized, as `None` hints.
        cgmd: CGMD
        learning_rate: float
            `z += -learning_rate * gradient_by_z for z in [a, b, w]`.
        num_samples: int
            Number of samples in the Monte Carlo integration herein.
        clip_limit: float or None
            Clipping each element in the gradients of `a`, `b`, and `w` by
            `clip_limit`. If `None`, then no clipping to be established.
    
    Returns:
        CGMD
            The updated `cgmd`.
    """

    gradients = nabla_perfm(log_p,
                            cgmd,
                            num_samples=num_samples,
                            epsilon=epsilon,
                            clip_limit=clip_limit)
    
    delta_a = -learning_rate * gradients[0]
    delta_b = -learning_rate * gradients[1]
    delta_w = -learning_rate * gradients[2]
    
    a = cgmd.get_a()
    b = cgmd.get_b()
    w = cgmd.get_w()
    
    cgmd.set_a(a + delta_a)
    cgmd.set_b(b + delta_b)
    cgmd.set_w(w + delta_w)

    return cgmd


def sgd(log_posterior,
        data_batchs,
        cgmd,
        learning_rate,
        epsilon=_DEFAULT_EPSILON,
        clip_limit=_DEFAULT_CLIP_LIMIT,
        num_samples=_DEFAULT_NUM_SAMPLES):
    """ Update `cgmd` by using stochastic gradient descent (SGD) method which
        **minimizes** the KL-divergence (as the performance) between `log_p`
        and `cgmd.log_pdf`.
        
    Numerical Treatments:
        Sometimes the gradients shall be clipped, since there's `1/a`, `1/w`
        in the gradients, which may lead to an overflow (thus returns `nan`)
        when `a` or `w` becomes small enough.
        
        And the epsilon is added (BUGS HEREIN) to the denominator of `1/a` and
        '1/w` in the computation of gradients to avoid `nan`.
        
    Args:
        log_posterior:
            Map(**{data: (x: np.array(shape=None, dtype=None),
                          y: np.array(shape=None, dtype=None))
                   theta: np.array(shape=[cgmd.get_dim()], dtype=float32)},
                float)
            where `np.array(shape=None, dtype=None)` is an element in
            `data_batchs`.
        data_batchs: [np.array(shape=None, dtype=None)]
        cgmd: CGMD
        learning_rate: float
            `z += -learning_rate * gradient_by_z for z in [a, b, w]`.
        num_samples: int
            Number of samples in the Monte Carlo integration herein.
        clip_limit: float or None
            Clipping each element in the gradients of `a`, `b`, and `w` by
            `clip_limit`. If `None`, then no clipping to be established.
    
    Returns:
        CGMD
            The updated `cgmd`.
    """
    
    for data in data_batchs:
        
        # Recall that `gradient_descent()` requires its argument `log_p` being
        # vectorized
        def log_p(thetae):
            num_thetae = thetae.shape[0]
            return np.array([log_posterior(data, thetae[i,:])
                             for i in range(num_thetae)])
        
        gradient_descent(
            log_p,
            cgmd,
            learning_rate,
            epsilon=epsilon,
            clip_limit=clip_limit,
            num_samples=num_samples)

    return cgmd


if __name__ == '__main__':
    """ Tests. """
    
    # --- The First Test ---
    
    import tools

    DIM = 1
    NUM_PEAKS = 10
    
    cgmd = CGMD(DIM, NUM_PEAKS)
    log_p = lambda theta: (-0.5 * np.sum(np.square(theta), axis=1)
                           - 0.5 * np.log(2 * np.pi))
    
    # --- Before gradient descent
    #print('b: {0}'.format(cgmd.get_b()))
    #print('w: {0}'.format(cgmd.get_w()))
    #print('cat: {0}'.format(cgmd.get_cat()))
    #print('components: {0}'.format(cgmd.get_components()))
    old_performance = performance(log_p, cgmd)
    print('performance: {0}'.format(old_performance))
    
    # --- Making gradient descent
    with tools.Timer():
        for step in range(1000):
            gradient_descent(log_p, cgmd, learning_rate=0.0001)
        
    # After gradient descent
    print('After updating ......\n')
    #print('updated a: {0}'.format(cgmd.get_a()))
    #print('updated b (shall be near `0`): {0}'.format(cgmd.get_b()))
    #print('updated w (shall be near `+-1`): {0}'.format(cgmd.get_w()))
    #print('updated cat: {0}'.format(cgmd.get_cat()))
    #print('updated components: {0}'.format(cgmd.get_components()))
    
    # --- Improvement by gradient descent
    new_performance = performance(log_p, cgmd)
    print('\nupdated performance:\n\t{0}  -->  {1}\n'.format(
            old_performance, new_performance))
    
    # --- Plot the result out
    #     **Valid only when `DIM = 1`**
    assert DIM == 1
    import matplotlib.pyplot as plt
    boundary = 2
    num_x = boundary * 10
    x = np.linspace(-boundary, boundary, num_x)
    plt.plot(x, log_p(np.array([[_] for _ in x])))
    plt.plot(x, cgmd.log_pdf(np.array([[_] for _ in x])), '--')    
    plt.show()

#    # --- The Second Test ---
#
#    import tools
#    import matplotlib.pyplot as plt
#
#    DIM = 1
#    NUM_PEAKS = 100
#    
#    cgmd = CGMD(DIM, NUM_PEAKS)
#    
#    NUM_DATA = 1000
#    THETA_STAR = 1
#    x = np.linspace(-1, 1, NUM_DATA)
#    mu = - np.sqrt(np.sum(np.square(x))) * THETA_STAR
#    one_by_sigma = np.sqrt(np.sum(np.square(x)))
#    print(mu, one_by_sigma)
#
#    def log_p(theta):
#        noise = np.random.normal()
#        return -0.5 * (np.sum(np.square(one_by_sigma * (theta - mu)), axis=1)
#                       + np.log(2 * np.pi)
#                       - np.log(np.square(one_by_sigma))) + noise
#    
#    # --- Before gradient descent
#    old_performance = performance(log_p, cgmd)
#    print('performance: {0}'.format(old_performance))
#    
#    # --- Making gradient descent
#    with tools.Timer():
#        
#        epochs = 10 ** 3
#        performance_log = []
#        performance_log.append(np.log(old_performance))
#        
#        for epoch in range(epochs):
#            gradient_descent(log_p, cgmd, learning_rate=0.001, num_samples=10**2)
#            new_performance = performance(log_p, cgmd)
#            performance_log.append(np.log(new_performance))
#            
#    new_performance = performance(log_p, cgmd)
#    print('\nupdated performance:\n\t{0}  -->  {1}\n'.format(
#            old_performance, new_performance))
#    
#    plt.plot(performance_log)
#    plt.show()
