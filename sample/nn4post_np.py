#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Numpy Version.
"""

import numpy as np


def no_underflow(epsilon, x):
    """ If x underflow, then return `epsilon`, as protection in `1/x`.
        Args:
            x: float
        Returns:
            float
    """
    return np.where(x==0, epsilon, x)


class FGMD(object):
    """ Finite Gaussian Mixture Distribution.
    
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
    
    def __init__(self, dim, num_peaks, epsilon=np.exp(-10)):
        
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
        """ Randomly sample `num_samples` samples from FGMD.
        
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
            """ Generate one sample from FGMD. """
            
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
        fgmd = FGMD(self.get_dim(),
                    self.get_num_peaks(),
                    self.get_epsilon())
        fgmd.set_a(self.get_a())
        fgmd.set_b(self.get_b())
        fgmd.set_w(self.get_w())
        return fgmd


def performance(log_p, fgmd, num_samples=100):
    """ Use KL-divergence as performance of fitting a posterior `log_p` by a
        finite Gaussian mixture distribution (FGMD).
        
    NOTE:
        Even though KL-divergence is principly non-negative, the `log_p` and
        `fgmd.log_pdf` may not be normalized, so that the KL-divergence is
        "non-negative up to an overall constant".
    
    Args:
        log_p: np.array(shape=[fgmd.get_dim()], dtype=float32) -> float
        fgmd: FGMD
        num_sample: int
            For Monte Carlo integration.
            
    Returns:
        float
    """
    
    thetae = fgmd.sample(num_samples)

    kl_divergence = np.mean(fgmd.log_pdf(thetae) - log_p(thetae))
    return kl_divergence


def nabla_perfm(fgmd, log_p, epsilon, clip_limit, num_samples):
    """ :math:`\nabla_z \textrm(KL) (q(z) || p)`, where :math:`z := (a, b, w)`,
        :math:`q` is the PDF of `fgmd` (whose logrithm is `fgmd.log_pdf`), and
        `log_p` is the logrithm of PDF of :math:`p`.
        
        Helper function for `gradient_descent()`.
        
    Numerical Treatments:
        Sometimes the gradients shall be clipped, since there's `1/a`, `1/w`
        in the gradients, which may lead to an overflow (thus returns `nan`)
        when `a` or `w` becomes small enough.
        
        And the epsilon is added (BUGS HEREIN) to the denominator of `1/a` and
        '1/w` in the computation of gradients to avoid `nan`.
        
    Args:
        fgmd: FGMD
        log_p: np.array(shape=[fgmd.get_dim()], dtype=float32) -> float
        epsilon: float
            asdf   BUGS HEREIN
        clip_limit: float or None
            Clipping each element in the gradients of `a`, `b`, and `w` by
            `clip_limit`. If `None`, then no clipping to be established.
        num_samples: int
            Number of samples in the Monte Carlo integration herein.
            
    Returns:
        [
            nabla_perfm_by_a: np.array(shape=fgmd.get_a().shape,
                                       dtype=float32),
            nabla_perfm_by_b: np.array(shape=fgmd.get_b().shape,
                                       dtype=float32),
            nabla_perfm_by_w: np.array(shape=fgmd.get_w().shape,
                                       dtype=float32),
        ]
    """

    log_q = fgmd.log_pdf
    a = fgmd.get_a()
    b = fgmd.get_b()
    w = fgmd.get_w()

    thetae = fgmd.sample(num_samples)

    beta_max, delta_beta = fgmd.beta(thetae)
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


def gradient_descent(log_p, fgmd, num_steps, learning_rate,
        epsilon=np.exp(-10), clip_limit=np.exp(7), num_samples=100):
    """ Update `fgmd` by using gradient descent method which **minimizes** the
        KL-divergence (as the performance) between `log_p` and `fgmd.log_pdf`.
        
    Numerical Treatments:
        Sometimes the gradients shall be clipped, since there's `1/a`, `1/w`
        in the gradients, which may lead to an overflow (thus returns `nan`)
        when `a` or `w` becomes small enough.
        
        And the epsilon is added (BUGS HEREIN) to the denominator of `1/a` and
        '1/w` in the computation of gradients to avoid `nan`.
        
    Args:
        fgmd: FGMD
        log_p: np.array(shape=[fgmd.get_dim()], dtype=float32) -> float
        num_steps: int
            Number of times of updating by gradient descent.
        learning_rate: float
            `z += -learning_rate * gradient_by_z for z in [a, b, w]`.
        num_samples: int
            Number of samples in the Monte Carlo integration herein.
        clip_limit: float or None
            Clipping each element in the gradients of `a`, `b`, and `w` by
            `clip_limit`. If `None`, then no clipping to be established.
    
    Returns:
        FGMD
            The updated `fgmd`.
    """
        
    for step in range(num_steps):
        
        try:
            gradients = nabla_perfm(fgmd, log_p,
                                    num_samples=num_samples,
                                    epsilon=epsilon,
                                    clip_limit=clip_limit)
            
            delta_a = -learning_rate * gradients[0]
            delta_b = -learning_rate * gradients[1]
            delta_w = -learning_rate * gradients[2]
            
            a = fgmd.get_a()
            b = fgmd.get_b()
            w = fgmd.get_w()
            
            fgmd.set_a(a + delta_a)
            fgmd.set_b(b + delta_b)
            fgmd.set_w(w + delta_w)
        
        except Exception as e:
            
            print('ERROR')
            print('step at {0}'.format(step))
            raise Exception(e)
           
    return fgmd


if __name__ == '__main__':
    """ Tests. """
    
    # --- The First Test ---
    
    import tools

    DIM = 100
    NUM_PEAKS = 2
    
    fgmd = FGMD(DIM, NUM_PEAKS)
    log_p = lambda theta: (-0.5 * np.sum(np.square(theta), axis=1)
                           - 0.5 * np.log(2 * np.pi))
    
    # --- Before gradient descent
    #print('b: {0}'.format(fgmd.get_b()))
    #print('w: {0}'.format(fgmd.get_w()))
    #print('cat: {0}'.format(fgmd.get_cat()))
    #print('components: {0}'.format(fgmd.get_components()))
    old_performance = performance(log_p, fgmd)
    print('performance: {0}'.format(old_performance))
    
    # --- Making gradient descent
    with tools.Timer():
        gradient_descent(log_p, fgmd, num_steps=1000, learning_rate=0.001)
        
    # After gradient descent
    print('After updating ......\n')
    #print('updated a: {0}'.format(fgmd.get_a()))
    #print('updated b (shall be near `0`): {0}'.format(fgmd.get_b()))
    #print('updated w (shall be near `+-1`): {0}'.format(fgmd.get_w()))
    #print('updated cat: {0}'.format(fgmd.get_cat()))
    #print('updated components: {0}'.format(fgmd.get_components()))
    
    # --- Improvement by gradient descent
    new_performance = performance(log_p, fgmd)
    print('\nupdated performance:\n\t{0}  -->  {1}\n'.format(
            old_performance, new_performance))


#    # --- The Second Test ---
#
#    import tools
#    import matplotlib.pyplot as plt
#
#    DIM = 1
#    NUM_PEAKS = 1
#    
#    fgmd = FGMD(DIM, NUM_PEAKS)
#    
#    NUM_DATA = 100
#    THETA_STAR = 1
#    x = np.linspace(-1, 1, NUM_DATA)
#    mu = - np.sqrt(np.sum(np.square(x))) * THETA_STAR
#    one_by_sigma = np.sqrt(np.sum(np.square(x)))
#    print(mu, one_by_sigma)
#
#    def log_p(theta):
#        return -0.5 * (np.sum(np.square(one_by_sigma * (theta - mu)), axis=1)
#                       + np.log(2 * np.pi)
#                       - np.log(np.square(one_by_sigma)))
#    
#    # --- Before gradient descent
#    old_performance = performance(log_p, fgmd)
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
#            gradient_descent(log_p, fgmd, num_steps=10, learning_rate=0.001, num_samples=10**3)
#            new_performance = performance(log_p, fgmd)
#            performance_log.append(np.log(new_performance))
#            
#    new_performance = performance(log_p, fgmd)
#    print('\nupdated performance:\n\t{0}  -->  {1}\n'.format(
#            old_performance, new_performance))
#    
#    plt.plot(performance_log)
#    plt.show()
