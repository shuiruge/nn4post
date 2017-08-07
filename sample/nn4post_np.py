#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Numpy Version.
"""

import numpy as np


_DEFAULT_NUM_SAMPLES = 10 ** 2
_GREAT_NEGATIVE = -10



def vectorize(f):
    """ Vectorize function `f` (as the `None` in the "Returns" hints).
    Args:
        f: Map(x: np.array(shape=[*a]),
               y: np.array(shape=[*b]))
    
    Returns:
        Map(xs: np.array(shape=[None, *a]),
            ys: np.array(shape=[None, *b]))
    """
    
    def vectorized_f(xs):
        num_xs = xs.shape[0]
        return np.array([f(xs[i]) for i in range(num_xs)])
    
    return vectorized_f


def softmax(x):
    """ :math:`\forall i, \frac{\exp(x_i)}{\sum_j \exp(x_j)}`. 
    
    NOTE:
        Numerical trick is taken, for avoiding underflow and overflow (c.f.
        *Deep Learning* (MIT), chapter 4).
    
    Args:
        x: np.array(shape=[None], dtype=np.float32)
    
    Returns:
        np.array(shape=[None], dtype=np.float32)
    """
    x_max = np.max(x)
    delta_x = x - x_max
    
    return np.exp(delta_x) / np.sum(np.exp(delta_x))


def log_softmax(x):
    """ :math:`\forall i, \ln\left(\frac{\exp(x_i)}{\sum_j \exp(x_j)}\right)`. 
    
    NOTE:
        Numerical trick is taken, for avoiding underflow and overflow (c.f.
        *Deep Learning* (MIT), chapter 4).
    
    Args:
        x: np.array(shape=[None], dtype=np.float32)
    
    Returns:
        np.array(shape=[None], dtype=np.float32)
    """
    x_max = np.max(x)
    delta_x = x - x_max
    
    return delta_x - np.log(np.sum(np.exp(delta_x)))



def sigmoid(x):
    """ :math:`\frac{1}{1 + \exp(-x)}`.
    
    Args:
        x: array_like
    
    Returns:
        ndarray
    """
    return 1 / (1 + np.exp(-x))


def softplus(x):
    """ :math:`\ln(1 + \exp(x))`.
    
    Args:
        x: array_like
        
    Returns:
        ndarray
    """
    return np.log(1 + np.exp(x))


def log_softplus(x):
    """ :math:`\ln(1 + \exp(x))`.
    
    NOTE:
        Numerical trick is taken.
        
    TODO:
        Test.
    
    Args:
        x: array_like
        
    Returns:
        ndarray
    """
    return np.where(np.less(x, _GREAT_NEGATIVE * np.ones(x.shape)),
                    x,
                    np.log(softplus(x)))




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
        get_mu, set_mu,
        get_zeta, set_zeta,
        get_cat, get_components
    """
    
    def __init__(self, dim, num_peaks):
        
        self._dim = dim
        self._num_peaks = num_peaks
        
        # --- Initialize a, mu, and zeta ---
        self._a = np.random.normal(size=[self._num_peaks])
        self._mu = np.random.normal(size=[self._num_peaks, self._dim])
        self._zeta = np.random.normal(size=[self._num_peaks, self._dim])
        
        # shape: [self._num_peak]
        self._w = softmax(self._a)
        self._log_w = log_softmax(self._a)
        # shape: [self._num_peak, self._dim]
        self._sigma = softplus(self._zeta)
        self._log_sigma = log_softplus(self._zeta)
        
        self.vectorized_log_pdf = vectorize(self.log_pdf)
        self.vlog_pdf = self.vectorized_log_pdf  # for short.

        
    
    def beta(self, theta):
        """ The :math:`\beta` in `'../docs/nn4post.tm'`.
        
        Args:
            theta: np.array(shape=[self.get_dim()], dtype=np.float32)
            
        Returns:
            np.array(shape=[self.get_num_peaks()], dtype=np.float32)
        """
        
        return (self.get_log_w()
                + self.get_dim() * (-0.5 * np.log(2 * np.pi))
                + np.sum(
                    - self.get_log_sigma()
                    - 0.5 * np.square(theta * self.get_mu() / self.get_sigma()),
                    axis=1))
    
        
    def log_pdf(self, theta):
        """ :math:`\ln q (\theta; a, b, w)`, where `theta` is argument and
        `(a, b, w)` is parameter. Un-vectorized.
        
        Args:
            theta: np.array(shape=[self.get_dim()],
                            dtype=np.float32)
            
        Returns:
            np.array(shape=[], dtype=np.float32)
        """
        
        beta = self.beta(theta)  # shape: [self.get_num_peaks()]
        
        beta_max = np.max(beta)
        delta_beta = beta - beta_max
        
        return beta_max + np.log(np.sum(np.exp(delta_beta)))
       

    def sample(self, num_samples):
        """ Randomly sample `num_samples` samples from CGMD.
        
        Args:
            num_samples: int
            
        Returns:
            np.array(shape=[num_samples, self.get_dim()],
                     dtype=np.float32)
        """
        
        num_peaks = self.get_num_peaks()
        weights = self.get_w()
        mu = self.get_mu()
        sigma = self.get_sigma()
        
        def generate_sample():
            """ Generate one sample from CGMD. """
            
            index = np.random.choice(num_peaks, p=weights)
            samp = np.random.normal(loc=mu[index], scale=sigma[index])
            
            return samp
        
        return np.array([generate_sample() for _ in range(num_samples)])
    
    
    # --- Get-Functions ---
    
    def get_dim(self):
        return self._dim
    
    def get_num_peaks(self):
        return self._num_peaks
    
    def get_a(self):
        return self._a
    
    def get_w(self):
        return self._w
    
    def get_log_w(self):
        return self._log_w
    
    def get_mu(self):
        return self._mu
    
    def get_zeta(self):
        return self._zeta
    
    def get_sigma(self):
        return self._sigma
    
    def get_log_sigma(self):
        return self._sigma

    
    # --- Set-Functions ---
    
    def set_a(self, value):
        """
        Args:
            value: np.array(shape=self.get_a(), dtype=np.float32)
        """
        self._a = value
        self._w = softmax(self._a)
        self._log_w = log_softmax(self._a)
    
    def set_mu(self, value):
        """
        Args:
            value: np.array(shape=self.get_mu(), dtype=np.float32)
        """
        self._mu = value
    
    def set_zeta(self, value):
        """
        Args:
            value: np.array(shape=self.get_zeta(), dtype=np.float32)
        """
        self._zeta = value
        self._sigma = softplus(self._zeta)
        self._log_sigma = log_softplus(self._zeta) 


    # --- Copy-Function ---
    
    def copy(self):
        cgmd = CGMD(self.get_dim(),
                    self.get_num_peaks(),
                    self.get_epsilon())
        cgmd.set_a(self.get_a())
        cgmd.set_mu(self.get_mu())
        cgmd.set_sigma(self.get_sigma())
        return cgmd



if __name__ == '__main__':
    
    
    # --- The First Test ---
    
    import tools
    import matplotlib.pyplot as plt


    DIM = 10
    NUM_PEAKS = 5
    
    cgmd = CGMD(DIM, NUM_PEAKS)
    thetae = cgmd.sample(20)
    y = cgmd.vlog_pdf(thetae)
    
    # So far so good.