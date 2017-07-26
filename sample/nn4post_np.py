#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Numpy Version.
"""


import numpy as np




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
                   
    Args:
        dim: int
        num_peaks: int
        
    Methods:
        log_pdf, sample, 
        get_a, set_a,
        get_b, set_b,
        get_w, set_w,
        get_cat, get_components
    """
    
    def __init__(self, dim, num_peaks):
        
        self._dim = dim
        self._num_peaks = num_peaks
        
        self._a = np.random.normal(size=[self._num_peaks])
        self._b = np.random.normal(size=[self._dim, self._num_peaks])
        self._w = np.random.normal(size=[self._dim, self._num_peaks])
        
        # --- `prob` in the finite categorical distribution ---
        self._cat = np.square(self._a) / np.sum(np.square(self._a))

        # --- :math:`\mu` and :math:`\sigma` for each Gaussian component ---
        self._components = []
        w_square = np.square(self._w)
        for i in range(self._num_peaks):
            mu = self._b[:,i] / w_square[:,i]  # shape: [self._dim]
            sigma = np.diag(1 / w_square[:,i])  # shape: [self._dim, self._dim]
            self._components.append({'mu': mu, 'sigma': sigma})
            

    def log_pdf(self, theta):
        """ :math:`\ln q (\theta; a, b, w)`, where `theta` is argument and
        `(a, b, w)` is parameter.
        
        Args:
            theta: np.array(shape=[None, self._dim], dtype=float32)
            
        Returns:
            np.array(shape=[None], dtype=float32)
        """
        log_cat = np.expand_dims(
            np.log(np.square(self._a)),
            axis=0)  # shape: [1, self._num_peaks]
        log_gaussian = np.sum(
            (   -0.5 * np.square(np.expand_dims(theta, axis=2) \
                     * np.square(self._w) + self._b)
                +0.5 * np.log(np.square(self._w) / (2 * 3.14))
            ),
            axis=1)  # shape: [None, self._num_peaks]
    
        beta = log_cat + log_gaussian  # shape: [None, self._num_peaks]
        
        beta_max = np.max(beta, axis=1)  # shape: [None]
        delta_beta = beta - np.expand_dims(beta_max, axis=1)  # shape: [None, self._dim]
        
        return beta_max + np.log(np.sum(np.exp(delta_beta), axis=1))  # shape: [None]
    

    def sample(self, num_samples):
        """
        Args:
            num_samples: int
        Returns:
            np.array(shape=[num_samples, self._dim], dtype=float32)
        """
        
        def generate_sample():
            """ as Gaussian mixture distribution. """
            
            i = np.random.choice(self._num_peaks, p=self._cat)
            
            samp = np.random.multivariate_normal(
                      mean=self._components[i]['mu'],
                      cov=self._components[i]['sigma'])
            
            return samp
            
        return np.asarray([generate_sample() for _ in range(num_samples)])
        


# --- Test ---

DIM = 100
NUM_PEAKS = 10

thetae = np.random.uniform(size=[50, DIM])
gm = FGMD(DIM, NUM_PEAKS)
print(gm.log_pdf(thetae).shape)
print(gm.sample(30).shape)
