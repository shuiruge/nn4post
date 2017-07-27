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
            
    
    def beta(self, theta):
        """ The :math:`\beta` in `'../docs/nn4post.tm'`.
        
        Args:
            theta: np.array(shape=[None, self.get_dim()], dtype=float32)
            
        Returns:
            (
                beta_max:
                    np.array(shape=[None])
                delta_beta:
                    np.array(shape=[None, self.get_num_peaks()])
            )
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
        
        beta0 = log_cat + log_gaussian  # shape: [None, self._num_peaks]
        
        beta_max = np.max(beta0, axis=1)  # shape: [None]
        delta_beta = beta0 - np.expand_dims(beta_max, axis=1)  # shape: [None, self._num_peaks]
    
        return (beta_max, delta_beta)
        

    def log_pdf(self, theta):
        """ :math:`\ln q (\theta; a, b, w)`, where `theta` is argument and
        `(a, b, w)` is parameter.
        
        Args:
            theta: np.array(shape=[None, self.get_dim()], dtype=float32)
            
        Returns:
            np.array(shape=[None], dtype=float32)
        """
        
        beta_max, delta_beta = self.beta(theta)
        
        return beta_max + np.log(np.sum(np.exp(delta_beta), axis=1))  # shape: [None]
    

    def sample(self, num_samples):
        """
        Args:
            num_samples: int
        Returns:
            np.array(shape=[num_samples, self.get_dim()], dtype=float32)
        """
        
        def generate_sample():
            """ as Gaussian mixture distribution. """
            
            i = np.random.choice(self._num_peaks, p=self._cat)
            
            samp = np.random.multivariate_normal(
                      mean=self._components[i]['mu'],
                      cov=self._components[i]['sigma'])
            
            return samp
            
        return np.asarray([generate_sample() for _ in range(num_samples)])
    
    
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
    
    def set_a(self, value):
        self._a = value
        return None
    
    def set_b(self, value):
        self._b = value
        return None
    
    def set_w(self, value):
        self._w = value
        return None
    

def performance(log_p, fgmd, num_samples=100):
    """ Use KL-divergence as performance of fitting a posterior `log_p` by a
        finite Gaussian mixture distribution (FGMD).\
    
    Args:
        log_p: np.array(shape=[fgmd.get_dim()]) -> float
        fgmd: FGMD
        num_sample: int
            For Monte Carlo integration.
    Returns:
        float
    """
    
    thetae = fgmd.sample(num_samples)
    log_q = fgmd.log_pdf
    
    kl_divergence = np.mean(log_p(thetae) - log_q(thetae))
    return kl_divergence



def nabla_perfm(log_p, fgmd, num_samples=100):
    
    num_peaks = fgmd.get_num_peaks()
    dim = fgmd.get_dim()
    log_q = fgmd.log_pdf
    a = fgmd.get_a()
    b = fgmd.get_b()
    w = fgmd.get_w()

    thetae = fgmd.sample(num_samples)

    beta_max, delta_beta = fgmd.beta(thetae)
    proportion = np.exp(delta_beta) \
               / np.expand_dims(np.sum(np.exp(delta_beta), axis=1), axis=1)  # shape: [num_samples, num_peaks]
             
    def nabla_perfm_sub(nabla_beta):
        """
        Args:
            nabla_beta_i: np.array(shape=?)
                :math:`\frac{\partial \beta}{\partial z}`
        Returns:
            np.array(shape=)
        """
        
        x = np.expand_dims(log_p(thetae) - log_q(thetae) - 1, axis=1) \
            * proportion  # shape: [num_samples, num_peaks]
            
        if len(nabla_beta.shape) == 1:  # like `a`, with shape: [num_peaks].
            
            return np.mean(x * nabla_beta, axis=0)
        
        else:
            assert len(nabla_beta.shape) == 3  # like `b` and `w`, with shape: [num_samples, dim, num_peaks].
            
            return np.mean(
                np.expand_dims(x, axis=1) * nabla_beta,
                axis=0)
            

    nabla_beta_by_a = 2 / a  # shape: [num_samples, num_peaks]
    nabla_beta_by_b = (- np.expand_dims(thetae, axis=2) * np.expand_dims(w, axis=0)
                       + np.expand_dims(b, axis=0))  # shape: [num_samples, dim, num_peaks]
    nabla_beta_by_w = (- np.expand_dims(thetae, axis=2) * np.expand_dims(w, axis=0)
                       + np.expand_dims(b, axis=0)) * np.expand_dims(thetae, axis=2) \
                      + 1 / np.expand_dims(w, axis=0)  # shape: [num_samples, dim, num_peaks]

    return (
        nabla_perfm_sub(nabla_beta_by_a),
        nabla_perfm_sub(nabla_beta_by_b),
        nabla_perfm_sub(nabla_beta_by_w)
        )



# --- Test ---

DIM = 100
NUM_PEAKS = 2

NUM_SAMPLES = 50

thetae = np.random.uniform(size=[NUM_SAMPLES, DIM])
fgmd = FGMD(DIM, NUM_PEAKS)
log_p = lambda theta: -0.5 * np.sum(np.square(theta), axis=1)

for _ in nabla_perfm(log_p, fgmd, num_samples=NUM_SAMPLES):
    print(_.shape)