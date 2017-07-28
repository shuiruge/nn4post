#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Numpy Version.
"""


import numpy as np


from time import time
class Timer(object):
    """
    C.f. [here](https://www.huyng.com/posts/python-performance-analysis)
    """

    def __init__(self, verbose=True):
        self.verbose = verbose

    def __enter__(self):
        self.start = time()
        return self

    def __exit__(self, *args):
        self.end = time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000  # millisecs
        if self.verbose:
            print('=> elapsed time: %f secs' % self.secs)



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
        
        self._a = np.ones(shape=[num_peaks])
        self._b = np.random.normal(size=[dim, num_peaks])
        self._w = np.random.normal(size=[dim, num_peaks])
        
        # --- `prob` in the finite categorical distribution ---
        a = self.get_a()
        a_square = np.square(a)
        self._cat = a_square / np.sum(a_square)

        # --- :math:`\mu` and :math:`\sigma` for each Gaussian component ---
        b = self.get_b()
        w = self.get_w()
        self._components = [
            {
                'mu': -b[:,i] / w[:,i],  # shape: [dim]
                'sigma': np.diag(1 / np.abs(w[:,i])),  # shape: [dim, dim]
            }
            for i in range(num_peaks)]
            
    
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
        
        cat = self.get_cat()
        w = self.get_w()
        b = self.get_b()
        
        log_cat = np.expand_dims(
            np.log(cat),
            axis=0)  # shape: [1, self._num_peaks]
        log_gaussian = np.sum(
            (   -0.5 * np.square(np.expand_dims(theta, axis=2) * w + b)
                +0.5 * np.log(np.square(w) / (2 * 3.14))
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
        
        num_peaks = self.get_num_peaks()
        cat = self.get_cat()
        components = self.get_components()
        
        def generate_sample():
            """ as Gaussian mixture distribution. """
            
            i = np.random.choice(num_peaks, p=cat)
            
            samp = np.random.multivariate_normal(
                      mean=components[i]['mu'],
                      cov=components[i]['sigma'])
            
            return samp
            
        return np.asarray([generate_sample() for _ in range(num_samples)])
    
    
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
        return self._cat
    
    def get_components(self):
        return self._components
    
    
    # --- Set-Functions ---
    
    def _update_cat(self):
        """ Helper of `self.set_a()`. """ 
        a = self.get_a()
        a_square = np.square(a)
        self._cat = a_square / np.sum(a_square)
        return None
    
    def _update_components(self):
        """ Helper of `self.set_b()` and `self.set_w(). """
        num_peaks = self.get_num_peaks()
        b = self.get_b()
        w = self.get_w()
        self._components = [
            {
                'mu': -b[:,i] / w[:,i],  # shape: [dim]
                'sigma': np.diag(1 / np.abs(w[:,i])),  # shape: [dim, dim]
            }
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
        fgmd = FGMD(self.get_dim(), self.get_num_peaks())
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
        log_p: np.array(shape=[fgmd.get_dim()]) -> float
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
        log_p: np.array(shape=[fgmd.get_dim()]) -> float
        epsilon: float
            asdf   BUGS HEREIN
        clip_limit: float or None
            Clipping each element in the gradients of `a`, `b`, and `w` by
            `clip_limit`. If `None`, then no clipping to be established.
        num_samples: int
            Number of samples in the Monte Carlo integration herein.
            
    Returns:
        [
            nabla_perfm_by_a: np.array(shape=fgmd.get_a().shape),
            nabla_perfm_by_b: np.array(shape=fgmd.get_b().shape),
            nabla_perfm_by_w: np.array(shape=fgmd.get_w().shape),
        ]
    """

    log_q = fgmd.log_pdf
    a = fgmd.get_a()
    b = fgmd.get_b()
    w = fgmd.get_w()

    thetae = fgmd.sample(num_samples)

    beta_max, delta_beta = fgmd.beta(thetae)
    proportion = np.exp(delta_beta) \
               / np.expand_dims(np.sum(np.exp(delta_beta), axis=1), axis=1)  # shape: [num_samples, num_peaks]
             
    def nabla_perfm_sub(nabla_beta):
        """ Helper of `nabla_perfm()`.
        
        Args:
            nabla_beta_i: np.array(shape=?)
                :math:`\frac{\partial \beta}{\partial z}`
        Returns:
            np.array(shape=?)
        """
        
        x = np.expand_dims(log_q(thetae) - log_p(thetae) - 1, axis=1) \
            * proportion  # shape: [num_samples, num_peaks]
            
        if len(nabla_beta.shape) == 1:  # like `a`, with shape: [num_peaks].
            
            return np.mean(x * nabla_beta, axis=0)
        
        else:
            assert len(nabla_beta.shape) == 3  # like `b` and `w`, with shape: [num_samples, dim, num_peaks].
            
            return np.mean(
                np.expand_dims(x, axis=1) * nabla_beta,
                axis=0)
            
    # :math:`\frac{\partial beta_i}{\partial a_i} = \frac{2}{a_i}`
    nabla_beta_by_a = 2 / (a + epsilon)  # shape: [num_samples, num_peaks]
    # :math:`\frac{\partial beta_i}{\partial b_{ji}} = -(\theta_j w_{ji} + b_{ji})`
    nabla_beta_by_b = (- np.expand_dims(thetae, axis=2) * np.expand_dims(w, axis=0)
                       + np.expand_dims(b, axis=0))  # shape: [num_samples, dim, num_peaks]
    # :math:`\frac{\partial beta_i}{\partial b_{ji}} = -(\theta_j w_{ji} + b_{ji}) \theta_j + \frac{1}{w_{ji}}`
    nabla_beta_by_w = (- np.expand_dims(thetae, axis=2) * np.expand_dims(w, axis=0)
                       + np.expand_dims(b, axis=0)) * np.expand_dims(thetae, axis=2) \
                      + 1 / np.expand_dims(w + epsilon, axis=0)  # shape: [num_samples, dim, num_peaks]

    gradients = [
        nabla_perfm_sub(nabla_beta_by_a),  # shape: [num_peaks]
        nabla_perfm_sub(nabla_beta_by_b),  # shape: [dim, num_peaks]
        nabla_perfm_sub(nabla_beta_by_w),  # shape: [dim, num_peaks]
        ]

    # Clip the gradients
    for i, grad in enumerate(gradients):
        gradients[i] = np.clip(grad, -clip_limit, clip_limit)
        
    return gradients


def gradient_descent(log_p, fgmd, num_steps, learning_rate,
        epsilon=np.exp(-20), clip_limit=np.exp(7), num_samples=100):
    """ Using gradient descent method to update `fgmd` by **minimizing** the
        KL-divergence (as the performance) between `log_p` and `fgmd.log_pdf`.
        
    Numerical Treatments:
        Sometimes the gradients shall be clipped, since there's `1/a`, `1/w`
        in the gradients, which may lead to an overflow (thus returns `nan`)
        when `a` or `w` becomes small enough.
        
        And the epsilon is added (BUGS HEREIN) to the denominator of `1/a` and
        '1/w` in the computation of gradients to avoid `nan`.
        
    Args:
        fgmd: FGMD
        log_p: np.array(shape=[fgmd.get_dim()]) -> float
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
            This instance of `FGMD` is **not** the one in the argument (i.e.
            the `fgmd`), but a new instance initially copied from the `fgmd`.
    """
    
    fgmd0 = fgmd.copy()
    
    for step in range(num_steps):
        
        try:
            gradients = nabla_perfm(fgmd0, log_p,
                                    num_samples=num_samples,
                                    epsilon=epsilon,
                                    clip_limit=clip_limit)
            
            delta_a = -learning_rate * gradients[0]
            delta_b = -learning_rate * gradients[1]
            delta_w = -learning_rate * gradients[2]
            
            a = fgmd0.get_a()
            b = fgmd0.get_b()
            w = fgmd0.get_w()
            
            fgmd0.set_a(a + delta_a)
            fgmd0.set_b(b + delta_b)
            fgmd0.set_w(w + delta_w)
        
        except Exception as e:
            
            print('ERROR')
            print('step at {0}'.format(step))
            raise Exception(e)
           
    return fgmd0


# --- Test ---

DIM = 100
NUM_PEAKS = 2

fgmd = FGMD(DIM, NUM_PEAKS)
log_p = lambda theta: (-0.5 * np.sum(np.square(theta), axis=1)
                       - 0.5 * np.log(2 * 3.14))

# Before gradient descent
print('b: {0}'.format(fgmd.get_b()))
print('w: {0}'.format(fgmd.get_w()))
print('cat: {0}'.format(fgmd.get_cat()))
print('components: {0}'.format(fgmd.get_components()))
old_performance = performance(log_p, fgmd)
print('performance: {0}'.format(old_performance))

# Making gradient descent
with Timer():
    fgmd = gradient_descent(log_p, fgmd, num_steps=500, learning_rate=0.001)
    
# After gradient descent
print('After updating ......\n')
new_performance = performance(log_p, fgmd)
print('updated performance:\n\t{0}  -->  {1}\n'.format(
        old_performance, new_performance))
print('updated a: {0}'.format(fgmd.get_a()))
print('updated b (shall be near `0`): {0}'.format(fgmd.get_b()))
print('updated w (shall be near `+-1`): {0}'.format(fgmd.get_w()))
print('updated cat: {0}'.format(fgmd.get_cat()))
print('updated components: {0}'.format(fgmd.get_components()))

''' outputs of one arbitrary (i.e. without any post-selection) trial:
    >>> updated_performance:
    ...     779.4549314953962  -->  0.19749876522128815
    ... updated cat: ?
    ... updated components:
    ...     ?
    
    with parameters ..code:
        
        DIM = 100
        NUM_PEAKS = 2
        
        fgmd = FGMD(DIM, NUM_PEAKS)
        log_p = lambda theta: (-0.5 * np.sum(np.square(theta), axis=1)
                               - 0.5 * np.log(2 * 3.14))
    
    Looks good on performance. But I do not know if there's some bug within
    the code. Thus a careful overall check is called for.
'''
