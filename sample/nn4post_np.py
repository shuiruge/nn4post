#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Numpy Version.
"""

import numpy as np


_DEFAULT_NUM_SAMPLES = 1000
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
    """ :math:`\frac{1}{1 + \exp(-x)}`. Apply to `x` element-wisely.
    
    Args:
        x: array_like
    
    Returns:
        ndarray
    """
    return 1 / (1 + np.exp(-x))


def softplus(x):
    """ :math:`\ln(1 + \exp(x))`. Apply to `x` element-wisely.
    
    Args:
        x: array_like
        
    Returns:
        ndarray
    """
    return np.log(1 + np.exp(x))


def log_softplus(x):
    """ :math:`\ln(1 + \exp(x))`. Apply to `x` element-wisely.
    
    NOTE:
        Numerical trick is taken. That is, if one element of `x`, say `y` is
        tiny, so that :math:`\exp(y) \ll 1 \to \ln(\ln(1 + \exp(y))) \approx y`.
        
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
    

def inverse_softplus(x):
    """
    Args:
        x: array_like
        
    Returns:
        ndarray
    """
    return np.log(np.exp(x) - 1)




class CGMD(object):
    """ Categorical Gaussian Mixture Distribution.
    
    C.f. WikiPedia_.
    
    .. _WikiPedia: https://en.wikipedia.org/wiki/Mixture_distribution#Finite_\
                   and_countable_mixtures
        
    NOTE:
        `a`, `mu`, and `zeta` are randomly initialized by standard normal
        distribution.
                   
    Args:
        dim: int
            The dimension of domain the PDF of the CGMD acts on.
        num_peaks: int
            The number of categories of the CGMD (Gaussianities, thus we call
            peaks).
        
    Methods:
        beta,
        log_pdf, vlog_pdf (vectorized_log_pdf),
        sample,
        get_a, set_a,
        get_mu, set_mu,
        get_zeta, set_zeta,
        get_w, get_log_w,
        get_sigma, get_log_sigma
    """
    
    def __init__(self, dim, num_peaks):
        
        self._dim = dim
        self._num_peaks = num_peaks
        
        # --- Initialize a, mu, and zeta ---
        self._a = np.ones(shape=[self._num_peaks])
        self._mu = np.zeros(shape=[self._num_peaks, self._dim])
        self._zeta = inverse_softplus(
            np.ones(shape=[self._num_peaks, self._dim]))
        
        # shape: [self._num_peak]
        self._w = softmax(self._a)
        self._log_w = log_softmax(self._a)
        # shape: [self._num_peak, self._dim]
        self._sigma = softplus(self._zeta)
        self._log_sigma = log_softplus(self._zeta)
        
        
    
    def beta(self, theta):
        """ The :math:`\beta` in `'../docs/nn4post.tm'`.
        
        Args:
            theta: np.array(shape=[self.get_dim()], dtype=np.float32)
            
        Returns:
            np.array(shape=[self.get_num_peaks()], dtype=np.float32)
        """
        
        dim = self.get_dim()
        # `num_peaks = self.get_num_peaks()`
        log_w = self.get_log_w()  # shape: [num_peaks]
        mu = self.get_mu()  # shape: [num_peaks, dim]
        sigma = self.get_sigma()  # shape: [num_peaks, dim]
        log_sigma = self.get_log_sigma()  # shape: [num_peaks, dim]
        
        # shape: [1, dim]
        theta = np.expand_dims(theta, axis=0)
        
        # shape: [num_peaks]
        return (log_w - dim * 0.5 * np.log(2 * np.pi)
                - np.sum(log_sigma + 0.5*np.square((theta-mu)/sigma),
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
            
            # Since when `self.get_dim() == 1`, `np.random.normal` generates
            # scalars instead of arraies, we have to convert scalars to arries
            # manually
            if self.get_dim() == 1:
                samp = [samp]
            
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
        return self._log_sigma

    
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
    
    
def elbo(cgmd, log_p, num_samples=_DEFAULT_NUM_SAMPLES):
    """ ELBO between `cgmd.log_pdf` and `log_p`.
    
    Args:
        cgmd: CGMD
        log_p: Map(**{theta:
                        np.array(shape[=cgmd.get_dim()], dtype=np.float32)},
                   np.array(shape=[], dtype=np.float32))
            The :math:`\ln(p(\theta))` in the documentation.
        num_samples: int
            Number of samples in the Monte Carlo integration.
    
    Returns:
        np.array(shape=[], dtype=np.float32)
    """
    
    # `vectorized_log_q` and `vectorized_log_p` both returns `np.array` with
    # shape `[num_samples]` and dtype `np.float32`.
    vectorized_log_q = vectorize(cgmd.log_pdf)
    vectorized_log_p = vectorize(log_p)
    
    thetae = cgmd.sample(num_samples)
    return np.mean(vectorized_log_q(thetae) - vectorized_log_p(thetae))


def grad_elbo(cgmd, log_p, num_samples=_DEFAULT_NUM_SAMPLES):
    """ Gradient of the ELBO between `cgmd.log_pdf` and `log_p`.
    
    TODO: debug.
    
    Args:
        cgmd: CGMD
        log_p: Map(**{theta:
                        np.array(shape=[cgmd.get_dim()], dtype=np.float32)},
                   np.array(shape=[], dtype=np.float32))
            The :math:`\ln(p(\theta))` in the documentation.
        num_samples: int
            Number of samples in the Monte Carlo integration.
    
    Returns:
        {
            'a':
                np.array(shape=[cgmd.get_num_peaks()],
                         dtype=np.float32),
            'mu':
                np.array(shape=[cgmd.get_num_peaks(), cgmd.get_dim()],
                                dtype=np.float32),
            'zeta':
                np.array(shape=[cgmd.get_num_peaks(), cgmd.get_dim()],
                                dtype=np.float32)
        }
    """
    # `dim = cgmd.get_dim()`
    # `num_peaks = cgmd.get_num_peaks()`
    log_q = cgmd.log_pdf
    
    # -- Re-shape all in the "standard shape", wherein the first axis is for
    #    samples, the second for categories (peaks), and the third for dim.
    # shape: [1, num_peaks]
    w = np.expand_dims(cgmd.get_w(), axis=0)    
    # shape: [1, num_peaks, dim]
    mu = np.expand_dims(cgmd.get_mu(), axis=0)
    # shape: [1, num_peaks, dim]
    zeta = np.expand_dims(cgmd.get_zeta(), axis=0)
    # shape: [1, num_peaks, dim]
    sigma = np.expand_dims(cgmd.get_sigma(), axis=0)
    
    # -- Vectorizations
    vlog_q = vectorize(log_q)
    vlog_p = vectorize(log_p)
    vbeta = vectorize(cgmd.beta)
    
    # shape: [num_samples, dim]
    thetae = cgmd.sample(num_samples)
    
    # shape: [num_samples]
    propotion_1 = (vlog_q(thetae) - vlog_p(thetae) + 1)
    # shape: [num_samples, num_peaks]
    propotion_2 = softmax(vbeta(thetae))
    # shape: [num_samples, num_peaks]
    propotion = np.expand_dims(propotion_1, axis=1) * propotion_2

    # -- Re-shape for later.                          
    # shape = [num_samples, 1, dim]
    thetae = np.expand_dims(thetae, axis=1)
    
    # shape: [1, num_peaks]      
    partial_beta_by_a = 1 - w
    # shape: [num_samples, num_peaks, dim]
    partial_beta_by_mu = (thetae - mu) / np.square(sigma)
    # shape: [num_samples, num_peaks, dim]
    partial_beta_by_zeta = ((-1 + np.square((thetae-mu)/sigma))
                            * sigmoid(zeta) / sigma)
    # shape: [num_peaks]
    partial_elbo_by_a = np.mean(propotion * partial_beta_by_a, axis=0)
    
    # -- Re-shape for later.
    # shape: [num_samples, num_peaks, 1]
    propotion = np.expand_dims(propotion, axis=2)
    
    # shape: [num_peaks, dim]
    partial_elbo_by_mu = np.mean(propotion * partial_beta_by_mu, axis=0)
    # shape: [num_peaks, dim]
    partial_elbo_by_zeta = np.mean(propotion * partial_beta_by_zeta, axis=0)
    
    return {
        'a': partial_elbo_by_a,
        'mu': partial_elbo_by_mu,
        'zeta': partial_elbo_by_zeta,
        }
    
    
def gradient_descent(cgmd, log_p, learning_rate):
    """ Modifies (upates) the `cgmd` by gradient descent algorithm.
    
    Args:
        cgmd: CGMD
        log_p: Map(**{theta:
                        np.array(shape[=cgmd.get_dim()], dtype=np.float32)},
                   np.array(shape=[], dtype=np.float32))
        learning_rate: float
    """
    
    a = cgmd.get_a()
    mu = cgmd.get_mu()
    zeta = cgmd.get_zeta()
    
    grads = grad_elbo(cgmd, log_p)
    
    a = a - grads['a'] * learning_rate
    mu = mu - grads['mu'] * learning_rate
    zeta = zeta - grads['zeta'] * learning_rate
    
    cgmd.set_a(a)
    cgmd.set_mu(mu)
    cgmd.set_zeta(zeta)
    
                       
    



if __name__ == '__main__':
    
    import tools
    import matplotlib.pyplot as plt
    
    # -- Double-check gradients with `nn4post_np.py`
    NUM_PEAKS = 1
    DIM = 1
    def log_p(theta):
        return (-0.5 * np.sum(np.square(theta-100))
                -0.5 * np.log(2 * np.pi) * DIM)    
    cgmd = CGMD(DIM, NUM_PEAKS)
    
    grad_a_list = []
    grad_mu_list = []
    grad_zeta_list = []
    for i in range(1000):
        grads = grad_elbo(cgmd, log_p)
        grad_a_list.append(grads['a'])
        grad_mu_list.append(grads['mu'])
        grad_zeta_list.append(grads['zeta'])
    print('grad_a: {0}'.format(np.mean(grad_a_list)))
    print('grad_mu: {0}'.format(np.mean(grad_mu_list)))
    print('grad_zeta: {0}'.format(np.mean(grad_zeta_list)))
    
    
    
    
#    # --- Test `CGMD` ---
#    
#    DIM = 2
#    NUM_PEAKS = 1
#    
#    cgmd = CGMD(DIM, NUM_PEAKS)
#    thetae = np.sort(cgmd.sample(20), axis=0)
#    y = vectorize(cgmd.log_pdf)(thetae)
#    
#    print('mu, zeta, sigma: ',
#          cgmd.get_mu(),
#          cgmd.get_zeta(),
#          cgmd.get_sigma())
#    plt.plot(thetae, y, '-')
#    plt.show()
#    
#    # So far so good.
#    
#    
#    # --- Test `elbo()` ---
#    
#    DIM = 1
#    NUM_PEAKS = 1
#    
#    def log_p(theta):
#        """
#        Args:
#            theta: np.array(shape=[DIM], dtype=np.float32)
#        Returns:
#            np.array(shape=[], dtype=np.float32)
#        """
#        return (-0.5 * np.log(2 * np.pi) * DIM
#                -0.5 * np.sum(np.square(theta)))
#    
#    cgmd = CGMD(DIM, NUM_PEAKS)
#    el = elbo(cgmd, log_p)
#    
#    thetae = np.sort(cgmd.sample(_DEFAULT_NUM_SAMPLES))
#    y_1 = vectorize(cgmd.beta)(thetae)
#    plt.plot(thetae, y_1, '.')
#    y_2 = vectorize(log_p)(thetae)
#    plt.plot(thetae, y_2, '-')
#    plt.show()
#    
#    # So far so good.
#
#    
#    # --- Test `grad_elbo()` ---
#    
#    DIM = 1
#    NUM_PEAKS = 1
#    num_samples = 10 ** 2
#    
#    cgmd = CGMD(DIM, NUM_PEAKS)
#    
#    def log_p(theta):
#        """
#        Args:
#            theta: np.array(shape=[DIM], dtype=np.float32)
#        Returns:
#            np.array(shape=[], dtype=np.float32)
#        """
#        return (-0.5 * np.log(2 * np.pi) * DIM
#                -0.5 * np.sum(np.square(theta-10)))    
#    
#    cgmd = CGMD(DIM, NUM_PEAKS)
#    grads = grad_elbo(cgmd, log_p)
#    grad_a = grads['a']
#    grad_mu = grads['mu']
#    grad_zeta = grads['zeta']
#    
#    # So far so good.


## --- Test `gradient_descent()` on **mean-field approximation ** ---
#
#    DIM = 20
#    NUM_PEAKS = 1  # shall be `1` for **mean-field approximation**.
#    
#    def log_p(theta, mean=3):
#        """
#        Args:
#            theta: np.array(shape=[DIM], dtype=np.float32)
#        Returns:
#            np.array(shape=[], dtype=np.float32)
#        """
#        return (-0.5 * np.log(2 * np.pi) * DIM
#                -0.5 * np.sum(np.square(theta - mean)))
#        
#    cgmd = CGMD(DIM, NUM_PEAKS)    
#    elbo_log = [elbo(cgmd, log_p)]
#    
#    for step in range(500):
#        gradient_descent(cgmd, log_p, 0.1)
#        
#        grads = grad_elbo(cgmd, log_p)
#        elbo_log.append(elbo(cgmd, log_p))
#        
#        if step % 10 == 0:
#            print('grad_by_zeta (max, min): ',
#                  np.max(grads['zeta'].reshape([-1])),
#                  np.min(grads['zeta'].reshape([-1])))
#            print('zeta (max, min): ',
#                  np.max(cgmd.get_zeta().reshape([-1])),
#                  np.min(cgmd.get_zeta().reshape([-1])))
#            print('elbo: ', elbo(cgmd, log_p))
#            
#    plt.plot(elbo_log)
#    plt.show()
#    
#    if cgmd.get_dim() == 1:
#        # -- Plot `cgmd.log_pdf()` only.
#        thetae = np.sort(cgmd.sample(_DEFAULT_NUM_SAMPLES), axis=0)
#        y_1 = vectorize(cgmd.log_pdf)(thetae)
#        plt.plot(thetae, y_1, '.')
#        plt.show()
#        # -- Plot `cgmd.log_pdf()` and `log_p()` together
#        thetae = np.sort(cgmd.sample(_DEFAULT_NUM_SAMPLES), axis=0)
#        y_1 = vectorize(cgmd.log_pdf)(thetae)
#        plt.plot(thetae, y_1, '.')
#        y_2 = vectorize(log_p)(thetae)
#        plt.plot(thetae, y_2, '-')
#        plt.show()
