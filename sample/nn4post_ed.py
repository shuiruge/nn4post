#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description
-----------
Implementation of the model illustrated in '../docs/nn4post.pdf', via
TensorFlow and Edward.

Documentation
-------------
C.f. '../docs/nn4post.pdf'.
"""


import os
import tensorflow as tf
import numpy as np
import edward as ed
# -- `contrib` module in TensorFlow version: 1.3
from tensorflow.contrib.distributions import \
    Categorical, Mixture, MultivariateNormalDiagWithSoftplusScale
from tensorflow.contrib.bayesflow import entropy


# -- For testing (and debugging)
tf.set_random_seed(42)
np.random.seed(42)


class Nn4post(VariationalInference):
  """

  """

  def __init__(self, *args, **kwargs):
    super(KLqp, self).__init__(*args, **kwargs)


  def initialize(self, n_peaks, n_samples=1, *args, **kwargs):
    """Initialize inference algorithm. It initializes hyperparameters
    and builds ops for the algorithm's computation graph.

    Args:
      n_samples: int, optional.
        Number of samples from variational model for calculating
        stochastic gradients.
    """

    self.n_peaks = n_peaks
    self.n_samples = n_samples
    self.cat_gauss_mix_dist = self._get_cat_gauss_mix_dist()
    return super(Nn4post, self).initialize(*args, **kwargs)


  def build_loss_and_gradients(self, var_list):
    """
    """

    if self.logging:
      tf.summary.scalar("loss", loss,
                        collections=[self._summary_key])


    return

  def _get_cat_gauss_mix_dist(self):

    # -- Parameters
    a_shape = [self.n_peaks]
    mu_shape = [self.n_peaks, dim]
    zeta_shape = [self.n_peaks, dim]

    # -- initialize the values of variables of `cat_gauss_mix_dist`.
    a_val = np.zeros(shape=a_shape)
    mu_val = np.random.normal(size=mu_shape)
    # To make `softplus(self._init_zeta) == np.ones(self._zeta_shape)`
    zeta_val = np.log((np.e-1) * np.ones(zeta_shape))

    
