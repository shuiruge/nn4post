#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description
-----------
`KLqp` class by pure TensorFlow.


Versions
--------
- Python: 3.6
- TensorFlow: 1.3.0
- Edward: 1.3.3


Remarks
-------
* Hard to impliment, since the components in the variable of the posterior are
  **not** independent with each other!
"""


import six
import edward as ed
from edward import RandomVariable
import tensorflow as tf
from tensorflow.contrib.bayesflow import variational_inference
from tensorflow.contrib.bayesflow import stochastic_tensor


class KLqp(ed.KLqp):
  """ XXX """

  def __init__(self, *arg, **kwargs):
    """ The same as the `__init__` of `ed.KLqp`. """

    super(KLqp, self).__init__(*arg, **kwargs)


  def build_loss_and_gradients(self, var_list):
    """ Override the associated method in `ed.KLqp` by pure TensorFlow. """

    # Construct `loss`
    for x, qx in six.iteritems(self.data):
      if isinstance(x, RandomVariable):
        print(' --- ', x.name, qx.name)  # test!
        variational_with_prior = {
            stochastic_tensor.StochasticTensor(q): p
            for p, q in six.iteritems(self.latent_vars)}
        loss = - variational_inference.elbo(
            log_likelihood=tf.reduce_sum(x.log_prob(qx)),
            variational_with_prior=variational_with_prior)
        break

    # Nothing is to be modified in the following
    grads = tf.gradients(loss, var_list)
    grads_and_vars = list(zip(grads, var_list))

    print(' --- Succeed in Overriding --- ')
    return (loss, grads_and_vars)
