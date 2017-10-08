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
import tensorflow as tf
import edward
from edward.models import RandomVariable, NormalWithSoftplusScale



class KLqp(edward.KLqp):
  """ `KLqp` class by pure TensorFlow.


  TODO:
    Testing shows that this version is still costy on timing and memory. And
    it greatly ceases when reducing `n_samples`. So, use tf.profiler to find out
    where it is costy.
  """

  def __init__(self, model, *arg, **kwargs):
    r""" The same as the `__init__` of `ed.KLqp`, except for `model`

    Args:
      model:
        Callable, mapping from paramters to prediction-values.

        Example:

          ```python
          x = tf.placeholder(...)  # used in `feed_dict`.

          def model(w, b):
              ''' Regression model. '''
              # `x` employs the variable in the outer-frame, so that ensure
              # only parameters (e.g. the `w` and `b`) are in the arguments.
              return tf.nn.sigmoid( x * w + b )
          ```

      *args, **kwargs:
        Arguments in `self.initialize`.
    """

    self.model = model
    super(KLqp, self).__init__(*arg, **kwargs)


  def build_loss_and_gradients(self, var_list):
    r""" Override the associated method in `ed.KLqp` by pure TensorFlow.


    ### Efficiency

    In [link](https://www.tensorflow.org/api_docs/python/tf/contrib/bayesflow/\
    entropy/elbo_ratio):

    > The term E_q[ Log[p(Z)] ] is always computed as a sample mean. The term
    > E_q[ Log[q(z)] ] can be computed with samples, or an exact formula if
    > q.entropy() is defined. This is controlled with the kwarg form.

    Now it is clear. The surprising efficiency of TensorFlow in calculating
    `elbo` is gained by and only by the usage of analytic entropy called by
    `q.entropy()`. That mystery encountered in Shanghai.

    However, `tf.distributions.Mixture` has no `entropy` method. Instead, use
    method `entropy_lower_bound`. This is plausible, c.f. "Derivation" section.

    So, all left is how to efficiently compute the **value** of likelihood on
    the **sampled values** of parameters.


    ### Derivation

    Denote $z$ element in parameter-space, $D$ data, $c_i$ the category
    distribution logist, and $q_i$ the component distribution. We have

    $$ \textrm{KL} ( q \mid p ) = - \textrm{ELBO} + \ln p(D), $$

    wherein

    $$ \textrm{ELBO} := H[q] + E_q [ \ln p(z, D) ] $$

    as usual, thus

    $$ \textrm{ELBO} \leq \ln p(D). $$

    With `entropy_lower_bound` instead of `entropy`, and let

    $$ \mathcal{E} := \sum_i c_i H[q_i] + E_q [ \ln p(z, D) ], $$

    we have, since $ \sum_i c_i H[q_i] \leq H[q] $,

    $$ \mathcal{E} \leq \textrm{ELBO} \leq \ln p(D). $$

    Thus, if let $ \textrm{loss} := - \mathcal{E} $, optimizer can find the
    minimum of the loss-function, as expected.


    ### On Edward

    It seems that `edward.utils.copy` is costy, and seems non-essential. Thus
    we shall avoid employing it.
    """

    # Construct `loss`
    p_log_prob = [0.0] * self.n_samples

    # Get all samples in one go
    dict_samples = {}
    for z, qz in six.iteritems(self.latent_vars):
      z_samples = qz.sample(self.n_samples)
      z_samples = tf.unstack(z_samples)
      dict_samples[z] = z_samples  # the `z` is for labeling.

    # (Temporally) assume that the data obeys a normal distribution,
    # realized by Gauss's limit-theorem
    dict_chi_square = {}
    for y, y_data in six.iteritems(self.data):
      normal = NormalWithSoftplusScale(loc=y_data, scale=y.scale)
      dict_chi_square[y] = lambda x: normal.log_prob(x)

    # Construct `p_log_prob` by sampling
    for s in range(self.n_samples):

      # Compute prior values
      for z in six.iterkeys(self.latent_vars):
        p_log_prob[s] += tf.reduce_sum(
            z.log_prob(dict_samples[z][s]) \
            * self.scale.get(z, 1.0)
        )

      # Compute likelihood values
      params = {}
      for z, z_samples in six.iteritems(dict_samples):
        params[z.name] = z_samples[s]
      for y, y_data in six.iteritems(self.data):
        p_log_prob[s] += tf.reduce_mean(
            dict_chi_square[y](self.model(**params))
            * self.scale.get(y, 1.0)
        )

    p_log_prob = tf.reduce_mean(p_log_prob)


    q_log_prob = 0.0
    for z, qz in six.iteritems(self.latent_vars):
      try:
        q_log_prob += - tf.reduce_mean(qz.entropy())
      except AttributeError:
        q_log_prob += - tf.reduce_mean(qz.entropy_lower_bound())

    loss = - ( p_log_prob - q_log_prob )


    # Nothing is to be modified in the following
    grads = tf.gradients(loss, var_list)
    grads_and_vars = list(zip(grads, var_list))

    print(' --- Succeed in Overriding --- ')
    return (loss, grads_and_vars)
