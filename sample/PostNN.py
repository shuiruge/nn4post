#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description
-----------
`PostNN` class that inherits from `edward.Inference`, implemented by pure
TensorFlow.


Versions
--------
- Python: 3.6
- TensorFlow: 1.3.0
- Edward: 1.3.3
"""


import six
import tensorflow as tf
import edward
from edward.models import Categorical, NormalWithSoftplusScale
from mixture_same_family import MixtureSameFamily



class PostNN(edward.Inference):
  r""" `KLqp` class by pure TensorFlow.

  ### Example

  Logits regression:

  ```python

  import tensorflow as tf
  from edward.models import Normal
  from somewhere import generate_dataset  # needs override.

  # DATA
  x_data = tf.placeholder([None, n_inputs], name='x_data')
  y_data = tf.placeholder([None, n_outputs], name='y_data')
  y_error = tf.placeholder([None, n_outputs], name='y_error')
  x_train, y_train, y_error_train = generate_dataset(...)

  # MODEL
  n_inputs = 5
  with tf.name_scope('model'):
      w = Normal(loc=tf.zeros([n_inputs, n_outputs]),
                 scale=tf.ones([n_inputs, n_outputs]),
                 name='w')
      b = Normal(loc=tf.zeros([n_outputs]),
                 scale=tf.ones([n_outputs]),
                 name='b')
      def model(w, b):
          return tf.nn.sigmoid(tf.matmul(x_data, w) + b)
      prediction = model(w, b)
      y = Normal(loc=prediction,
                 scale=y_error,
                 name='y')

  # INFERENCE
  # -- Argument `cats` can be a `int`, such `n_cats` for all variables;
  #    or a `dict` like `{'b': np.ones([n_output]) * 5, 'w': ...}`, with
  #    each element of the array represents how many cats are for the
  #    variable.
  inference = PostNN(
      n_cats=5,
      model=model,
      params={'w': w, 'b': b},
      data={ x_data: x_train,
             y_data: y_train,
             y_error: y_error_train },
      )
  inference.run(...)
  ```

  TODO:
    Testing shows that this version is still costy on timing and memory. And
    it greatly ceases when reducing `n_samples`. So, use tf.profiler to find out
    where it is costy.
  """

  def __init__(self, cats, model, params, *arg, **kwargs):
    r"""
    Args:
      n_cats:
        It can be a `int`, such `n_cats` for all variables;
        or a `dict` like `{'b': np.ones([n_output]) * 5, 'w': ...}`, with
        each element of the array represents how many cats are for the
        variable.

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

      data:
        XXX

      *args, **kwargs:
        Arguments in `self.initialize`.
    """

    self._check_args(cats, model, param_dict)

    self._model = model
    self._param_dict = param_dict

    self._model_arg_names = self.get_arg_names(model)
    self._event_shape_dict = {name: rv.event_shape
                              for name, rv in self._param_dict.items()}
    self._params_space_dim = sum([np.prod(event_shape) for event_shape
                                  in self._event_shape_dict.values()])

    '''
    if n_cats not isinstance(list):
      self._n_cats = {name: n_cats * np.ones(rv.event_shape)
                    for name, rv in self._param_dict.items()}
    else:
      self._n_cats = n_cats

    n_cats_flatten = [self._n_cats[name] for name in self._model_arg_names]
    n_cats_flatten = np.concatenate([_.flatten() for _ in n_cats_flatten])

    # Let `n_cats` just an `int`.
    '''

    super(Inference, self).__init__(*arg, **kwargs)



  def initialize(self, optimizer=None,
                 global_step=None, *args, **kwargs):
    """Initialize inference algorithm. It initializes hyperparameters
    and builds ops for the algorithm's computation graph.
    Args:
      optimizer: str or tf.train.Optimizer, optional.
        A TensorFlow optimizer, to use for optimizing the variational
        objective. Alternatively, one can pass in the name of a
        TensorFlow optimizer, and default parameters for the optimizer
        will be used.
      var_list: list of tf.Variable, optional.
        List of TensorFlow variables to optimize over. Default is all
        trainable variables that `latent_vars` and `data` depend on,
        excluding those that are only used in conditionals in `data`.
      use_prettytensor: bool, optional.
        `True` if aim to use PrettyTensor optimizer (when using
        PrettyTensor) or `False` if aim to use TensorFlow optimizer.
        Defaults to TensorFlow.
      global_step: tf.Variable, optional.
        A TensorFlow variable to hold the global step.
    """
    super(PostNN, self).initialize(*args, **kwargs)




    with tf.name_scope('inference'):

      with tf.name_scope('variables'):
        self._cat_logits = tf.Variables(
          tf.zeros([self._params_space_dim, self._n_cats]),
          name='cat_logits')
        self._loc = tf.Variables(
          tf.random_normal([self._params_space_dim, self._n_cats]),
          name='locs')
        self._softplus_scale = tf.Variables(
          tf.zeros([self._params_space_dim, self._n_cats]),
          name='softplus_scales')

      with tf.name_scope('q_distribution'):

        mixture_distribution = Categorical(logits=self._cat_logits),
        components_distribution = NormalWithSoftplusScale(
          loc=self._loc, scale=self._softplus_scale)
        self._q = MixtureSameFamily(
          mixture_distribution, components_distribution)

    self.loss = self._build_loss()




    if optimizer is None and global_step is None:
      # Default optimizer always uses a global step variable.
      global_step = tf.Variable(0, trainable=False, name="global_step")

    if isinstance(global_step, tf.Variable):
      starter_learning_rate = 0.1
      learning_rate = tf.train.exponential_decay(starter_learning_rate,
                                                 global_step,
                                                 100, 0.9, staircase=True)
    else:
      learning_rate = 0.01

    # Build optimizer.
    if optimizer is None:
      optimizer = tf.train.AdamOptimizer(learning_rate)
    elif isinstance(optimizer, str):
      if optimizer == 'gradientdescent':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
      elif optimizer == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(learning_rate)
      elif optimizer == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(learning_rate)
      elif optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
      elif optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate)
      elif optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(learning_rate)
      elif optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
      else:
        raise ValueError('Optimizer class not found:', optimizer)
    elif not isinstance(optimizer, tf.train.Optimizer):
      raise TypeError("Optimizer must be str, tf.train.Optimizer, or None.")

    with tf.variable_scope(None, default_name="optimizer") as scope:
      self.train = optimizer.minimize(loss,
                                      global_step=global_step)

    self.reset.append(tf.variables_initializer(
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)))



  def _build_loss(self):
    r""" XXX


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
    distribution logits, and $q_i$ the component distribution. We have

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


    NOTE:
      Let `z` in `self.latent_vars`. Demand that the `z.name` and the name of
      `z` shall be the same, at least temporally.
    """

    # --- Construct `loss` ---


    # Get all samples in one go
    dict_samples = {}
    for z, qz in six.iteritems(self.latent_vars):
      z_samples = qz.sample(self.n_samples)
      dict_samples[z.name] = z_samples
      # To list ordered by `model_arg_names`
      samples = [dict_samples[_] for _ in model_arg_names]

    # Define likelihood (unvectorized for sampling)
    # (Temporally) assume that the data obeys a normal distribution,
    # realized by Gauss's limit-theorem
    def chi_square(model_output):
      chi_square_val = 0.0
      for y, y_data in six.iteritems(self.data):
        normal = NormalWithSoftplusScale(loc=y_data, scale=y.scale)
        chi_square_val += tf.reduce_mean(
            normal.log_prob(model_output) \
            * self.scale.get(y, 1.0) )
      return chi_square_val
    def log_likelihood(*model_args):
      return chi_square(self.model(*model_args))

    # Define prior (unvectorized for sampling)  # XXX: now here.
    def prior(*model_args):
      return sum([
          tf.reduce_sum(
              z.log_prob() \  # XXX
              * self.scale.get(z, 1.0) )
          for z in six.iterkeys(self.latent_vars) ])


      # Compute likelihood values  XXX: to delete
      params = {}
      for z, z_samples in six.iteritems(dict_samples):
        params[z.name] = z_samples[s]
      for y, y_data in six.iteritems(self.data):
        p_log_prob[s] += tf.reduce_mean(
            dict_chi_square[y](self.model(**params))
            * self.scale.get(y, 1.0)
        )



    log_posterior = log_likelihood + log_prior
    p_log_prob = tf.map_fn(log_posterior, samples)
    p_log_prob = tf.reduce_mean(p_log_prob)

    # Construct `q_log_prob` by analytic method
    q_log_prob = 0.0
    for z, qz in six.iteritems(self.latent_vars):
      try:
        q_log_prob += - tf.reduce_mean(qz.entropy())
      except NotImplementedError:
        q_log_prob += - tf.reduce_mean(qz.entropy_lower_bound())

    loss = - ( p_log_prob - q_log_prob )

    return loss


  @staticmethod
  def get_arg_names(fn):
    """ Get model argument names in order. """
    n_args = fn.__code__.co_argcount
    arg_names = fn.__code__.co_varnames[:n_args]
    return arg_names


  def _get_cats(self, cats):
    if cats isinstance(int):



  def _check_args(self, cats, model, param_dict):
    pass
